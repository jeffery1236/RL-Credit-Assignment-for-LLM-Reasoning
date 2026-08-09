"""Exact enumeration and evaluation of the complete Countdown reasoning DAG.

This module builds the full state graph reachable from a root problem, with
transpositions (different action sequences reaching "the same" bag) merged
into a single node. It then computes, exactly:

  - V*(s), Q*(s, a): optimal value and optimal action values (binary,
    undiscounted, no step cost).
  - min_steps_to_success(s): shortest winning line length, computed by a
    SEPARATE recurrence from V* (see docs in that function).
  - Vpi(s), Qpi(s, a), Api(s, a): exact policy value, policy action value,
    and policy advantage for any stationary Policy[Fraction].
  - Exact on-policy state/action occupancy over the merged DAG.

Everything here uses fractions.Fraction for exact rational arithmetic --
there is no floating point anywhere in this module. This is what lets
invariants like "advantages sum to zero" be asserted as literal equality
in tests, rather than "close to zero within some tolerance."

CRITICAL CORRECTNESS NOTE (stationarity): the Vpi/Qpi/Api recurrences are
only the true win probability of the real sampling process if a policy's
action_probs(state, actions) depends ONLY on the canonical state and its
legal action set -- never on trajectory history, step count, or any other
hidden memory. Merging transpositions into one DAG node is only valid under
this assumption. See docs/project_plan.md and the Phase 2 Markov warning in
README.md for what this means for a future LLM policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Real
from typing import Mapping, Protocol, TypeVar

from env.countdown import (
    Action,
    DEFAULT_RULES,
    RuleSet,
    State,
    is_success,
    is_terminal,
    legal_actions,
    reward,
    transition,
)


class OracleError(Exception):
    """Base class for oracle/evaluation errors."""


class DagLimitExceeded(OracleError):
    """Raised when a DAG build exceeds its configured max_nodes/max_edges.

    An "exact" oracle must never silently truncate -- a partially-explored
    graph would produce a V* that looks exactly like a correct one but
    isn't. Callers must catch this and either raise the limit or reject
    the problem, never swallow it.
    """


class PolicyValidationError(OracleError):
    """A Policy returned probabilities that don't validate: wrong action
    set, negative probability, or a sum that isn't exactly 1.
    """


# ---------------------------------------------------------------------------
# DAG enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DagNode:
    """A single canonical state reached during DAG expansion. `depth` is
    relative to the DAG's own root: depth(root) == 0, and depth increases by
    exactly 1 along every edge (since every action removes exactly one
    number). Depth order is therefore automatically a topological order.
    """

    state: State
    depth: int


@dataclass(frozen=True, slots=True)
class DagEdge:
    """One legal action taken from `source`, landing on `target_state`.
    Two edges with the same (source, target_state) but different actions
    (e.g. 2+2 and 2x2 both landing on (4,)) are BOTH kept -- this is the
    multigraph property the (2,2) fixture tests.
    """

    source: State
    action: Action
    target_state: State


@dataclass(frozen=True, slots=True)
class ReasoningDag:
    root: State
    nodes: Mapping[State, DagNode]
    edges_by_source: Mapping[State, tuple[DagEdge, ...]]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.edges_by_source.values())

    def descendants(self) -> tuple[State, ...]:
        """All unique reachable states EXCLUDING the root."""
        return tuple(s for s in self.nodes if s != self.root)

    def topological_order(self) -> tuple[State, ...]:
        """Ascending-depth order, which is a valid topological order because
        depth strictly increases along every edge. Ties are broken by the
        state's values tuple for full determinism.
        """
        return tuple(
            sorted(
                self.nodes.keys(),
                key=lambda s: (self.nodes[s].depth, s.values),
            )
        )

    def out_edges(self, state: State) -> tuple[DagEdge, ...]:
        return self.edges_by_source.get(state, ())


def build_reasoning_dag(
    root: State,
    rules: RuleSet = DEFAULT_RULES,
    max_nodes: int = 100_000,
    max_edges: int = 1_000_000,
) -> ReasoningDag:
    """Breadth-first expansion of the complete reasoning DAG from `root`.

    Transpositions merge automatically: a state is only ever expanded once
    (the `nodes` dict check), regardless of how many different action
    sequences reach it. This is what keeps the graph polynomial-ish in size
    rather than exploding into a tree of factorial size.

    Raises DagLimitExceeded if the graph would exceed max_nodes or
    max_edges -- this must never be caught and silently treated as "good
    enough," since a truncated graph invalidates every V*/Q* computed over
    it.
    """
    nodes: dict[State, DagNode] = {root: DagNode(state=root, depth=0)}
    edges_by_source: dict[State, list[DagEdge]] = {}
    edge_count = 0

    worklist: list[State] = [root]
    seen_for_expansion: set[State] = set()

    while worklist:
        current = worklist.pop()
        if current in seen_for_expansion:
            continue
        seen_for_expansion.add(current)

        current_depth = nodes[current].depth
        actions = legal_actions(current, rules)
        if not actions:
            continue

        edges: list[DagEdge] = []
        for action in actions:
            child = transition(current, action, rules)
            edges.append(DagEdge(source=current, action=action, target_state=child))
            edge_count += 1
            if edge_count > max_edges:
                raise DagLimitExceeded(
                    f"Reasoning DAG exceeded max_edges={max_edges} while "
                    f"expanding root {root!r}"
                )

            if child not in nodes:
                nodes[child] = DagNode(state=child, depth=current_depth + 1)
                if len(nodes) > max_nodes:
                    raise DagLimitExceeded(
                        f"Reasoning DAG exceeded max_nodes={max_nodes} while "
                        f"expanding root {root!r}"
                    )
                worklist.append(child)

        edges_by_source[current] = edges

    return ReasoningDag(
        root=root,
        nodes=nodes,
        edges_by_source={s: tuple(es) for s, es in edges_by_source.items()},
    )
