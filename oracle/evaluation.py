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
  - Exact on-policy visit probabilities (state and action) over the merged
    DAG -- how often the policy actually reaches each state and takes each
    action, summed over every path that gets there.

Everything here uses fractions.Fraction for exact rational arithmetic --
there is no floating point anywhere in this module. This is what lets
invariants like "advantages sum to zero" be asserted as literal equality
in tests, rather than "close to zero within some tolerance."

CRITICAL CORRECTNESS NOTE (stationarity): the Vpi/Qpi/Api recurrences are
only the true win probability of the real sampling process if a policy's
action_probs(state, actions) depends ONLY on the canonical state and its
allowed action set -- never on trajectory history, step count, or any other
hidden memory. Merging transpositions into one DAG node is only valid under
this assumption. See docs/project_plan.md and the Phase 2 Markov warning in
README.md for what this means for a future LLM policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Mapping, Protocol, TypeVar

from env.countdown import (
    Action,
    DEFAULT_RULES,
    RuleSet,
    State,
    allowed_actions,
    is_success,
    is_terminal,
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
    """One allowed action taken from `source`, landing on `target_state`.
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
        actions = allowed_actions(current, rules)
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


# ---------------------------------------------------------------------------
# Optimal value: V*, Q*
# ---------------------------------------------------------------------------


def compute_v_star(
    dag: ReasoningDag, rules: RuleSet = DEFAULT_RULES
) -> dict[State, int]:
    """Exact optimal value for every state in `dag`, computed in a single
    reverse topological (deepest-first) pass over the DAG.

    V*(s) = 1                              if is_success(s)
          = 0                              if terminal and not success
          = max over legal a of V*(child)  otherwise

    Binary, undiscounted, no step cost, no tie-breaking -- there is nothing
    to break ties over since this returns a single scalar per state.
    """
    v_star: dict[State, int] = {}
    for state in reversed(dag.topological_order()):
        if is_success(state):
            v_star[state] = 1
        elif is_terminal(state, rules):
            v_star[state] = 0
        else:
            v_star[state] = max(
                v_star[edge.target_state] for edge in dag.out_edges(state)
            )
    return v_star


def compute_q_star(
    dag: ReasoningDag, v_star: Mapping[State, int]
) -> dict[State, dict[Action, int]]:
    """Q*(s, a) = V*(child(s, a)) for every allowed action at every
    non-terminal state. Returns the COMPLETE mapping per state, not just
    the best action -- fragility statistics need to count every
    irreversibly destructive action, not only identify the best move.
    """
    q_star: dict[State, dict[Action, int]] = {}
    for state, edges in dag.edges_by_source.items():
        q_star[state] = {edge.action: v_star[edge.target_state] for edge in edges}
    return q_star


def best_actions(
    state: State, q_star_for_state: Mapping[Action, int]
) -> tuple[Action, ...]:
    """All actions achieving the maximum Q* at `state` -- every maximizer
    is retained, never an arbitrary single pick.
    """
    if not q_star_for_state:
        return ()
    best_value = max(q_star_for_state.values())
    return tuple(
        sorted(
            (a for a, q in q_star_for_state.items() if q == best_value),
            key=Action.sort_key,
        )
    )


def is_pivotal(
    state: State,
    action: Action,
    v_star: Mapping[State, int],
    q_star_for_state: Mapping[Action, int],
) -> bool:
    """An action is "irreversibly destructive" (the project plan's pivotal/
    catastrophic decision) iff the state WAS winnable under optimal play
    and this action makes it unwinnable: V*(s) == 1 and Q*(s, a) == 0.

    This is a policy-INDEPENDENT, optimal-play concept. It must never be
    confused with policy-relative harm (Api(s, a) < 0), which is
    continuous and depends on a specific policy's behavior. See the module
    docstring and PolicyCreditProfile in analysis/profiles.py (Step 5).
    """
    return v_star[state] == 1 and q_star_for_state[action] == 0


def compute_min_steps_to_success(
    dag: ReasoningDag,
    v_star: Mapping[State, int],
    q_star: Mapping[State, Mapping[Action, int]],
) -> dict[State, int | None]:
    """Shortest winning line length from each state, computed by a
    recurrence SEPARATE from V*. V* is binary and says nothing about how
    fast a win can be reached; conflating the two recurrences risks
    quietly changing what V* means (e.g. by favoring shorter wins as if
    they were more "optimal" in the value sense, which they are not).

    min_steps_to_success(s) = 0      if is_success(s)
                             = None   if V*(s) == 0 (unwinnable)
                             = 1 + min over solution-preserving a
                                   (Q*(s,a) == 1) of
                                   min_steps_to_success(child(s, a))
                                otherwise
    """
    min_steps: dict[State, int | None] = {}
    for state in reversed(dag.topological_order()):
        if is_success(state):
            min_steps[state] = 0
        elif v_star[state] == 0:
            min_steps[state] = None
        else:
            candidates = [
                min_steps[edge.target_state]
                for edge in dag.out_edges(state)
                if q_star[state][edge.action] == 1
            ]
            # candidates cannot be empty: v_star[state] == 1 guarantees at
            # least one solution-preserving action exists.
            min_steps[state] = 1 + min(c for c in candidates if c is not None)
    return min_steps


# ---------------------------------------------------------------------------
# Policy protocol and exact policy evaluation: Vpi, Qpi, Api
# ---------------------------------------------------------------------------

P_co = TypeVar("P_co", bound=Real, covariant=True)


class Policy(Protocol[P_co]):
    """A stationary policy: probabilities depend ONLY on the canonical
    state and its allowed action set, never on trajectory history. This is
    generic over the probability type (Real) so that Phase 1's exact
    Fraction-based evaluation and a future Phase 2 float-based LLM policy
    share the same abstraction without any protocol changes.
    """

    def action_probs(
        self, state: State, actions: tuple[Action, ...]
    ) -> Mapping[Action, P_co]: ...


class ExactPolicyEvaluator:
    """Exact Vpi/Qpi/Api evaluation for a Policy[Fraction] over a
    ReasoningDag, using Fraction arithmetic throughout so that invariants
    (e.g. sum of advantages weighted by pi equals zero) hold as literal
    equality.

    The internal memo cache is scoped to THIS evaluator instance and is
    valid for exactly one (policy, dag, rules) combination. Never share an
    evaluator's cache across different policies -- construct a fresh
    evaluator per policy instead.
    """

    def __init__(
        self,
        policy: "Policy[Fraction]",
        dag: ReasoningDag,
        rules: RuleSet = DEFAULT_RULES,
    ) -> None:
        self._policy = policy
        self._dag = dag
        self._rules = rules
        self._v_pi_cache: dict[State, Fraction] = {}
        self._action_probs_cache: dict[State, Mapping[Action, Fraction]] = {}
        self._compute_all()

    def _get_action_probs(self, state: State) -> Mapping[Action, Fraction]:
        cached = self._action_probs_cache.get(state)
        if cached is not None:
            return cached

        actions = allowed_actions(state, self._rules)
        probs = self._policy.action_probs(state, actions)

        prob_keys = set(probs.keys())
        action_keys = set(actions)
        if prob_keys != action_keys:
            raise PolicyValidationError(
                f"Policy returned probabilities for {prob_keys} but allowed "
                f"actions at {state!r} are {action_keys}"
            )
        for action, exact_probability in probs.items():
            if exact_probability < 0:
                raise PolicyValidationError(
                    f"Policy returned negative probability {exact_probability} "
                    f"for action {action.render()} at state {state!r}"
                )
        total = sum(probs.values(), Fraction(0))
        if total != Fraction(1):
            raise PolicyValidationError(
                f"Policy probabilities at {state!r} sum to {total}, not "
                f"exactly 1 (probs={dict(probs)})"
            )

        self._action_probs_cache[state] = probs
        return probs

    def _compute_all(self) -> None:
        for state in reversed(self._dag.topological_order()):
            if is_success(state):
                self._v_pi_cache[state] = Fraction(1)
                continue
            if is_terminal(state, self._rules):
                self._v_pi_cache[state] = Fraction(0)
                continue

            probs = self._get_action_probs(state)
            win_probability = Fraction(0)
            for edge in self._dag.out_edges(state):
                action_probability = probs[edge.action]
                win_probability += action_probability * self._v_pi_cache[edge.target_state]
            self._v_pi_cache[state] = win_probability

    def v_pi(self, state: State) -> Fraction:
        return self._v_pi_cache[state]

    def q_pi(self, state: State, action: Action) -> Fraction:
        child = transition(state, action, self._rules)
        return self._v_pi_cache[child]

    def a_pi(self, state: State, action: Action) -> Fraction:
        return self.q_pi(state, action) - self.v_pi(state)

    def advantages(self, state: State) -> dict[Action, Fraction]:
        return {
            action: self.a_pi(state, action)
            for action in allowed_actions(state, self._rules)
        }

    def action_probs(self, state: State) -> Mapping[Action, Fraction]:
        return self._get_action_probs(state)


# ---------------------------------------------------------------------------
# Exact stationary-policy visit probabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VisitProbabilities:
    """How likely a stationary policy is to be at each state, and to take
    each action, summed over every path that reaches it. The root state's
    probability_of_reaching_state is always exactly 1.
    """

    probability_of_reaching_state: Mapping[State, Fraction]
    probability_of_taking_action: Mapping[tuple[State, Action], Fraction]


def compute_visit_probabilities(
    dag: ReasoningDag,
    evaluator: ExactPolicyEvaluator,
    rules: RuleSet = DEFAULT_RULES,
) -> VisitProbabilities:
    """Propagate probability mass forward in topological (depth-ascending)
    order. A state accumulates mass from ALL of its incoming action edges
    -- this summation over merged transpositions is the entire point of
    building a DAG instead of a tree.
    """
    probability_of_reaching_state: dict[State, Fraction] = {dag.root: Fraction(1)}
    probability_of_taking_action: dict[tuple[State, Action], Fraction] = {}

    for state in dag.topological_order():
        reaching_probability = probability_of_reaching_state.get(state, Fraction(0))
        if reaching_probability == 0:
            continue
        if is_terminal(state, rules):
            continue

        probs = evaluator.action_probs(state)
        for edge in dag.out_edges(state):
            action_probability = probs[edge.action]
            edge_mass = reaching_probability * action_probability
            probability_of_taking_action[(state, edge.action)] = edge_mass
            probability_of_reaching_state[edge.target_state] = (
                probability_of_reaching_state.get(edge.target_state, Fraction(0))
                + edge_mass
            )

    return VisitProbabilities(
        probability_of_reaching_state=probability_of_reaching_state,
        probability_of_taking_action=probability_of_taking_action,
    )
