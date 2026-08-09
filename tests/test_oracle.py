"""Tests for oracle/evaluation.py: DAG enumeration (Step 3) and exact
V*/Q*/Vpi/Qpi/Api oracles + occupancy (Step 4).
"""

from fractions import Fraction

import pytest

from env.countdown import Action, Operator, State, is_terminal, legal_actions, transition
from oracle.evaluation import (
    DagLimitExceeded,
    PolicyValidationError,
    build_reasoning_dag,
)


def make_state(values, target):
    return State(values=tuple(sorted(values)), target=target)


# ---------------------------------------------------------------------------
# DAG enumeration (Step 3)
# ---------------------------------------------------------------------------


def test_reachability_is_exact():
    root = make_state([3, 7, 10, 25], 245)
    dag = build_reasoning_dag(root)

    # Every node is reachable from root by following edges.
    reachable = {root}
    frontier = [root]
    while frontier:
        s = frontier.pop()
        for edge in dag.out_edges(s):
            if edge.target_state not in reachable:
                reachable.add(edge.target_state)
                frontier.append(edge.target_state)
    assert reachable == set(dag.nodes.keys())

    # Every state reached by following edges is present in nodes.
    for state, edges in dag.edges_by_source.items():
        assert state in dag.nodes
        for edge in edges:
            assert edge.target_state in dag.nodes


def test_action_completeness_matches_legal_actions():
    root = make_state([3, 7, 10, 25], 245)
    dag = build_reasoning_dag(root)
    for state in dag.nodes:
        edge_actions = {e.action for e in dag.out_edges(state)}
        assert edge_actions == set(legal_actions(state))


def test_parallel_edges_from_2_2_fixture():
    # (2, 2) -> target unreachable (99). Both 2+2 and 2x2 land on child (4,).
    root = make_state([2, 2], 99)
    dag = build_reasoning_dag(root)

    edges = dag.out_edges(root)
    assert len(edges) == 2  # ADD and MUL, no SUB (equal operands)
    targets = {e.target_state for e in edges}
    assert targets == {make_state([4], 99)}
    actions = {e.action for e in edges}
    assert actions == {Action(Operator.ADD, 2, 2), Action(Operator.MUL, 2, 2)}

    # The child appears exactly once in nodes, with two incoming edges.
    assert dag.node_count == 2  # root + child(4,)
    assert dag.edge_count == 2


def test_transposition_merging():
    # From (1, 2, 3): two different 2-step sequences can reach the same bag.
    #   1+2=3 -> (3, 3) -> 3+3=6 -> (6,)
    #   1x3=3? not directly comparable; use a cleaner transposition:
    # (1, 2, 4): order-independent combination
    #   step A: 1+2=3 -> (3, 4); step B: 3+4=7 -> (7,)
    #   step A': 2+... let's construct explicit merge via commutative ops.
    # Simplest reliable transposition: (1, 2, 3) with 1+2 then +3, vs 2+... :
    # Use (1, 1, 2): 1+1=2 -> (2, 2) -> 2+2=4 -> (4,)
    #           and: 1+2=3 -> (1, 3) -> 1+3=4 -> (4,)  [different path, same child]
    root = make_state([1, 1, 2], 999)
    dag = build_reasoning_dag(root)
    child_4 = make_state([4], 999)
    assert child_4 in dag.nodes

    incoming = [
        edge
        for edges in dag.edges_by_source.values()
        for edge in edges
        if edge.target_state == child_4
    ]
    assert len(incoming) >= 2
    sources = {edge.source for edge in incoming}
    assert len(sources) >= 2  # reached from at least 2 distinct intermediate states


def test_stable_ordering_across_builds():
    root = make_state([3, 7, 10, 25], 245)
    dag1 = build_reasoning_dag(root)
    dag2 = build_reasoning_dag(root)
    assert dag1.topological_order() == dag2.topological_order()
    for state in dag1.nodes:
        assert dag1.out_edges(state) == dag2.out_edges(state)


def test_terminal_leaves_have_no_outgoing_edges():
    root = make_state([3, 7, 10, 25], 245)
    dag = build_reasoning_dag(root)
    for state in dag.nodes:
        if is_terminal(state):
            assert dag.out_edges(state) == ()


def test_depth_and_cardinality_strictly_decrease_along_edges():
    root = make_state([3, 7, 10, 25], 245)
    dag = build_reasoning_dag(root)
    for state, edges in dag.edges_by_source.items():
        source_depth = dag.nodes[state].depth
        for edge in edges:
            target_depth = dag.nodes[edge.target_state].depth
            assert target_depth == source_depth + 1
            assert len(edge.target_state.values) == len(state.values) - 1


def test_dag_limit_exceeded_raises_rather_than_truncates():
    root = make_state([3, 7, 10, 25], 245)
    with pytest.raises(DagLimitExceeded):
        build_reasoning_dag(root, max_nodes=2)


def test_topological_order_is_depth_ascending():
    root = make_state([1, 1, 2], 999)
    dag = build_reasoning_dag(root)
    order = dag.topological_order()
    depths = [dag.nodes[s].depth for s in order]
    assert depths == sorted(depths)
