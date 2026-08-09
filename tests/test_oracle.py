"""Tests for oracle/evaluation.py: DAG enumeration (Step 3) and exact
V*/Q*/Vpi/Qpi/Api oracles + visit_probabilities (Step 4).
"""

from fractions import Fraction

import pytest

from env.countdown import Action, Operator, State, is_terminal, allowed_actions
from oracle.evaluation import (
    DagLimitExceeded,
    ExactPolicyEvaluator,
    PolicyValidationError,
    build_reasoning_dag,
    compute_min_steps_to_success,
    compute_visit_probabilities,
    compute_q_star,
    compute_v_star,
    is_pivotal,
)
from policies.baselines import EpsilonGreedyDistancePolicy, GreedyDistancePolicy, UniformPolicy
from tests.reference_solver import (
    brute_force_min_steps_to_success,
    brute_force_v_pi,
    brute_force_v_star,
    enumerate_successful_paths,
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


def test_action_completeness_matches_allowed_actions():
    root = make_state([3, 7, 10, 25], 245)
    dag = build_reasoning_dag(root)
    for state in dag.nodes:
        edge_actions = {e.action for e in dag.out_edges(state)}
        assert edge_actions == set(allowed_actions(state))


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


# ---------------------------------------------------------------------------
# V*, Q*, min_steps_to_success (Step 4)
# ---------------------------------------------------------------------------


SMALL_ROOTS = [
    make_state([1, 2], 3),
    make_state([1, 2, 3], 10),
    make_state([2, 3, 5], 30),
    make_state([1, 2, 3, 4], 24),
    make_state([3, 7, 10, 25], 245),
]


@pytest.mark.parametrize("root", SMALL_ROOTS)
def test_v_star_matches_independent_brute_force(root):
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    for state in dag.nodes:
        assert v_star[state] == brute_force_v_star(state)


@pytest.mark.parametrize("root", SMALL_ROOTS)
def test_min_steps_to_success_matches_independent_brute_force(root):
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)
    for state in dag.nodes:
        assert min_steps[state] == brute_force_min_steps_to_success(state)


def test_min_steps_to_success_agrees_with_shortest_enumerated_path():
    root = make_state([1, 2, 3, 4], 24)
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)

    paths = enumerate_successful_paths(root)
    assert paths, "expected at least one successful path for this fixture"
    shortest = min(len(p) for p in paths)
    assert min_steps[root] == shortest


def test_q_star_is_v_star_of_child():
    root = make_state([3, 7, 10, 25], 245)
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    for state, edges in dag.edges_by_source.items():
        for edge in edges:
            assert q_star[state][edge.action] == v_star[edge.target_state]


def test_is_pivotal_hand_computed():
    # (2, 3, 5), target 30: 2*3=6 -> (5,6); 5*6=30 win. So root is winnable.
    # Some other root action must destroy that possibility, e.g. 3+5=8 -> (2,8):
    #   2+8=10, 2*8=16, 8-2=6: none reach 30. So this action is irreversibly
    #   destructive IF root V*==1 (it is, via the other branch).
    root = make_state([2, 3, 5], 30)
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    assert v_star[root] == 1

    destructive_action = Action(Operator.ADD, 3, 5)  # 3+5=8 -> (2, 8), a dead end
    assert is_pivotal(root, destructive_action, v_star, q_star[root])

    safe_action = Action(Operator.MUL, 2, 3)  # 2*3=6 -> (5, 6), preserves win
    assert not is_pivotal(root, safe_action, v_star, q_star[root])


def test_dead_end_from_2_8():
    # Confirms the (2, 8) branch used above really is a dead end for target 30.
    root = make_state([2, 8], 30)
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    assert v_star[root] == 0


# ---------------------------------------------------------------------------
# Vpi, Qpi, Api (Step 4)
# ---------------------------------------------------------------------------


def test_hand_computed_uniform_policy_value():
    # (1, 2) -> target 3. Legal actions: 1+2=3 (win), 1*2=2 (dead), 2-1=1 (dead).
    root = make_state([1, 2], 3)
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(UniformPolicy(), dag)

    assert evaluator.v_pi(root) == Fraction(1, 3)

    add = Action(Operator.ADD, 1, 2)
    mul = Action(Operator.MUL, 1, 2)
    sub = Action(Operator.SUB, 2, 1)

    assert evaluator.q_pi(root, add) == Fraction(1)
    assert evaluator.q_pi(root, mul) == Fraction(0)
    assert evaluator.q_pi(root, sub) == Fraction(0)

    assert evaluator.a_pi(root, add) == Fraction(1) - Fraction(1, 3)
    assert evaluator.a_pi(root, mul) == Fraction(0) - Fraction(1, 3)
    assert evaluator.a_pi(root, sub) == Fraction(0) - Fraction(1, 3)


def test_policy_validation_rejects_wrong_action_set():
    class BadPolicy:
        def action_probs(self, state, actions):
            return {actions[0]: Fraction(1)}  # missing other actions

    root = make_state([1, 2, 3], 10)
    dag = build_reasoning_dag(root)
    with pytest.raises(PolicyValidationError):
        ExactPolicyEvaluator(BadPolicy(), dag)


def test_policy_validation_rejects_non_unit_sum():
    class BadPolicy:
        def action_probs(self, state, actions):
            n = len(actions)
            return {a: Fraction(1, n + 1) for a in actions}  # sums to n/(n+1) != 1

    root = make_state([1, 2, 3], 10)
    dag = build_reasoning_dag(root)
    with pytest.raises(PolicyValidationError):
        ExactPolicyEvaluator(BadPolicy(), dag)


def test_policy_validation_rejects_negative_probability():
    class BadPolicy:
        def action_probs(self, state, actions):
            probs = {a: Fraction(0) for a in actions}
            probs[actions[0]] = Fraction(-1, 2)
            probs[actions[-1]] = Fraction(3, 2)
            return probs

    root = make_state([1, 2, 3], 10)
    dag = build_reasoning_dag(root)
    with pytest.raises(PolicyValidationError):
        ExactPolicyEvaluator(BadPolicy(), dag)


@pytest.mark.parametrize("root", SMALL_ROOTS)
@pytest.mark.parametrize(
    "policy_factory",
    [UniformPolicy, GreedyDistancePolicy, EpsilonGreedyDistancePolicy],
)
def test_v_pi_matches_independent_brute_force(root, policy_factory):
    policy = policy_factory()
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(policy, dag)
    for state in dag.nodes:
        assert evaluator.v_pi(state) == brute_force_v_pi(state, policy)


@pytest.mark.parametrize("root", SMALL_ROOTS)
@pytest.mark.parametrize(
    "policy_factory",
    [UniformPolicy, GreedyDistancePolicy, EpsilonGreedyDistancePolicy],
)
def test_bellman_residual_is_exactly_zero(root, policy_factory):
    policy = policy_factory()
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(policy, dag)
    for state in dag.nodes:
        if is_terminal(state):
            continue
        probs = evaluator.action_probs(state)
        rhs = sum(
            (probs[edge.action] * evaluator.v_pi(edge.target_state) for edge in dag.out_edges(state)),
            Fraction(0),
        )
        assert evaluator.v_pi(state) - rhs == Fraction(0)


@pytest.mark.parametrize("root", SMALL_ROOTS)
@pytest.mark.parametrize(
    "policy_factory",
    [UniformPolicy, GreedyDistancePolicy, EpsilonGreedyDistancePolicy],
)
def test_advantages_sum_to_exactly_zero(root, policy_factory):
    policy = policy_factory()
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(policy, dag)
    for state in dag.nodes:
        if is_terminal(state):
            continue
        probs = evaluator.action_probs(state)
        advantages = evaluator.advantages(state)
        weighted_sum = sum(
            (probs[a] * advantages[a] for a in advantages), Fraction(0)
        )
        assert weighted_sum == Fraction(0)


@pytest.mark.parametrize("root", SMALL_ROOTS)
@pytest.mark.parametrize(
    "policy_factory",
    [UniformPolicy, GreedyDistancePolicy, EpsilonGreedyDistancePolicy],
)
def test_v_pi_never_exceeds_v_star(root, policy_factory):
    policy = policy_factory()
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    evaluator = ExactPolicyEvaluator(policy, dag)
    for state in dag.nodes:
        assert evaluator.v_pi(state) <= Fraction(v_star[state])


# ---------------------------------------------------------------------------
# VisitProbabilities (Step 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("root", SMALL_ROOTS)
@pytest.mark.parametrize(
    "policy_factory",
    [UniformPolicy, GreedyDistancePolicy, EpsilonGreedyDistancePolicy],
)
def test_root_occupancy_is_one(root, policy_factory):
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(policy_factory(), dag)
    visit_probabilities = compute_visit_probabilities(dag, evaluator)
    assert visit_probabilities.probability_of_reaching_state[root] == Fraction(1)


@pytest.mark.parametrize("root", SMALL_ROOTS)
@pytest.mark.parametrize(
    "policy_factory",
    [UniformPolicy, GreedyDistancePolicy, EpsilonGreedyDistancePolicy],
)
def test_outgoing_action_mass_equals_state_mass(root, policy_factory):
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(policy_factory(), dag)
    visit_probabilities = compute_visit_probabilities(dag, evaluator)
    for state in dag.nodes:
        if is_terminal(state):
            continue
        state_mass = visit_probabilities.probability_of_reaching_state.get(state, Fraction(0))
        outgoing = sum(
            (
                visit_probabilities.probability_of_taking_action.get((state, edge.action), Fraction(0))
                for edge in dag.out_edges(state)
            ),
            Fraction(0),
        )
        assert outgoing == state_mass


@pytest.mark.parametrize("root", SMALL_ROOTS)
@pytest.mark.parametrize(
    "policy_factory",
    [UniformPolicy, GreedyDistancePolicy, EpsilonGreedyDistancePolicy],
)
def test_incoming_action_mass_equals_state_mass_for_non_root(root, policy_factory):
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(policy_factory(), dag)
    visit_probabilities = compute_visit_probabilities(dag, evaluator)

    incoming_by_target: dict[State, Fraction] = {}
    for state, edges in dag.edges_by_source.items():
        for edge in edges:
            key = (state, edge.action)
            mass = visit_probabilities.probability_of_taking_action.get(key, Fraction(0))
            incoming_by_target[edge.target_state] = (
                incoming_by_target.get(edge.target_state, Fraction(0)) + mass
            )

    for state in dag.nodes:
        if state == root:
            continue
        expected = visit_probabilities.probability_of_reaching_state.get(state, Fraction(0))
        assert incoming_by_target.get(state, Fraction(0)) == expected


@pytest.mark.parametrize("root", SMALL_ROOTS)
@pytest.mark.parametrize(
    "policy_factory",
    [UniformPolicy, GreedyDistancePolicy, EpsilonGreedyDistancePolicy],
)
def test_terminal_occupancy_sums_to_one(root, policy_factory):
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(policy_factory(), dag)
    visit_probabilities = compute_visit_probabilities(dag, evaluator)
    total = sum(
        (
            visit_probabilities.probability_of_reaching_state.get(s, Fraction(0))
            for s in dag.nodes
            if is_terminal(s)
        ),
        Fraction(0),
    )
    assert total == Fraction(1)


@pytest.mark.parametrize("root", SMALL_ROOTS)
@pytest.mark.parametrize(
    "policy_factory",
    [UniformPolicy, GreedyDistancePolicy, EpsilonGreedyDistancePolicy],
)
def test_occupancy_weighted_reward_equals_v_pi_root(root, policy_factory):
    from env.countdown import reward as env_reward

    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(policy_factory(), dag)
    visit_probabilities = compute_visit_probabilities(dag, evaluator)
    total = sum(
        (
            visit_probabilities.probability_of_reaching_state.get(s, Fraction(0)) * env_reward(s)
            for s in dag.nodes
            if is_terminal(s)
        ),
        Fraction(0),
    )
    assert total == evaluator.v_pi(root)


def test_occupancy_transposition_fixture_sums_both_paths():
    # (1, 1, 2) -> child (4,) reachable via two distinct intermediate states,
    # as established in test_transposition_merging above.
    root = make_state([1, 1, 2], 999)
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(UniformPolicy(), dag)
    visit_probabilities = compute_visit_probabilities(dag, evaluator)

    child_4 = make_state([4], 999)
    incoming = [
        (edge.source, edge.action)
        for edges in dag.edges_by_source.values()
        for edge in edges
        if edge.target_state == child_4
    ]
    assert len(incoming) >= 2
    expected = sum(
        (visit_probabilities.probability_of_taking_action[(s, a)] for s, a in incoming), Fraction(0)
    )
    assert visit_probabilities.probability_of_reaching_state[child_4] == expected
