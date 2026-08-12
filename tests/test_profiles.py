"""Tests for analysis/profiles.py: OptimalActionProfile, PolicyCreditProfile,
and ProblemProfile.
"""

from fractions import Fraction

import pytest

from env.countdown import Action, Operator, State
from analysis.profiles import (
    compute_occupancy_weighted_advantage_summary,
    compute_optimal_action_profile,
    compute_policy_credit_profile,
    compute_problem_profile,
    has_at_least_one_ruining_move,
)
from oracle.evaluation import (
    ExactPolicyEvaluator,
    build_reasoning_dag,
    compute_min_steps_to_success,
    compute_q_star,
    compute_v_star,
    compute_visit_probabilities,
)
from policies.baselines import UniformPolicy


def make_state(values, target):
    return State(values=tuple(sorted(values)), target=target)


# ---------------------------------------------------------------------------
# has_at_least_one_ruining_move
# ---------------------------------------------------------------------------


def test_has_at_least_one_ruining_move_true_when_winnable_with_a_bad_option():
    state = make_state([1, 2], 999)
    safe_action = Action(Operator.ADD, 1, 2)
    ruin_action = Action(Operator.MUL, 1, 2)
    q_star_for_state = {safe_action: 1, ruin_action: 0}

    assert has_at_least_one_ruining_move(state, {state: 1}, q_star_for_state)
    assert not has_at_least_one_ruining_move(state, {state: 0}, q_star_for_state)


def test_has_at_least_one_ruining_move_false_when_all_actions_safe():
    state = make_state([1, 2], 999)
    a = Action(Operator.ADD, 1, 2)
    b = Action(Operator.MUL, 1, 2)
    q_star_for_state = {a: 1, b: 1}
    assert not has_at_least_one_ruining_move(state, {state: 1}, q_star_for_state)


# ---------------------------------------------------------------------------
# OptimalActionProfile: fragility discrimination (hand-constructed, per the
# plan's "1 of 15 safe" vs "14 of 15 safe" discrimination test)
# ---------------------------------------------------------------------------


def test_fragility_extremes_discriminate():
    state = make_state([1, 2], 999)  # state identity only; actions below are synthetic
    actions = tuple(Action(Operator.ADD, i, i + 1) for i in range(1, 16))
    assert len(actions) == 15

    v_star = {state: 1}

    # Only 1 of 15 actions preserves the solution -> highly fragile.
    q_star_fragile = {a: (1 if i == 0 else 0) for i, a in enumerate(actions)}
    profile_fragile = compute_optimal_action_profile(
        state, v_star, {state: q_star_fragile}, {state: None}
    )
    assert profile_fragile.fraction_of_moves_that_ruin_solution == pytest.approx(14 / 15)
    assert profile_fragile.solution_preserving_count == 1
    assert len(profile_fragile.ruining_actions) == 14

    # 14 of 15 actions preserve the solution -> barely fragile.
    q_star_safe = {a: (0 if i == 0 else 1) for i, a in enumerate(actions)}
    profile_safe = compute_optimal_action_profile(
        state, v_star, {state: q_star_safe}, {state: None}
    )
    assert profile_safe.fraction_of_moves_that_ruin_solution == pytest.approx(1 / 15)
    assert profile_safe.solution_preserving_count == 14
    assert len(profile_safe.ruining_actions) == 1

    assert (
        profile_fragile.fraction_of_moves_that_ruin_solution
        > profile_safe.fraction_of_moves_that_ruin_solution
    )


def test_fragility_none_for_terminal_and_unsolvable_states():
    root = make_state([1, 2], 3)
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)

    win_state = make_state([3], 3)  # terminal (success)
    profile = compute_optimal_action_profile(win_state, v_star, q_star, min_steps)
    assert profile.allowed_action_count == 0
    assert profile.fraction_of_moves_that_ruin_solution is None
    assert profile.min_steps_to_success == 0

    dead_state = make_state([2], 3)  # terminal (failure)
    profile = compute_optimal_action_profile(dead_state, v_star, q_star, min_steps)
    assert profile.fraction_of_moves_that_ruin_solution is None
    assert profile.min_steps_to_success is None


def test_optimal_action_profile_on_real_pivotal_fixture():
    # (2, 3, 5) -> 30: 3+5=8 is irreversibly destructive (see test_oracle.py).
    root = make_state([2, 3, 5], 30)
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)

    profile = compute_optimal_action_profile(root, v_star, q_star, min_steps)
    destructive_action = Action(Operator.ADD, 3, 5)
    assert destructive_action in profile.ruining_actions
    assert profile.fraction_of_moves_that_ruin_solution is not None
    assert has_at_least_one_ruining_move(root, v_star, q_star[root])


# ---------------------------------------------------------------------------
# PolicyCreditProfile
# ---------------------------------------------------------------------------


def test_policy_credit_profile_hand_computed():
    # (1, 2) -> 3, uniform policy: Vpi(root) = 1/3.
    root = make_state([1, 2], 3)
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(UniformPolicy(), dag)

    profile = compute_policy_credit_profile(root, evaluator)

    add = Action(Operator.ADD, 1, 2)
    mul = Action(Operator.MUL, 1, 2)
    sub = Action(Operator.SUB, 2, 1)

    assert profile.action_to_advantage[add] == Fraction(2, 3)
    assert profile.action_to_advantage[mul] == Fraction(-1, 3)
    assert profile.action_to_advantage[sub] == Fraction(-1, 3)

    assert profile.positive_count == 1
    assert profile.negative_count == 2
    assert profile.zero_count == 0
    assert profile.mean_abs_advantage == Fraction(4, 9)
    assert profile.max_abs_advantage == Fraction(2, 3)


def test_policy_credit_profile_empty_for_terminal_state():
    root = make_state([1, 2], 3)
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(UniformPolicy(), dag)

    win_state = make_state([3], 3)
    profile = compute_policy_credit_profile(win_state, evaluator)

    assert profile.action_to_advantage == {}
    assert profile.action_probs == {}
    assert profile.negative_count == 0
    assert profile.zero_count == 0
    assert profile.positive_count == 0
    assert profile.mean_abs_advantage == Fraction(0)
    assert profile.max_abs_advantage == Fraction(0)


# ---------------------------------------------------------------------------
# ProblemProfile
# ---------------------------------------------------------------------------


def test_problem_profile_end_to_end():
    root = make_state([1, 2], 3)
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)
    evaluators = {"uniform": ExactPolicyEvaluator(UniformPolicy(), dag)}

    profile = compute_problem_profile(dag, v_star, q_star, min_steps, evaluators)

    assert profile.solvable is True
    assert profile.initially_solved is False
    assert profile.node_count == dag.node_count
    assert profile.edge_count == dag.edge_count
    assert profile.min_solution_depth == 1
    assert profile.solution_preserving_root_action_count == 1
    assert profile.branching is not None
    assert profile.branching.maximum >= profile.branching.minimum
    assert "uniform" in profile.policy_advantage_summaries
    summary = profile.policy_advantage_summaries["uniform"]
    assert summary.positive_count + summary.negative_count + summary.zero_count > 0


def test_problem_profile_initially_solved_flag():
    root = make_state([1, 3], 3)  # target already present in the initial bag
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)
    profile = compute_problem_profile(dag, v_star, q_star, min_steps, evaluators={})

    assert profile.initially_solved is True
    assert profile.solvable is True
    assert profile.node_count == 1  # root is terminal (already a success)
    assert profile.branching is None


def test_occupancy_weighted_advantage_summary_hand_computed():
    # (1, 2) -> 3, uniform policy: each of the 3 root actions gets exactly
    # 1/3 visit mass. add is positive (2/3), mul and sub are both negative
    # (-1/3 each) -- so occupancy-weighted mass should be 2/3 negative,
    # 1/3 positive, 0 zero.
    root = make_state([1, 2], 3)
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(UniformPolicy(), dag)
    visit_probs = compute_visit_probabilities(dag, evaluator)

    summary = compute_occupancy_weighted_advantage_summary([root], evaluator, visit_probs)

    assert summary.negative_mass == pytest.approx(2 / 3)
    assert summary.positive_mass == pytest.approx(1 / 3)
    assert summary.zero_mass == pytest.approx(0.0)
    assert summary.mean_abs_advantage == pytest.approx(4 / 9)


def test_problem_profile_includes_occupancy_weighted_summary_when_provided():
    root = make_state([1, 2], 3)
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)
    evaluator = ExactPolicyEvaluator(UniformPolicy(), dag)
    evaluators = {"uniform": evaluator}
    visit_probabilities = {"uniform": compute_visit_probabilities(dag, evaluator)}

    profile = compute_problem_profile(
        dag, v_star, q_star, min_steps, evaluators, visit_probabilities=visit_probabilities
    )

    assert "uniform" in profile.occupancy_weighted_advantage_summaries
    summary = profile.occupancy_weighted_advantage_summaries["uniform"]
    assert summary.negative_mass == pytest.approx(2 / 3)


def test_problem_profile_occupancy_weighted_summary_empty_when_not_requested():
    root = make_state([1, 2], 3)
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)
    evaluators = {"uniform": ExactPolicyEvaluator(UniformPolicy(), dag)}

    profile = compute_problem_profile(dag, v_star, q_star, min_steps, evaluators)
    assert profile.occupancy_weighted_advantage_summaries == {}


def test_problem_profile_unsolvable_root():
    root = make_state([2, 8], 30)  # dead end, established in test_oracle.py
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)
    profile = compute_problem_profile(dag, v_star, q_star, min_steps, evaluators={})

    assert profile.solvable is False
    assert profile.min_solution_depth is None
    assert profile.ruining_move_state_count == 0
    assert profile.ruining_move_state_rate is None
