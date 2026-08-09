"""Tests for policies/baselines.py: the three state-only, oracle-free
baseline policies (Uniform, GreedyDistance, EpsilonGreedyDistance).
"""

from fractions import Fraction

import pytest

from env.countdown import Action, Operator, State, allowed_actions
from policies.baselines import EpsilonGreedyDistancePolicy, GreedyDistancePolicy, UniformPolicy


def make_state(values, target):
    return State(values=tuple(sorted(values)), target=target)


def test_uniform_policy_sums_to_one_and_is_equal():
    s = make_state([3, 7, 10, 25], 245)
    actions = allowed_actions(s)
    probs = UniformPolicy().action_probs(s, actions)
    assert sum(probs.values(), Fraction(0)) == Fraction(1)
    values = set(probs.values())
    assert values == {Fraction(1, len(actions))}


def test_greedy_worked_example_prefers_25_times_10():
    # From the plan: (3,7,10,25) -> 245.
    #   25*10=250 -> (3,7,250): distances 242,238,5 -> score 5
    #   25+10=35  -> (3,7,35):  distances 242,238,210 -> score 210
    #   25-10=15  -> (3,7,15):  distances ... -> score 230
    s = make_state([3, 7, 10, 25], 245)
    actions = allowed_actions(s)
    probs = GreedyDistancePolicy().action_probs(s, actions)

    # Hand-verified: mul(10,25)=250 has score 5, the unique minimum across
    # all 18 allowed actions from this state (next best is mul(7,25)=175,
    # score 70) -- so it gets ALL the probability mass, not just some.
    mul_25_10 = Action(Operator.MUL, 10, 25)
    assert probs[mul_25_10] == Fraction(1)

    add_25_10 = Action(Operator.ADD, 10, 25)
    sub_25_10 = Action(Operator.SUB, 25, 10)
    assert probs[add_25_10] == Fraction(0)
    assert probs[sub_25_10] == Fraction(0)


def test_greedy_wins_immediately_when_available():
    # (1, 2) -> 3: 1+2=3 wins immediately, distance-to-target score 0.
    s = make_state([1, 2], 3)
    actions = allowed_actions(s)
    probs = GreedyDistancePolicy().action_probs(s, actions)
    winning_action = Action(Operator.ADD, 1, 2)
    assert probs[winning_action] == Fraction(1)


def test_greedy_splits_genuine_ties_uniformly():
    # Hand-verified tie: (1, 1, 5) -> target 3.
    #   1+1=2 -> (2,5): distances 1,2 -> score 1
    #   5-1=4 -> (1,4): distances 2,1 -> score 1   <- ties with 1+1
    #   1*1=1 -> (1,5): distances 2,2 -> score 2
    #   1+5=6 -> (1,6): distances 2,3 -> score 2
    #   1*5=5 -> (1,5): distances 2,2 -> score 2
    # Two actions tie at the minimum score of 1; each must get exactly 1/2.
    s = make_state([1, 1, 5], 3)
    actions = allowed_actions(s)
    probs = GreedyDistancePolicy().action_probs(s, actions)

    winner_a = Action(Operator.ADD, 1, 1)
    winner_b = Action(Operator.SUB, 5, 1)
    assert probs[winner_a] == Fraction(1, 2)
    assert probs[winner_b] == Fraction(1, 2)

    for a in actions:
        if a not in (winner_a, winner_b):
            assert probs[a] == Fraction(0)
    assert sum(probs.values(), Fraction(0)) == Fraction(1)


def test_epsilon_greedy_sums_to_one():
    s = make_state([3, 7, 10, 25], 245)
    actions = allowed_actions(s)
    policy = EpsilonGreedyDistancePolicy(epsilon=Fraction(1, 10))
    probs = policy.action_probs(s, actions)
    assert sum(probs.values(), Fraction(0)) == Fraction(1)


def test_epsilon_greedy_composition_matches_formula():
    s = make_state([3, 7, 10, 25], 245)
    actions = allowed_actions(s)
    eps = Fraction(1, 10)
    policy = EpsilonGreedyDistancePolicy(epsilon=eps)
    probs = policy.action_probs(s, actions)

    greedy_probs = GreedyDistancePolicy().action_probs(s, actions)
    n = len(actions)
    for a in actions:
        expected = (Fraction(1) - eps) * greedy_probs[a] + eps * Fraction(1, n)
        assert probs[a] == expected


def test_epsilon_greedy_rejects_out_of_range_epsilon():
    with pytest.raises(ValueError):
        EpsilonGreedyDistancePolicy(epsilon=Fraction(3, 2))
    with pytest.raises(ValueError):
        EpsilonGreedyDistancePolicy(epsilon=Fraction(-1, 10))


# ---------------------------------------------------------------------------
# Hand-computed fixture from the plan: (1, 2) -> target 3
# ---------------------------------------------------------------------------


def test_hand_computed_advantage_fixture():
    from oracle.evaluation import ExactPolicyEvaluator, build_reasoning_dag, compute_v_star

    root = make_state([1, 2], 3)
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    assert v_star[root] == 1

    evaluator = ExactPolicyEvaluator(UniformPolicy(), dag)
    assert evaluator.v_pi(root) == Fraction(1, 3)

    add = Action(Operator.ADD, 1, 2)
    assert evaluator.a_pi(root, add) == Fraction(2, 3)
