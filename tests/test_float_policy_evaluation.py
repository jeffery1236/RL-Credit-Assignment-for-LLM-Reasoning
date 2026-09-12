"""Tests for floating-point policy evaluation over a complete reasoning DAG."""

import math

import pytest

from env.countdown import Action, Operator, State, is_terminal
from oracle.evaluation import (
    ExactPolicyEvaluator,
    PolicyValidationError,
    build_reasoning_dag,
)
from oracle.float_policy_evaluation import (
    NUMERICAL_TOLERANCE,
    FloatPolicyEvaluator,
)
from policies.baselines import UniformPolicy


def make_state(values, target):
    return State(values=tuple(sorted(values)), target=target)


class FloatUniformPolicy:
    def action_probs(self, state, actions):
        probability = 1.0 / len(actions)
        return {action: probability for action in actions}


def test_hand_computed_uniform_policy_value_and_advantage():
    root = make_state([1, 2], 3)
    evaluator = FloatPolicyEvaluator(FloatUniformPolicy(), build_reasoning_dag(root))

    add = Action(Operator.ADD, 1, 2)
    multiply = Action(Operator.MUL, 1, 2)
    subtract = Action(Operator.SUB, 2, 1)

    assert evaluator.v_pi(root) == pytest.approx(1.0 / 3.0)
    assert evaluator.q_pi(root, add) == pytest.approx(1.0)
    assert evaluator.a_pi(root, add) == pytest.approx(2.0 / 3.0)
    assert evaluator.a_pi(root, multiply) == pytest.approx(-1.0 / 3.0)
    assert evaluator.a_pi(root, subtract) == pytest.approx(-1.0 / 3.0)


def test_advantages_have_zero_policy_weighted_mean():
    root = make_state([1, 2, 3], 10)
    dag = build_reasoning_dag(root)
    evaluator = FloatPolicyEvaluator(FloatUniformPolicy(), dag)

    for state in dag.nodes:
        probabilities = evaluator.action_probs(state)
        if not probabilities:
            continue
        weighted_mean = math.fsum(
            probabilities[action] * evaluator.a_pi(state, action)
            for action in probabilities
        )
        assert abs(weighted_mean) <= NUMERICAL_TOLERANCE


@pytest.mark.parametrize(
    "probability_factory",
    [
        lambda actions: {actions[0]: 1.0},
        lambda actions: {action: 0.1 for action in actions},
        lambda actions: {
            action: (-0.1 if index == 0 else 1.1 / (len(actions) - 1))
            for index, action in enumerate(actions)
        },
        lambda actions: {
            action: (math.nan if index == 0 else 0.0)
            for index, action in enumerate(actions)
        },
        lambda actions: {
            action: (math.inf if index == 0 else 0.0)
            for index, action in enumerate(actions)
        },
    ],
)
def test_invalid_float_policy_output_raises(probability_factory):
    class InvalidPolicy:
        def action_probs(self, state, actions):
            return probability_factory(actions)

    root = make_state([1, 2, 3], 10)
    with pytest.raises(PolicyValidationError):
        FloatPolicyEvaluator(InvalidPolicy(), build_reasoning_dag(root))


@pytest.mark.parametrize(
    "root",
    [
        make_state([1, 2], 3),
        make_state([1, 2, 3], 10),
        make_state([3, 7, 10, 25], 245),
    ],
)
def test_float_uniform_policy_matches_exact_evaluator(root):
    dag = build_reasoning_dag(root)
    exact = ExactPolicyEvaluator(UniformPolicy(), dag)
    numerical = FloatPolicyEvaluator(FloatUniformPolicy(), dag)

    for state in dag.nodes:
        assert float(exact.v_pi(state)) == pytest.approx(
            numerical.v_pi(state), abs=1e-12
        )
        if is_terminal(state):
            continue
        bellman_rhs = math.fsum(
            numerical.action_probs(state)[edge.action]
            * numerical.v_pi(edge.target_state)
            for edge in dag.out_edges(state)
        )
        assert abs(numerical.v_pi(state) - bellman_rhs) <= NUMERICAL_TOLERANCE
