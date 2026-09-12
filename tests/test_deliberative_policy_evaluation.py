"""Tests for sampled deliberative policies with explicit invalid-output mass."""

import pytest

from env.countdown import Action, Operator, State
from oracle.deliberative_policy_evaluation import DeliberativePolicyEvaluator
from oracle.evaluation import build_reasoning_dag


def test_invalid_probability_contributes_terminal_failure_to_value():
    root = State(values=(1, 2), target=3)
    add = Action(Operator.ADD, 1, 2)
    multiply = Action(Operator.MUL, 1, 2)
    subtract = Action(Operator.SUB, 2, 1)
    policy = {
        root: {
            "action_probabilities": {add: 0.5, multiply: 0.0, subtract: 0.0},
            "invalid_probability": 0.5,
        }
    }

    evaluator = DeliberativePolicyEvaluator(policy, build_reasoning_dag(root))

    assert evaluator.v_pi(root) == pytest.approx(0.5)
    assert evaluator.q_pi(root, add) == pytest.approx(1.0)
    assert evaluator.a_pi(root, add) == pytest.approx(0.5)
    assert evaluator.invalid_output_advantage(root) == pytest.approx(-0.5)
