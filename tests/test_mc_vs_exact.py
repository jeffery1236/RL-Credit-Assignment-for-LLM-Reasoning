"""The project plan's Phase 2 "critical sanity test" (project_plan.md S15),
pulled forward to Phase 1 and run against a random policy instead of an
LLM: does exact DP Vpi agree with a large Monte Carlo rollout estimate?

If exact and MC estimates disagree outside a generous confidence interval,
the oracle itself is wrong and nothing downstream can be trusted -- so this
check must run and pass before Phase 2 begins.

Marked slow: run explicitly with `pytest -m slow`.
"""

import math
import random
from fractions import Fraction

import pytest

from env.countdown import is_terminal, allowed_actions, transition
from oracle.evaluation import ExactPolicyEvaluator, build_reasoning_dag
from policies.baselines import EpsilonGreedyDistancePolicy, UniformPolicy
from tests.test_countdown import make_state


def _rollout_once(root, policy, rng: random.Random) -> int:
    state = root
    while not is_terminal(state):
        actions = allowed_actions(state)
        probs = policy.action_probs(state, actions)
        weights = [float(probs[a]) for a in actions]
        action = rng.choices(actions, weights=weights, k=1)[0]
        state = transition(state, action)
    from env.countdown import reward

    return reward(state)


@pytest.mark.slow
@pytest.mark.parametrize("policy_factory", [UniformPolicy, EpsilonGreedyDistancePolicy])
def test_monte_carlo_matches_exact_within_confidence_interval(policy_factory):
    root = make_state([3, 7, 10, 25], 245)
    dag = build_reasoning_dag(root)
    policy = policy_factory()
    evaluator = ExactPolicyEvaluator(policy, dag)
    exact = float(evaluator.v_pi(root))

    rng = random.Random(12345)
    n_rollouts = 20_000
    successes = sum(_rollout_once(root, policy, rng) for _ in range(n_rollouts))
    estimate = successes / n_rollouts

    # 5-sigma binomial confidence interval around the exact probability.
    se = math.sqrt(exact * (1 - exact) / n_rollouts) if 0 < exact < 1 else 1e-9
    margin = 5 * se + 1e-6  # small floor to avoid a degenerate zero-width CI

    assert abs(estimate - exact) <= margin, (
        f"MC estimate {estimate:.4f} deviates from exact Vpi {exact:.4f} by "
        f"more than 5 sigma (margin={margin:.4f}) over {n_rollouts} rollouts "
        f"-- the oracle itself may be wrong"
    )
