"""Hypothesis property-based tests: generate random small Countdown problems
and assert the core oracle invariants hold for every baseline policy on
every reachable non-terminal state.

Kept to small state spaces (n_values <= 4, values <= 12) so DAG construction
stays fast enough for Hypothesis's many-examples default.
"""

from fractions import Fraction

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from env.countdown import is_terminal
from oracle.evaluation import ExactPolicyEvaluator, build_reasoning_dag, compute_v_star
from policies.baselines import EpsilonGreedyDistancePolicy, GreedyDistancePolicy, UniformPolicy
from tests.test_countdown import make_state

POLICIES = [UniformPolicy(), GreedyDistancePolicy(), EpsilonGreedyDistancePolicy()]

small_problems = st.builds(
    lambda values, target: make_state(values, target),
    values=st.lists(st.integers(min_value=1, max_value=12), min_size=2, max_size=4),
    target=st.integers(min_value=1, max_value=50),
)


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(root=small_problems)
def test_qpi_equals_vpi_of_child(root):
    dag = build_reasoning_dag(root, max_nodes=2000)
    for policy in POLICIES:
        evaluator = ExactPolicyEvaluator(policy, dag)
        for state in dag.nodes:
            if is_terminal(state):
                continue
            for edge in dag.out_edges(state):
                assert evaluator.q_pi(state, edge.action) == evaluator.v_pi(
                    edge.target_state
                )


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(root=small_problems)
def test_advantages_sum_to_zero_exactly(root):
    dag = build_reasoning_dag(root, max_nodes=2000)
    for policy in POLICIES:
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


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(root=small_problems)
def test_v_pi_never_exceeds_v_star(root):
    dag = build_reasoning_dag(root, max_nodes=2000)
    v_star = compute_v_star(dag)
    for policy in POLICIES:
        evaluator = ExactPolicyEvaluator(policy, dag)
        for state in dag.nodes:
            assert evaluator.v_pi(state) <= Fraction(v_star[state])
