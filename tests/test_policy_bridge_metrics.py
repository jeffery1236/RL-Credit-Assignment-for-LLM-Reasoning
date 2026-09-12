"""Tests for deterministic policy-bridge analysis helpers."""

import pytest

from env.countdown import Action, Operator
from analysis.policy_bridge_metrics import (
    BridgeStageAGate,
    average_action_distributions,
    bootstrap_response_counts,
    select_stage_b_roots,
    solution_preserving_probability,
    spearman_rank_correlation,
    total_variation_distance,
)


ADD = Action(Operator.ADD, 1, 2)
MUL = Action(Operator.MUL, 1, 2)
SUB = Action(Operator.SUB, 2, 1)


def test_menu_averaging_matches_semantic_actions_not_labels():
    averaged = average_action_distributions(
        [
            {ADD: 0.7, MUL: 0.2, SUB: 0.1},
            {MUL: 0.4, ADD: 0.3, SUB: 0.3},
        ]
    )

    assert averaged == pytest.approx({ADD: 0.5, MUL: 0.3, SUB: 0.2})


def test_menu_averaging_rejects_mismatched_actions():
    with pytest.raises(ValueError, match="same legal action set"):
        average_action_distributions([{ADD: 1.0}, {MUL: 1.0}])


def test_total_variation_and_tie_aware_spearman():
    assert total_variation_distance(
        {ADD: 0.6, MUL: 0.4}, {ADD: 0.2, MUL: 0.8}
    ) == pytest.approx(0.4)
    assert spearman_rank_correlation([1, 2, 2, 4], [3, 4, 4, 9]) == pytest.approx(
        1.0
    )
    assert spearman_rank_correlation([1, 2, 2, 4], [4, 3, 3, 1]) == pytest.approx(
        -1.0
    )


def test_solution_preserving_probability_uses_exact_q_star():
    probability = solution_preserving_probability(
        {ADD: 0.4, MUL: 0.5, SUB: 0.1}, {ADD: 1, MUL: 0, SUB: 1}
    )

    assert probability == pytest.approx(0.5)


def test_stage_a_gate_includes_threshold_boundaries_and_missing_correlations():
    gate = BridgeStageAGate(
        minimum_valid_rate=0.9,
        minimum_median_spearman=0.5,
        maximum_median_tv=0.35,
    )

    passed = gate.evaluate(valid_rate=0.9, spearman_values=[0.5], tv_values=[0.35])
    failed = gate.evaluate(valid_rate=0.9, spearman_values=[None], tv_values=[0.35])
    missing_state = gate.evaluate(
        valid_rate=0.95, spearman_values=[0.9, None], tv_values=[0.1, None]
    )

    assert passed.passed is True
    assert failed.passed is False
    assert missing_state.passed is False


def test_stage_b_selection_uses_lower_median_and_canonical_tiebreaker():
    selected = select_stage_b_roots(
        [
            {"state": {"target": 9, "values": [1, 2, 3, 4]}, "tv_distance": 0.3},
            {"state": {"target": 8, "values": [1, 2, 3, 4]}, "tv_distance": 0.1},
            {"state": {"target": 7, "values": [1, 2, 3, 4]}, "tv_distance": 0.2},
            {"state": {"target": 6, "values": [1, 2, 3, 4]}, "tv_distance": 0.4},
        ]
    )

    assert [row["state"]["target"] for row in selected] == [8, 7, 6]


def test_bootstrap_response_counts_are_reproducible_and_preserve_invalids():
    responses = [
        {"ordering_id": "a", "action": "1 + 2 = 3"},
        {"ordering_id": "a", "action": None},
        {"ordering_id": "b", "action": "1 * 2 = 2"},
        {"ordering_id": "b", "action": None},
    ]

    first = bootstrap_response_counts(responses, replicates=5, seed=17)
    second = bootstrap_response_counts(responses, replicates=5, seed=17)

    assert first == second
    assert all(sum(result.values()) == 4 for result in first)
