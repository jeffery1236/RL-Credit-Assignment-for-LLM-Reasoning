"""Tests for conclusion-robustness selection, estimators, and the gate."""

import pytest

from analysis.conclusion_robustness_metrics import (
    ESTIMATOR_POOL_SIZE,
    K_VALUES,
    METHOD_NAMES,
    UNIFORM_METHOD,
    BootstrapFidelity,
    ConclusionRobustnessGate,
    DirectionalSupport,
    PairSamples,
    StateCandidate,
    assigned_depth_for_root,
    bootstrap_method_fidelity,
    directional_support,
    log2k_slope,
    mc_reset_credit,
    method_fidelity,
    method_rank_agreement,
    reference_split_half_stability,
    select_actions_for_state,
    select_benchmark_pairs,
    select_state_for_root,
    state_occupancy,
    uniform_outcome_credit,
)


def _candidate(key, depth, occupancy, advantages):
    return StateCandidate(
        canonical_key=key,
        depth=depth,
        occupancy=occupancy,
        advantage_by_action=advantages,
    )


THREE_ACTIONS = {"a": 0.4, "b": 0.0, "c": -0.3}


def test_occupancy_sums_mass_from_every_incoming_path():
    # Two distinct actions from the root merge into the same child.
    occupancy = state_occupancy(
        topological_states=["root", "left", "merged"],
        out_edges={
            "root": [("x", "left"), ("y", "merged")],
            "left": [("z", "merged")],
        },
        action_probabilities={"root": {"x": 0.6, "y": 0.4}, "left": {"z": 1.0}},
        root="root",
    )

    assert occupancy["root"] == pytest.approx(1.0)
    assert occupancy["left"] == pytest.approx(0.6)
    assert occupancy["merged"] == pytest.approx(1.0)


def test_depths_cycle_so_each_depth_gets_an_equal_share():
    assert [assigned_depth_for_root(index) for index in range(6)] == [0, 1, 2, 0, 1, 2]


def test_selects_highest_occupancy_state_at_the_assigned_depth():
    chosen = select_state_for_root(
        [
            _candidate((10, (1, 2, 3)), 1, 0.2, THREE_ACTIONS),
            _candidate((10, (1, 5)), 1, 0.9, THREE_ACTIONS),
            _candidate((10, (2, 4)), 2, 1.0, THREE_ACTIONS),
        ],
        assigned_depth=1,
    )

    assert chosen.canonical_key == (10, (1, 5))


def test_skips_states_without_enough_actions_to_define_three_strata():
    chosen = select_state_for_root(
        [
            _candidate((10, (1, 2)), 1, 0.99, {"only": 0.1}),
            _candidate((10, (3, 4)), 1, 0.10, THREE_ACTIONS),
        ],
        assigned_depth=1,
    )

    assert chosen.canonical_key == (10, (3, 4))


def test_falls_back_to_the_nearest_depth_when_none_is_eligible():
    chosen = select_state_for_root(
        [_candidate((10, (1, 2, 3)), 0, 1.0, THREE_ACTIONS)],
        assigned_depth=2,
    )

    assert chosen.depth == 0


def test_ties_break_on_canonical_state_order():
    chosen = select_state_for_root(
        [
            _candidate((10, (2, 9)), 1, 0.5, THREE_ACTIONS),
            _candidate((10, (1, 9)), 1, 0.5, THREE_ACTIONS),
        ],
        assigned_depth=1,
    )

    assert chosen.canonical_key == (10, (1, 9))


def test_action_strata_track_constrained_advantage_ranks():
    actions = select_actions_for_state(
        {"best": 0.5, "mid": 0.1, "worst": -0.4, "also_mid": 0.0},
        omitted_stratum="middle",
    )

    assert [action.stratum for action in actions] == ["top", "bottom"]
    assert actions[0].action_key == "best"
    assert actions[1].action_key == "worst"
    assert actions[1].advantage_rank == 3


def test_advantage_ties_break_deterministically_on_action_key():
    actions = select_actions_for_state(
        {"b": 0.5, "a": 0.5, "c": -0.1}, omitted_stratum="bottom"
    )

    assert actions[0].action_key == "a"


def test_each_stratum_appears_equally_across_twelve_roots():
    candidates = [[_candidate((1, (index, 9)), 0, 1.0, THREE_ACTIONS)] for index in range(12)]

    selections = select_benchmark_pairs(candidates)

    strata = [action.stratum for selection in selections for action in selection.actions]
    assert len(strata) == 24
    assert {stratum: strata.count(stratum) for stratum in set(strata)} == {
        "top": 8,
        "middle": 8,
        "bottom": 8,
    }


def test_uniform_uses_one_outcome_and_no_baseline():
    assert uniform_outcome_credit([1, 0, 0, 0]) == 1.0
    assert uniform_outcome_credit([0, 1, 1, 1]) == 0.0


def test_mc_reset_subtracts_the_state_baseline_over_k_samples():
    credit = mc_reset_credit([1, 1, 0, 0], [0, 0, 0, 0], k=2)

    assert credit == pytest.approx(1.0)
    assert mc_reset_credit([1, 1, 0, 0], [1, 1, 1, 1], k=4) == pytest.approx(-0.5)


def test_mc_reset_uses_nested_prefixes_of_one_shared_pool():
    child = [1, 0, 0, 0, 0, 0, 0, 0]
    state = [0] * 8

    assert mc_reset_credit(child, state, k=1) == pytest.approx(1.0)
    assert mc_reset_credit(child, state, k=2) == pytest.approx(0.5)
    assert mc_reset_credit(child, state, k=4) == pytest.approx(0.25)


def test_mc_reset_rejects_a_pool_smaller_than_k():
    with pytest.raises(ValueError, match="at least k"):
        mc_reset_credit([1, 0], [0, 0], k=4)


def _pair(child_success, state_success, reference):
    return PairSamples(
        child_outcomes=tuple([1] * child_success + [0] * (ESTIMATOR_POOL_SIZE - child_success)),
        state_outcomes=tuple([1] * state_success + [0] * (ESTIMATOR_POOL_SIZE - state_success)),
        reference_advantage=reference,
    )


def test_fidelity_is_perfect_when_estimates_track_the_reference():
    pairs = [_pair(16, 0, 1.0), _pair(8, 8, 0.0), _pair(0, 16, -1.0)]

    assert method_fidelity(pairs, "mc_reset_k16") == pytest.approx(1.0)


def test_slope_detects_whether_more_samples_help():
    assert log2k_slope({1: 0.1, 2: 0.2, 4: 0.3, 8: 0.4, 16: 0.5}) > 0
    assert log2k_slope({1: 0.5, 2: 0.4, 4: 0.3, 8: 0.2, 16: 0.1}) < 0


def test_bootstrap_is_reproducible_and_covers_every_method():
    pairs = [_pair(16, 0, 1.0), _pair(9, 7, 0.1), _pair(0, 16, -1.0)]

    first = bootstrap_method_fidelity(pairs, replicates=25, seed=7)
    second = bootstrap_method_fidelity(pairs, replicates=25, seed=7)
    third = bootstrap_method_fidelity(pairs, replicates=25, seed=8)

    assert set(first.by_method) == set(METHOD_NAMES)
    assert first == second
    assert first != third


def test_bootstrap_keeps_methods_paired_within_a_replicate():
    # Uniform reads only one outcome, so it goes constant far more often than
    # K=16. Dropping such replicates whole is what keeps index i comparable.
    pairs = [_pair(16, 0, 1.0), _pair(15, 1, 0.9), _pair(16, 0, 0.8)]

    bootstrap = bootstrap_method_fidelity(pairs, replicates=200, seed=11)

    lengths = {len(values) for values in bootstrap.by_method.values()}
    assert len(lengths) == 1, "every method must report the same replicates"
    assert bootstrap.dropped_replicate_count > 0
    assert (
        lengths.pop() + bootstrap.dropped_replicate_count == 200
    ), "retained plus dropped must account for every replicate"


def test_directional_support_reports_fractions_between_zero_and_one():
    # Fidelity rises with K, and every K beats Uniform's flat 0.05.
    by_method = {UNIFORM_METHOD: (0.05,) * 3}
    for index, value in enumerate(K_VALUES):
        by_method[f"mc_reset_k{value}"] = (0.1 * (index + 1),) * 3

    support = directional_support(
        BootstrapFidelity(by_method=by_method, dropped_replicate_count=0)
    )

    assert support.k16_beats_uniform == pytest.approx(1.0)
    assert support.positive_log2k_slope == pytest.approx(1.0)


def test_directional_support_refuses_an_empty_bootstrap():
    empty = BootstrapFidelity(
        by_method={name: () for name in METHOD_NAMES}, dropped_replicate_count=5
    )

    with pytest.raises(ValueError, match="no comparable replicates"):
        directional_support(empty)


def test_reference_stability_is_high_for_separable_actions():
    pools = [
        ((1,) * 32, (0,) * 32),
        ((1,) * 16 + (0,) * 16, (0,) * 32),
        ((0,) * 32, (1,) * 32),
    ]

    stability = reference_split_half_stability(pools, replicates=50, seed=3)

    assert stability is not None and stability > 0.9


def test_reference_stability_is_low_for_pure_noise():
    # Every arm has the same coin-flip behavior, so halves cannot agree on order.
    pools = [((1, 0) * 16, (1, 0) * 16) for _ in range(6)]

    stability = reference_split_half_stability(pools, replicates=200, seed=5)

    assert stability is None or stability < 0.5


STRONG = DirectionalSupport(k16_beats_uniform=0.95, positive_log2k_slope=0.9)
WEAK = DirectionalSupport(k16_beats_uniform=0.4, positive_log2k_slope=0.3)


def test_unstable_reference_is_inconclusive_not_a_disagreement():
    decision = ConclusionRobustnessGate().evaluate(
        reference_stability=0.2,
        constrained_support=STRONG,
        deliberative_support=STRONG,
        method_rank_agreement=1.0,
    )

    assert decision.decision == "INCONCLUSIVE"
    assert "not stable enough" in decision.reasons[0]


def test_go_requires_both_setups_and_rank_agreement():
    gate = ConclusionRobustnessGate()

    assert (
        gate.evaluate(
            reference_stability=0.5,
            constrained_support=STRONG,
            deliberative_support=STRONG,
            method_rank_agreement=0.7,
        ).decision
        == "GO"
    )
    assert (
        gate.evaluate(
            reference_stability=0.9,
            constrained_support=STRONG,
            deliberative_support=WEAK,
            method_rank_agreement=1.0,
        ).decision
        == "NO-GO"
    )
    assert (
        gate.evaluate(
            reference_stability=0.9,
            constrained_support=STRONG,
            deliberative_support=STRONG,
            method_rank_agreement=0.69,
        ).decision
        == "NO-GO"
    )


def test_gate_boundaries_are_inclusive():
    gate = ConclusionRobustnessGate()
    boundary = DirectionalSupport(k16_beats_uniform=0.8, positive_log2k_slope=0.8)

    assert (
        gate.evaluate(
            reference_stability=0.5,
            constrained_support=boundary,
            deliberative_support=boundary,
            method_rank_agreement=0.7,
        ).decision
        == "GO"
    )


def test_method_rank_agreement_requires_every_method():
    with pytest.raises(ValueError, match="Missing method fidelity"):
        method_rank_agreement({"uniform": 0.1}, {name: 0.1 for name in METHOD_NAMES})
