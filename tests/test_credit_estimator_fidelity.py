"""Tests for the pure Phase 3 credit-estimator fidelity layer."""

import random

import pytest

from analysis.credit_estimator_fidelity import (
    MINIMUM_ROLLOUT_POOL_SIZE,
    PHASE3_ESTIMATOR_ORDER,
    ROLLOUT_COST_BY_ESTIMATOR,
    SelectedCreditState,
    StateActionCredit,
    bootstrap_problem_clustered_fidelity,
    credit_states_at_depths,
    derive_estimator_pools,
    nondegenerate_credit_states,
    pivotal_action_detection,
    select_widest_credit_states,
    stable_estimator_order,
    stratum_balanced_rank_fidelity,
    validate_rollout_pool_shape,
    within_state_rank_fidelity,
)

POOL_SIZE = MINIMUM_ROLLOUT_POOL_SIZE


def _pool(*values):
    """Repeat the given outcomes until they fill one full rollout pool."""
    repeated = (values * POOL_SIZE)[:POOL_SIZE]
    return tuple(float(value) for value in repeated)


def _state(
    problem_key,
    state_key,
    depth,
    occupancy,
    advantages,
    *,
    outcome_pools=None,
):
    outcome_pools = outcome_pools or {}
    return SelectedCreditState(
        problem_key=problem_key,
        state_key=state_key,
        depth=depth,
        occupancy=occupancy,
        action_credits=tuple(
            StateActionCredit(
                action_key=action_key,
                exact_advantage=advantage,
                parent_outcomes=outcome_pools.get(action_key, ((), ()))[0],
                child_outcomes=outcome_pools.get(action_key, ((), ()))[1],
            )
            for action_key, advantage in advantages.items()
        ),
    )


def test_selects_widest_spread_state_then_occupancy_then_canonical_key():
    selected = select_widest_credit_states(
        [
            _state("problem", "narrow", 0, 1.0, {"a": -0.5, "b": 0.0, "c": 0.5}),
            _state("problem", "widest", 0, 0.1, {"a": -2.0, "b": 0.0, "c": 2.0}),
            _state(
                "problem",
                "lower-occupancy",
                1,
                0.1,
                {"a": -1.0, "b": 0.0, "c": 1.0},
            ),
            _state(
                "problem",
                "higher-occupancy",
                1,
                0.9,
                {"a": -1.0, "b": 0.0, "c": 1.0},
            ),
            _state(
                "problem",
                "canonical-second",
                2,
                0.5,
                {"a": -1.0, "b": 0.0, "c": 1.0},
            ),
            _state(
                "problem",
                "canonical-first",
                2,
                0.5,
                {"a": -1.0, "b": 0.0, "c": 1.0},
            ),
        ],
        depths=(0, 1, 2),
    )

    assert [row.state_key for row in selected] == [
        "widest",
        "higher-occupancy",
        "canonical-first",
    ]


def test_unavailable_depth_is_recorded_without_fallback():
    selected = select_widest_credit_states(
        [
            _state("problem", "only-depth-zero", 0, 1.0, {"a": -1.0, "b": 0.0, "c": 1.0}),
            _state("problem", "too-few-actions", 1, 1.0, {"a": 1.0, "b": 0.0}),
        ],
        depths=(0, 1, 2),
    )

    assert [row.depth for row in selected] == [0]


def test_derived_estimators_are_nested_prefixes_of_one_raw_draw():
    parent_outcomes = tuple(float(index) for index in range(POOL_SIZE))
    child_outcomes = tuple(float(index) * 2 for index in range(POOL_SIZE))

    pools = derive_estimator_pools(parent_outcomes, child_outcomes)

    assert tuple(pools) == PHASE3_ESTIMATOR_ORDER
    assert pools["uniform"] == (child_outcomes[0],)
    for rollout_count in (1, 2, 4, 8, 16):
        estimator = f"mc_reset_k{rollout_count}"
        assert len(pools[estimator]) == rollout_count
        assert pools[estimator] == pools["mc_reset_k16"][:rollout_count]
        assert pools[estimator] == tuple(
            child - parent
            for child, parent in zip(
                child_outcomes[:rollout_count], parent_outcomes[:rollout_count]
            )
        )


def test_derived_estimator_key_order_is_canonical_regardless_of_input_order():
    parent_outcomes = _pool(0.0, 1.0)
    child_outcomes = _pool(1.0, 0.0)

    pools = derive_estimator_pools(parent_outcomes, child_outcomes)
    shuffled = list(pools)
    random.Random(11).shuffle(shuffled)

    assert stable_estimator_order(shuffled) == tuple(pools)


def test_rollout_costs_count_both_arms():
    assert ROLLOUT_COST_BY_ESTIMATOR == {
        "uniform": 1,
        "mc_reset_k1": 2,
        "mc_reset_k2": 4,
        "mc_reset_k4": 8,
        "mc_reset_k8": 16,
        "mc_reset_k16": 32,
    }


def _paired_states(*, parent_first, child_first, parent_second, child_second):
    outcome_pools = {
        "low": (parent_first, child_first),
        "middle": (parent_second, child_second),
        "high": (parent_first, child_second),
    }
    return [
        _state(
            "first-problem",
            "first-state",
            0,
            1.0,
            {"low": -1.0, "middle": 0.0, "high": 1.0},
            outcome_pools=outcome_pools,
        ),
        _state(
            "second-problem",
            "second-state",
            1,
            1.0,
            {"low": -1.0, "middle": 0.0, "high": 1.0},
            outcome_pools=outcome_pools,
        ),
    ]


def test_heterogeneous_rollout_pools_raise_a_descriptive_error():
    states = [
        _state(
            "problem",
            "state",
            0,
            1.0,
            {"low": -1.0, "high": 1.0},
            outcome_pools={
                "low": (_pool(0.0), _pool(1.0)),
                "high": (_pool(0.0)[:4], _pool(1.0)[:4]),
            },
        )
    ]

    with pytest.raises(ValueError, match="heterogeneous"):
        validate_rollout_pool_shape(states)


def test_unpaired_rollout_pools_raise_a_descriptive_error():
    states = [
        _state(
            "problem",
            "state",
            0,
            1.0,
            {"low": -1.0, "high": 1.0},
            outcome_pools={
                "low": (_pool(0.0)[:8], _pool(1.0)),
                "high": (_pool(0.0)[:8], _pool(1.0)),
            },
        )
    ]

    with pytest.raises(ValueError, match="must be paired"):
        validate_rollout_pool_shape(states)


def test_within_state_rank_fidelity_removes_cross_problem_baselines():
    states = [
        _state(
            "first-problem",
            "first-state",
            0,
            1.0,
            {"low": 10.0, "middle": 20.0, "high": 30.0},
            outcome_pools={
                "low": (_pool(0.0), _pool(1.0)),
                "middle": (_pool(0.0), _pool(2.0)),
                "high": (_pool(0.0), _pool(3.0)),
            },
        ),
        _state(
            "second-problem",
            "second-state",
            1,
            1.0,
            {"low": -30.0, "middle": -20.0, "high": -10.0},
            outcome_pools={
                "low": (_pool(0.0), _pool(1.0)),
                "middle": (_pool(0.0), _pool(2.0)),
                "high": (_pool(0.0), _pool(3.0)),
            },
        ),
    ]

    fidelity = within_state_rank_fidelity(states, "mc_reset_k16")

    assert fidelity == pytest.approx(1.0)


def test_stratum_balanced_fidelity_gives_each_depth_equal_weight():
    aligned_pools = {
        "low": (_pool(0.0), _pool(1.0)),
        "middle": (_pool(0.0), _pool(2.0)),
        "high": (_pool(0.0), _pool(3.0)),
    }
    reversed_pools = {
        "low": (_pool(0.0), _pool(3.0)),
        "middle": (_pool(0.0), _pool(2.0)),
        "high": (_pool(0.0), _pool(1.0)),
    }
    advantages = {"low": -1.0, "middle": 0.0, "high": 1.0}
    states = [
        _state("p", "aligned-a", 0, 1.0, advantages, outcome_pools=aligned_pools),
        _state("p", "aligned-b", 0, 1.0, advantages, outcome_pools=aligned_pools),
        _state("p", "aligned-c", 0, 1.0, advantages, outcome_pools=aligned_pools),
        _state("p", "reversed", 1, 1.0, advantages, outcome_pools=reversed_pools),
    ]

    pooled = within_state_rank_fidelity(states, "mc_reset_k16")
    balanced = stratum_balanced_rank_fidelity(states, "mc_reset_k16")

    assert pooled == pytest.approx(0.5)
    assert balanced == pytest.approx(0.0)


def test_nondegenerate_states_drop_fully_tied_exact_advantages():
    tied = _state(
        "p",
        "tied",
        2,
        1.0,
        {"a": 0.0, "b": 0.0, "c": 0.0},
        outcome_pools={key: (_pool(0.0), _pool(0.0)) for key in ("a", "b", "c")},
    )
    informative = _state(
        "p",
        "informative",
        0,
        1.0,
        {"a": -1.0, "b": 0.0, "c": 1.0},
        outcome_pools={key: (_pool(0.0), _pool(1.0)) for key in ("a", "b", "c")},
    )

    assert nondegenerate_credit_states([tied, informative]) == (informative,)
    # The filter removes unscorable states, not a whole depth: a depth-2 state
    # with distinct exact advantages survives.
    untied_depth_two = _state(
        "p",
        "untied-depth-two",
        2,
        1.0,
        {"a": -1.0, "b": 0.0, "c": 1.0},
        outcome_pools={key: (_pool(0.0), _pool(1.0)) for key in ("a", "b", "c")},
    )
    assert nondegenerate_credit_states([tied, untied_depth_two]) == (untied_depth_two,)


def test_credit_states_at_depths_excludes_a_whole_stratum():
    pools = {key: (_pool(0.0), _pool(1.0)) for key in ("a", "b", "c")}
    advantages = {"a": -1.0, "b": 0.0, "c": 1.0}
    states = [
        _state("p", f"depth-{depth}", depth, 1.0, advantages, outcome_pools=pools)
        for depth in (0, 1, 2)
    ]

    assert [
        state.depth for state in credit_states_at_depths(states, (0, 1))
    ] == [0, 1]
    assert credit_states_at_depths(states, ()) == ()


def test_pivotal_action_detection_requires_a_unique_exact_and_estimated_best():
    detected = _state(
        "p",
        "detected",
        0,
        1.0,
        {"a": -1.0, "b": 0.0, "c": 1.0},
        outcome_pools={
            "a": (_pool(0.0), _pool(0.0)),
            "b": (_pool(0.0), _pool(0.5)),
            "c": (_pool(0.0), _pool(1.0)),
        },
    )
    missed = _state(
        "p",
        "missed",
        0,
        1.0,
        {"a": -1.0, "b": 0.0, "c": 1.0},
        outcome_pools={
            "a": (_pool(0.0), _pool(1.0)),
            "b": (_pool(0.0), _pool(0.5)),
            "c": (_pool(0.0), _pool(0.0)),
        },
    )
    tied_exact = _state(
        "p",
        "tied-exact",
        0,
        1.0,
        {"a": 1.0, "b": 1.0, "c": 0.0},
        outcome_pools={
            "a": (_pool(0.0), _pool(1.0)),
            "b": (_pool(0.0), _pool(0.5)),
            "c": (_pool(0.0), _pool(0.0)),
        },
    )

    detection = pivotal_action_detection(
        [detected, missed, tied_exact], "mc_reset_k16"
    )

    assert detection.pivotal_state_count == 2
    assert detection.detected_state_count == 1
    assert detection.detection_rate == pytest.approx(0.5)


def test_problem_clustered_bootstrap_is_reproducible_and_keeps_state_groups():
    states = _paired_states(
        parent_first=_pool(0.0, 1.0),
        child_first=_pool(0.0, 0.0),
        parent_second=_pool(0.0, 1.0),
        child_second=_pool(1.0, 1.0),
    )

    first = bootstrap_problem_clustered_fidelity(states, replicates=25, seed=7)
    second = bootstrap_problem_clustered_fidelity(states, replicates=25, seed=7)

    assert first == second
    assert first.problem_draw_count == 2
    assert len(first.by_estimator["mc_reset_k16"]) + first.dropped_replicate_count == 25


def _noisy_child_states(*, parent_pool):
    """Two single-state problems whose child pools resampling can reorder."""
    outcome_pools = {
        "low": (parent_pool, _pool(0.0, 0.0, 0.0, 1.0)),
        "middle": (parent_pool, _pool(0.0, 1.0)),
        "high": (parent_pool, _pool(1.0, 1.0, 1.0, 0.0)),
    }
    advantages = {"low": -1.0, "middle": 0.0, "high": 1.0}
    return [
        _state("first", "first-state", 0, 1.0, advantages, outcome_pools=outcome_pools),
        _state("second", "second-state", 1, 1.0, advantages, outcome_pools=outcome_pools),
    ]


def test_bootstrap_resampling_reaches_the_uniform_estimator():
    # Uniform's derived sample is a single number, so resampling it after
    # derivation is a no-op and Uniform would be frozen at the original first
    # child outcome. With one problem, cluster resampling cannot vary anything
    # either, so Uniform can only move if the raw child pool is resampled first.
    states = _noisy_child_states(parent_pool=_pool(0.0, 0.0, 1.0))[:1]

    bootstrap = bootstrap_problem_clustered_fidelity(states, replicates=200, seed=3)

    assert bootstrap.problem_draw_count == 1
    assert len(set(bootstrap.by_estimator["uniform"])) > 1


def test_bootstrap_replicate_derives_every_estimator_from_one_raw_resample():
    states = _noisy_child_states(parent_pool=_pool(0.0, 0.0, 1.0))
    seed = 17

    bootstrap = bootstrap_problem_clustered_fidelity(
        states, replicates=1, seed=seed
    )

    # Replay the same RNG stream independently: draw the problem clusters, then
    # one parent and one child resample per state-action pair, and derive the
    # estimators from those. Matching all six draws shows each estimator came
    # from nested prefixes of that single raw draw rather than its own resample.
    rng = random.Random(seed)
    problem_groups = ([states[0]], [states[1]])
    reconstructed = []
    for state in [
        state
        for group in rng.choices(problem_groups, k=len(problem_groups))
        for state in group
    ]:
        action_credits = []
        for credit in state.action_credits:
            parent_outcomes = tuple(
                rng.choices(credit.parent_outcomes, k=len(credit.parent_outcomes))
            )
            child_outcomes = tuple(
                rng.choices(credit.child_outcomes, k=len(credit.child_outcomes))
            )
            action_credits.append(
                StateActionCredit(
                    action_key=credit.action_key,
                    exact_advantage=credit.exact_advantage,
                    parent_outcomes=parent_outcomes,
                    child_outcomes=child_outcomes,
                )
            )
        reconstructed.append(
            SelectedCreditState(
                problem_key=state.problem_key,
                state_key=state.state_key,
                depth=state.depth,
                occupancy=state.occupancy,
                action_credits=tuple(action_credits),
            )
        )

    assert bootstrap.dropped_replicate_count == 0
    assert {
        estimator: draws[0] for estimator, draws in bootstrap.by_estimator.items()
    } == {
        estimator: within_state_rank_fidelity(reconstructed, estimator)
        for estimator in PHASE3_ESTIMATOR_ORDER
    }
    # The reconstructed draw is genuinely a resample, not the original pools.
    assert any(
        credit.child_outcomes != original.child_outcomes
        for state, original_state in zip(reconstructed, states, strict=False)
        for credit, original in zip(
            state.action_credits, original_state.action_credits, strict=True
        )
    )


def _shared_baseline_states():
    noisy_parent_pool = _pool(0.0, 1.0)
    outcome_pools = {
        "low": (noisy_parent_pool, _pool(0.0)),
        "middle": (noisy_parent_pool, _pool(0.4)),
        "high": (noisy_parent_pool, _pool(0.6)),
    }
    advantages = {"low": -1.0, "middle": 0.0, "high": 1.0}
    return [
        _state("first", "first-state", 0, 1.0, advantages, outcome_pools=outcome_pools),
        _state("second", "second-state", 1, 1.0, advantages, outcome_pools=outcome_pools),
    ]


def test_shared_baseline_bootstrap_resamples_one_parent_pool_per_state():
    states = _shared_baseline_states()

    shared = bootstrap_problem_clustered_fidelity(
        states, replicates=200, seed=5, shared_parent_baseline=True
    )
    per_action = bootstrap_problem_clustered_fidelity(
        states, replicates=200, seed=5
    )

    # A baseline that is constant within a state cannot reorder that state's
    # actions, so every replicate recovers the exact ranking.
    assert set(shared.by_estimator["mc_reset_k16"]) == {1.0}
    assert any(draw < 1.0 for draw in per_action.by_estimator["mc_reset_k16"])


def test_shared_baseline_bootstrap_rejects_action_specific_parent_pools():
    states = [
        _state(
            "problem",
            "state",
            0,
            1.0,
            {"low": -1.0, "middle": 0.0, "high": 1.0},
            outcome_pools={
                "low": (_pool(0.0), _pool(0.0)),
                "middle": (_pool(1.0), _pool(0.4)),
                "high": (_pool(0.0), _pool(0.6)),
            },
        )
    ]

    with pytest.raises(ValueError, match="action-specific parent pools"):
        bootstrap_problem_clustered_fidelity(
            states, replicates=5, seed=5, shared_parent_baseline=True
        )


def test_bootstrap_draws_do_not_depend_on_estimator_iteration_order(monkeypatch):
    states = _paired_states(
        parent_first=_pool(0.0, 1.0),
        child_first=_pool(0.0, 0.0),
        parent_second=_pool(0.0, 1.0),
        child_second=_pool(1.0, 1.0),
    )
    in_canonical_order = bootstrap_problem_clustered_fidelity(
        states, replicates=25, seed=7
    )

    # Resampling happens before any estimator is derived, so iterating the
    # estimators in another order must consume randomness identically.
    monkeypatch.setattr(
        "analysis.credit_estimator_fidelity.PHASE3_ESTIMATOR_ORDER",
        tuple(reversed(PHASE3_ESTIMATOR_ORDER)),
    )
    in_reversed_order = bootstrap_problem_clustered_fidelity(
        states, replicates=25, seed=7
    )

    assert dict(in_reversed_order.by_estimator) == dict(
        in_canonical_order.by_estimator
    )
