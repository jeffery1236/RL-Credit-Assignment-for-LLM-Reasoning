"""Pure selection and fidelity metrics for the Phase 3 credit benchmark.

Key Terms:
- Credit state: a nonterminal state with exact action advantages and raw
  parent/child rollout outcomes for every action being evaluated.
- Raw outcome pool: the terminal rewards of independent policy rollouts from
  the parent state (estimating Vπ) or from an action's child state
  (estimating Qπ). Estimators are derived from these, never stored separately.
- Fidelity: Spearman correlation between exact and estimated within-state ranks.
- Pivotal state: a selected state whose exact best action is unique, so an
  estimator can be scored on whether it recovers that single action.

This layer accepts opaque problem, state, and action identifiers so orchestration
can supply environment-specific values without coupling selection or statistics
to the Countdown implementation. Selection only inspects exact advantages and
occupancy.

Estimators are derived rather than stored because storing them separately let
the bootstrap resample each derived estimator independently, which destroys the
pairing the gate depends on. Every replicate therefore resamples the underlying
parent and child outcomes exactly once per state-action pair and derives all
estimators from that single draw.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

from analysis.policy_bridge_metrics import spearman_rank_correlation

UNIFORM_ESTIMATOR = "uniform"
NESTED_ROLLOUT_COUNTS = (1, 2, 4, 8, 16)
MINIMUM_ROLLOUT_POOL_SIZE = max(NESTED_ROLLOUT_COUNTS)
PHASE3_ESTIMATOR_ORDER = (
    UNIFORM_ESTIMATOR,
    *(f"mc_reset_k{count}" for count in NESTED_ROLLOUT_COUNTS),
)
# One rollout for Uniform's single child draw; K parent plus K child rollouts
# for each Sampled Counterfactual Advantage budget.
ROLLOUT_COST_BY_ESTIMATOR = {
    UNIFORM_ESTIMATOR: 1,
    **{f"mc_reset_k{count}": 2 * count for count in NESTED_ROLLOUT_COUNTS},
}


@dataclass(frozen=True, slots=True)
class StateActionCredit:
    """Exact advantage and raw rollout outcomes for one action."""

    action_key: Hashable
    exact_advantage: float
    parent_outcomes: tuple[float, ...]
    child_outcomes: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SelectedCreditState:
    """One candidate state carrying all action-credit observations."""

    problem_key: Hashable
    state_key: Hashable
    depth: int
    occupancy: float
    action_credits: tuple[StateActionCredit, ...]

    @property
    def advantage_spread(self) -> float:
        """Return the range of exact advantages across the state's actions."""
        advantages = [credit.exact_advantage for credit in self.action_credits]
        if not advantages:
            raise ValueError("Credit state must contain at least one action")
        return max(advantages) - min(advantages)


@dataclass(frozen=True, slots=True)
class ProblemClusteredFidelity:
    """Bootstrap fidelity draws aligned across every estimator."""

    by_estimator: Mapping[str, tuple[float, ...]]
    problem_draw_count: int
    dropped_replicate_count: int


@dataclass(frozen=True, slots=True)
class PivotalActionDetection:
    """How often an estimator recovers a state's unique exact-best action."""

    detected_state_count: int
    pivotal_state_count: int

    @property
    def detection_rate(self) -> float | None:
        """Return the detection rate, or None when no state is pivotal."""
        if self.pivotal_state_count == 0:
            return None
        return self.detected_state_count / self.pivotal_state_count


def stable_estimator_order(estimator_keys: Sequence[str]) -> tuple[str, ...]:
    """Order Phase 3 estimators independently of checkpoint JSON key order."""
    estimator_key_set = set(estimator_keys)
    known_estimators = tuple(
        estimator
        for estimator in PHASE3_ESTIMATOR_ORDER
        if estimator in estimator_key_set
    )
    unexpected_estimators = tuple(
        sorted(estimator_key_set.difference(PHASE3_ESTIMATOR_ORDER))
    )
    return known_estimators + unexpected_estimators


def validate_rollout_pool_shape(
    credit_states: Sequence[SelectedCreditState],
) -> int:
    """Return the single rollout pool size every state-action pair must share."""
    observed_shapes = sorted(
        {
            (len(credit.parent_outcomes), len(credit.child_outcomes))
            for state in credit_states
            for credit in state.action_credits
        }
    )
    if not observed_shapes:
        raise ValueError("At least one state-action rollout pool is required")
    if len(observed_shapes) > 1:
        raise ValueError(
            "Credit states must share one rollout pool shape, but found "
            f"heterogeneous (parent, child) sizes {observed_shapes}"
        )
    parent_size, child_size = observed_shapes[0]
    if parent_size != child_size:
        raise ValueError(
            "Parent and child rollout pools must be paired, but found "
            f"{parent_size} parent and {child_size} child outcomes"
        )
    if parent_size < MINIMUM_ROLLOUT_POOL_SIZE:
        raise ValueError(
            f"Rollout pools of {parent_size} outcomes cannot support "
            f"K={MINIMUM_ROLLOUT_POOL_SIZE}"
        )
    return parent_size


def estimator_credit_samples(
    parent_outcomes: Sequence[float],
    child_outcomes: Sequence[float],
    estimator: str,
) -> tuple[float, ...]:
    """Derive one estimator's credit samples from a single raw rollout draw.

    Uniform uses the first child outcome and no baseline; Sampled
    Counterfactual Advantage at K uses the first K paired child-minus-parent
    differences, so every K is a nested prefix of the same draw.
    """
    if estimator == UNIFORM_ESTIMATOR:
        return (float(child_outcomes[0]),)
    rollout_count = int(estimator.removeprefix("mc_reset_k"))
    return tuple(
        float(child) - float(parent)
        for child, parent in zip(
            child_outcomes[:rollout_count],
            parent_outcomes[:rollout_count],
            strict=True,
        )
    )


def derive_estimator_pools(
    parent_outcomes: Sequence[float],
    child_outcomes: Sequence[float],
) -> dict[str, tuple[float, ...]]:
    """Derive every predeclared estimator's samples from one raw draw."""
    if (
        len(parent_outcomes) != len(child_outcomes)
        or len(child_outcomes) < MINIMUM_ROLLOUT_POOL_SIZE
    ):
        raise ValueError(
            "Estimator pools require paired parent and child outcomes through "
            f"K={MINIMUM_ROLLOUT_POOL_SIZE}, but found {len(parent_outcomes)} "
            f"parent and {len(child_outcomes)} child outcomes"
        )
    if any(
        not math.isfinite(value)
        for value in (*parent_outcomes, *child_outcomes)
    ):
        raise ValueError("Rollout outcomes must be finite")
    return {
        estimator: estimator_credit_samples(
            parent_outcomes, child_outcomes, estimator
        )
        for estimator in PHASE3_ESTIMATOR_ORDER
    }


def _mean_estimated_credit(
    parent_outcomes: Sequence[float],
    child_outcomes: Sequence[float],
    estimator: str,
) -> float:
    """Average one estimator's derived samples for a single action."""
    samples = estimator_credit_samples(parent_outcomes, child_outcomes, estimator)
    return math.fsum(samples) / len(samples)


def select_widest_credit_states(
    candidates: Sequence[SelectedCreditState],
    *,
    depths: Sequence[int] = (0, 1, 2),
) -> tuple[SelectedCreditState, ...]:
    """Select one eligible widest-advantage state at each requested depth."""
    selected = []
    for depth in depths:
        eligible = [
            candidate
            for candidate in candidates
            if candidate.depth == depth and len(candidate.action_credits) >= 3
        ]
        if eligible:
            selected.append(
                min(
                    eligible,
                    key=lambda candidate: (
                        -candidate.advantage_spread,
                        -candidate.occupancy,
                        candidate.state_key,
                    ),
                )
            )
    return tuple(selected)


def nondegenerate_credit_states(
    credit_states: Sequence[SelectedCreditState],
) -> tuple[SelectedCreditState, ...]:
    """Keep only states whose exact advantages actually rank their actions.

    This removes states that no estimator can be scored on. It does not remove
    states that every estimator happens to rank perfectly; use
    `credit_states_at_depths` to exclude a whole stratum.
    """
    return tuple(
        state
        for state in credit_states
        if len({credit.exact_advantage for credit in state.action_credits}) > 1
    )


def credit_states_at_depths(
    credit_states: Sequence[SelectedCreditState],
    depths: Sequence[int],
) -> tuple[SelectedCreditState, ...]:
    """Keep only states drawn from the requested decision depths."""
    requested = set(depths)
    return tuple(state for state in credit_states if state.depth in requested)


def _average_percentile_ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Convert values to average ranks normalized to the state's action count."""
    if not values:
        raise ValueError("At least one action value is required")
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        percentile_rank = (position + 1 + end) / (2.0 * len(indexed))
        for original_index, _ in indexed[position:end]:
            ranks[original_index] = percentile_rank
        position = end
    return tuple(ranks)


def _state_rank_pairs(
    credit_states: Sequence[SelectedCreditState],
    estimator: str,
) -> tuple[list[float], list[float]]:
    exact_ranks: list[float] = []
    estimated_ranks: list[float] = []
    for state in credit_states:
        if len(state.action_credits) < 2:
            raise ValueError("Every credit state needs at least two actions")
        exact_ranks.extend(
            _average_percentile_ranks(
                [credit.exact_advantage for credit in state.action_credits]
            )
        )
        estimated_ranks.extend(
            _average_percentile_ranks(
                [
                    _mean_estimated_credit(
                        credit.parent_outcomes, credit.child_outcomes, estimator
                    )
                    for credit in state.action_credits
                ]
            )
        )
    return exact_ranks, estimated_ranks


def within_state_rank_fidelity(
    credit_states: Sequence[SelectedCreditState],
    estimator: str,
) -> float | None:
    """Rank-correlate exact and estimated action ranks within each state."""
    exact_ranks, estimated_ranks = _state_rank_pairs(credit_states, estimator)
    return spearman_rank_correlation(exact_ranks, estimated_ranks)


def stratum_balanced_rank_fidelity(
    credit_states: Sequence[SelectedCreditState],
    estimator: str,
    *,
    depths: Sequence[int] = (0, 1, 2),
) -> float | None:
    """Average per-depth fidelity so no depth's action count dominates.

    Pooled fidelity weights every state-action pair equally, so a depth with
    many wide states — or a depth every estimator ranks perfectly — moves the
    aggregate more than the depth count suggests. This gives each populated
    depth equal weight instead.
    """
    per_depth = [
        within_state_rank_fidelity(states_at_depth, estimator)
        for states_at_depth in (
            tuple(state for state in credit_states if state.depth == depth)
            for depth in depths
        )
        if states_at_depth
    ]
    if not per_depth or any(fidelity is None for fidelity in per_depth):
        return None
    return math.fsum(float(fidelity) for fidelity in per_depth) / len(per_depth)


def pivotal_action_detection(
    credit_states: Sequence[SelectedCreditState],
    estimator: str,
) -> PivotalActionDetection:
    """Count states whose unique exact-best action the estimator ranks first.

    A state is pivotal when exactly one action attains the maximum exact
    advantage; only those states can distinguish an estimator that finds the
    right action from one that merely ranks ties consistently. Detection
    requires the estimator's mean credit to have that same unique argmax.
    """
    detected = 0
    pivotal = 0
    for state in credit_states:
        exact_advantages = [credit.exact_advantage for credit in state.action_credits]
        best_exact = max(exact_advantages)
        if exact_advantages.count(best_exact) != 1:
            continue
        pivotal += 1
        estimated_credits = [
            _mean_estimated_credit(
                credit.parent_outcomes, credit.child_outcomes, estimator
            )
            for credit in state.action_credits
        ]
        best_estimated = max(estimated_credits)
        if (
            estimated_credits.count(best_estimated) == 1
            and estimated_credits.index(best_estimated)
            == exact_advantages.index(best_exact)
        ):
            detected += 1
    return PivotalActionDetection(
        detected_state_count=detected,
        pivotal_state_count=pivotal,
    )


def _resampled_state(
    state: SelectedCreditState,
    rng: random.Random,
    *,
    shared_parent_baseline: bool,
) -> SelectedCreditState:
    """Resample one state's raw outcomes once, before any estimator is derived."""
    shared_parent_outcomes: tuple[float, ...] | None = None
    if shared_parent_baseline:
        baseline = state.action_credits[0].parent_outcomes
        if any(
            credit.parent_outcomes != baseline for credit in state.action_credits
        ):
            raise ValueError(
                "A shared-baseline bootstrap needs one parent pool per state, but "
                f"state {state.state_key!r} has action-specific parent pools"
            )
        shared_parent_outcomes = tuple(rng.choices(baseline, k=len(baseline)))
    return SelectedCreditState(
        problem_key=state.problem_key,
        state_key=state.state_key,
        depth=state.depth,
        occupancy=state.occupancy,
        action_credits=tuple(
            StateActionCredit(
                action_key=credit.action_key,
                exact_advantage=credit.exact_advantage,
                parent_outcomes=(
                    shared_parent_outcomes
                    if shared_parent_outcomes is not None
                    else tuple(
                        rng.choices(
                            credit.parent_outcomes, k=len(credit.parent_outcomes)
                        )
                    )
                ),
                child_outcomes=tuple(
                    rng.choices(credit.child_outcomes, k=len(credit.child_outcomes))
                ),
            )
            for credit in state.action_credits
        ),
    )


def bootstrap_problem_clustered_fidelity(
    credit_states: Sequence[SelectedCreditState],
    *,
    replicates: int,
    seed: int,
    stratum_balanced: bool = False,
    shared_parent_baseline: bool = False,
) -> ProblemClusteredFidelity:
    """Bootstrap whole problems and their raw rollout pools for each estimator.

    Steps:
    1. Group selected states by problem and draw those groups with replacement.
    2. Resample each state-action pair's parent and child outcomes exactly once
       per replicate, before any estimator is derived, so every estimator sees
       the same underlying draw. With `shared_parent_baseline`, the state's one
       parent pool is resampled once and reused by all of its actions.
    3. Score every estimator on that same draw, dropping undefined replicates
       as a whole to preserve paired estimator comparisons.
    """
    if replicates < 1:
        raise ValueError("replicates must be at least one")
    validate_rollout_pool_shape(credit_states)
    states_by_problem: dict[Hashable, list[SelectedCreditState]] = {}
    for state in credit_states:
        states_by_problem.setdefault(state.problem_key, []).append(state)
    if not states_by_problem:
        raise ValueError("At least one credit state is required")

    rng = random.Random(seed)
    collected: dict[str, list[float]] = {
        estimator: [] for estimator in PHASE3_ESTIMATOR_ORDER
    }
    problem_groups = tuple(states_by_problem.values())
    dropped = 0
    for _ in range(replicates):
        # Step 1: draw problem clusters, keeping each problem's states together.
        sampled_states = tuple(
            state
            for group in rng.choices(problem_groups, k=len(problem_groups))
            for state in group
        )
        # Step 2: resample the raw outcomes once per state-action pair, keeping
        # a shared baseline shared so the variant it models survives the
        # bootstrap.
        replica = tuple(
            _resampled_state(
                state, rng, shared_parent_baseline=shared_parent_baseline
            )
            for state in sampled_states
        )
        # Step 3: derive every estimator from that one draw.
        fidelity_by_estimator = {
            estimator: (
                stratum_balanced_rank_fidelity(replica, estimator)
                if stratum_balanced
                else within_state_rank_fidelity(replica, estimator)
            )
            for estimator in PHASE3_ESTIMATOR_ORDER
        }
        if any(fidelity is None for fidelity in fidelity_by_estimator.values()):
            dropped += 1
            continue
        for estimator, fidelity in fidelity_by_estimator.items():
            collected[estimator].append(float(fidelity))

    return ProblemClusteredFidelity(
        by_estimator={
            estimator: tuple(values) for estimator, values in collected.items()
        },
        problem_draw_count=len(problem_groups),
        dropped_replicate_count=dropped,
    )
