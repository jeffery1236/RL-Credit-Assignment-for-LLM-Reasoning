"""Pure analysis for the policy-bridge conclusion-robustness benchmark.

Key Terms:
- Benchmark pair: one selected (state, action) whose credit both setups estimate.
- Reference advantage: the quantity estimators are scored against. It is exact
  A-pi for the constrained setup and a held-out rollout estimate for the
  deliberative setup.
- Estimator pool: the 16 outcomes per arm used to emulate Uniform and MC-reset.
  Reference and estimator pools never share responses.
- Fidelity: rank correlation between a method's credit and the reference across
  all benchmark pairs.
- Arm: one state whose outcomes estimate V(s), or one child state whose outcomes
  estimate Q(s,a).

Selection is separated from estimation so the predeclared choice of states and
actions can be tested without a model, and so it provably never inspects
deliberative outcomes. Everything here is dependency-free and keyed by opaque
hashable identifiers rather than environment types.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence

from analysis.policy_bridge_metrics import spearman_rank_correlation

STRATA = ("top", "middle", "bottom")
K_VALUES = (1, 2, 4, 8, 16)
UNIFORM_METHOD = "uniform"
METHOD_NAMES = (UNIFORM_METHOD, *(f"mc_reset_k{value}" for value in K_VALUES))
MINIMUM_ACTION_COUNT = 3
ESTIMATOR_POOL_SIZE = 16


@dataclass(frozen=True, slots=True)
class StateCandidate:
    """One nonterminal DAG state eligible to host benchmark pairs."""

    canonical_key: tuple[int, tuple[int, ...]]
    depth: int
    occupancy: float
    advantage_by_action: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class SelectedAction:
    """One chosen action, with the stratum and rank that justified it."""

    action_key: str
    stratum: str
    constrained_advantage: float
    advantage_rank: int


@dataclass(frozen=True, slots=True)
class RootSelection:
    """The state chosen for one root plus its two benchmark actions."""

    root_index: int
    assigned_depth: int
    selected_depth: int
    canonical_key: tuple[int, tuple[int, ...]]
    occupancy: float
    omitted_stratum: str
    actions: tuple[SelectedAction, ...]


@dataclass(frozen=True, slots=True)
class PairSamples:
    """Estimator-pool outcomes and reference advantage for one benchmark pair."""

    child_outcomes: tuple[int, ...]
    state_outcomes: tuple[int, ...]
    reference_advantage: float


@dataclass(frozen=True, slots=True)
class DirectionalSupport:
    """Bootstrap support for the two directional claims in one setup."""

    k16_beats_uniform: float
    positive_log2k_slope: float


@dataclass(frozen=True, slots=True)
class RobustnessDecision:
    decision: str
    reference_stability: float | None
    constrained_support: DirectionalSupport | None
    deliberative_support: DirectionalSupport | None
    method_rank_agreement: float | None
    reasons: tuple[str, ...]


def state_occupancy(
    topological_states: Sequence[object],
    out_edges: Mapping[object, Sequence[tuple[str, object]]],
    action_probabilities: Mapping[object, Mapping[str, float]],
    root: object,
) -> dict[object, float]:
    """Propagate reach probability forward so merged paths accumulate mass.

    Terminal states have no outgoing edges and simply retain the mass that
    reached them.
    """
    occupancy: dict[object, float] = {root: 1.0}
    for state in topological_states:
        reaching = occupancy.get(state, 0.0)
        if reaching == 0.0:
            continue
        edges = out_edges.get(state) or ()
        if not edges:
            continue
        probabilities = action_probabilities[state]
        for action_key, child in edges:
            mass = reaching * probabilities[action_key]
            occupancy[child] = occupancy.get(child, 0.0) + mass
    return occupancy


def assigned_depth_for_root(root_index: int) -> int:
    """Cycle depths 0, 1, 2 so each depth receives an equal share of roots."""
    if root_index < 0:
        raise ValueError("root_index must be nonnegative")
    return root_index % len(STRATA)


def select_state_for_root(
    candidates: Sequence[StateCandidate],
    *,
    assigned_depth: int,
    minimum_action_count: int = MINIMUM_ACTION_COUNT,
) -> StateCandidate:
    """Pick the highest-occupancy eligible state at (or nearest) the depth.

    Falling back to another depth keeps selection total for roots whose assigned
    depth has no state with enough legal actions to define three strata. Ties
    prefer the shallower depth, then the canonical state order.
    """
    eligible = [
        candidate
        for candidate in candidates
        if len(candidate.advantage_by_action) >= minimum_action_count
    ]
    if not eligible:
        raise ValueError("No candidate state has enough legal actions")
    return min(
        eligible,
        key=lambda candidate: (
            abs(candidate.depth - assigned_depth),
            candidate.depth,
            -candidate.occupancy,
            candidate.canonical_key,
        ),
    )


def select_actions_for_state(
    advantage_by_action: Mapping[str, float],
    *,
    omitted_stratum: str,
) -> tuple[SelectedAction, ...]:
    """Take two of the top, middle, and bottom constrained-advantage actions."""
    if omitted_stratum not in STRATA:
        raise ValueError(f"Unknown stratum: {omitted_stratum!r}")
    if len(advantage_by_action) < MINIMUM_ACTION_COUNT:
        raise ValueError("At least three legal actions are required")
    ordered = sorted(
        advantage_by_action, key=lambda key: (-advantage_by_action[key], key)
    )
    rank_by_stratum = {
        "top": 0,
        "middle": (len(ordered) - 1) // 2,
        "bottom": len(ordered) - 1,
    }
    return tuple(
        SelectedAction(
            action_key=ordered[rank_by_stratum[stratum]],
            stratum=stratum,
            constrained_advantage=advantage_by_action[ordered[rank_by_stratum[stratum]]],
            advantage_rank=rank_by_stratum[stratum],
        )
        for stratum in STRATA
        if stratum != omitted_stratum
    )


def select_benchmark_pairs(
    candidates_by_root: Sequence[Sequence[StateCandidate]],
) -> tuple[RootSelection, ...]:
    """Deterministically choose one state and two actions for every root."""
    if not candidates_by_root:
        raise ValueError("At least one root is required")
    selections = []
    for root_index, candidates in enumerate(candidates_by_root):
        assigned = assigned_depth_for_root(root_index)
        state = select_state_for_root(candidates, assigned_depth=assigned)
        omitted = STRATA[root_index % len(STRATA)]
        selections.append(
            RootSelection(
                root_index=root_index,
                assigned_depth=assigned,
                selected_depth=state.depth,
                canonical_key=state.canonical_key,
                occupancy=state.occupancy,
                omitted_stratum=omitted,
                actions=select_actions_for_state(
                    state.advantage_by_action, omitted_stratum=omitted
                ),
            )
        )
    return tuple(selections)


def uniform_outcome_credit(child_outcomes: Sequence[int]) -> float:
    """Assign the whole post-action outcome to the action, with no baseline."""
    if not child_outcomes:
        raise ValueError("At least one outcome is required")
    return float(child_outcomes[0])


def mc_reset_credit(
    child_outcomes: Sequence[int],
    state_outcomes: Sequence[int],
    *,
    k: int,
) -> float:
    """Estimate Q - V from the first k outcomes of each arm."""
    if k < 1:
        raise ValueError("k must be at least one")
    if len(child_outcomes) < k or len(state_outcomes) < k:
        raise ValueError("Both arms need at least k outcomes")
    q_hat = math.fsum(child_outcomes[:k]) / k
    v_hat = math.fsum(state_outcomes[:k]) / k
    return q_hat - v_hat


def method_credit(
    method: str,
    child_outcomes: Sequence[int],
    state_outcomes: Sequence[int],
) -> float:
    if method == UNIFORM_METHOD:
        return uniform_outcome_credit(child_outcomes)
    if method not in METHOD_NAMES:
        raise ValueError(f"Unknown method: {method!r}")
    return mc_reset_credit(
        child_outcomes, state_outcomes, k=int(method.removeprefix("mc_reset_k"))
    )


def method_fidelity(
    pairs: Sequence[PairSamples],
    method: str,
) -> float | None:
    """Rank-correlate one method's credit with the reference across all pairs."""
    if len(pairs) < 2:
        raise ValueError("Fidelity requires at least two benchmark pairs")
    estimates = [
        method_credit(method, pair.child_outcomes, pair.state_outcomes)
        for pair in pairs
    ]
    reference = [pair.reference_advantage for pair in pairs]
    return spearman_rank_correlation(estimates, reference)


def log2k_slope(fidelity_by_k: Mapping[int, float]) -> float:
    """Least-squares slope of fidelity against log2(K)."""
    if len(fidelity_by_k) < 2:
        raise ValueError("At least two K values are required")
    xs = [math.log2(k) for k in sorted(fidelity_by_k)]
    ys = [fidelity_by_k[k] for k in sorted(fidelity_by_k)]
    x_mean = math.fsum(xs) / len(xs)
    y_mean = math.fsum(ys) / len(ys)
    denominator = math.fsum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        raise ValueError("K values must vary")
    numerator = math.fsum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)
    )
    return numerator / denominator


def _resample(values: Sequence[int], rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.choices(values, k=len(values)))


@dataclass(frozen=True, slots=True)
class BootstrapFidelity:
    """Aligned per-method fidelity draws, one entry per retained replicate."""

    by_method: Mapping[str, tuple[float, ...]]
    # Replicates where some method's credit was constant, so Spearman was
    # undefined. They are dropped whole to keep method comparisons paired.
    dropped_replicate_count: int


def bootstrap_method_fidelity(
    pairs: Sequence[PairSamples],
    *,
    replicates: int,
    seed: int,
) -> BootstrapFidelity:
    """Resample estimator pools to show how unstable each method's fidelity is.

    Every method is scored on the same resampled pools within a replicate, so
    comparisons between methods are paired rather than independent draws.
    """
    if replicates < 1:
        raise ValueError("replicates must be at least one")
    rng = random.Random(seed)
    collected: dict[str, list[float]] = {name: [] for name in METHOD_NAMES}
    dropped = 0
    for _ in range(replicates):
        replica = tuple(
            PairSamples(
                child_outcomes=_resample(pair.child_outcomes, rng),
                state_outcomes=_resample(pair.state_outcomes, rng),
                reference_advantage=pair.reference_advantage,
            )
            for pair in pairs
        )
        fidelity_by_method = {
            name: method_fidelity(replica, name) for name in METHOD_NAMES
        }
        if any(value is None for value in fidelity_by_method.values()):
            dropped += 1
            continue
        for name, value in fidelity_by_method.items():
            collected[name].append(float(value))
    return BootstrapFidelity(
        by_method={name: tuple(values) for name, values in collected.items()},
        dropped_replicate_count=dropped,
    )


def directional_support(bootstrap: BootstrapFidelity) -> DirectionalSupport:
    """Fraction of retained replicates supporting each directional claim."""
    by_method = bootstrap.by_method
    retained = len(by_method[UNIFORM_METHOD])
    if retained == 0:
        raise ValueError("Bootstrap retained no comparable replicates")

    uniform = by_method[UNIFORM_METHOD]
    k16 = by_method["mc_reset_k16"]
    beats = sum(1 for index in range(retained) if k16[index] > uniform[index])
    positive_slopes = sum(
        1
        for index in range(retained)
        if log2k_slope(
            {value: by_method[f"mc_reset_k{value}"][index] for value in K_VALUES}
        )
        > 0.0
    )
    return DirectionalSupport(
        k16_beats_uniform=beats / retained,
        positive_log2k_slope=positive_slopes / retained,
    )


def method_rank_agreement(
    constrained_median_fidelity: Mapping[str, float],
    deliberative_median_fidelity: Mapping[str, float],
) -> float | None:
    """Rank-correlate the six method-level fidelity scores across both setups."""
    missing = [
        name
        for name in METHOD_NAMES
        if name not in constrained_median_fidelity
        or name not in deliberative_median_fidelity
    ]
    if missing:
        raise ValueError(f"Missing method fidelity for {missing!r}")
    return spearman_rank_correlation(
        [constrained_median_fidelity[name] for name in METHOD_NAMES],
        [deliberative_median_fidelity[name] for name in METHOD_NAMES],
    )


def reference_split_half_stability(
    reference_pools: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    replicates: int,
    seed: int,
) -> float | None:
    """Median rank correlation between advantages from two halves of each pool.

    A low value means the sampled reference is too noisy to rank actions, which
    makes any estimator comparison against it uninterpretable.
    """
    if replicates < 1:
        raise ValueError("replicates must be at least one")
    if len(reference_pools) < 2:
        raise ValueError("Stability requires at least two benchmark pairs")
    for child, state in reference_pools:
        if len(child) < 2 or len(state) < 2:
            raise ValueError("Every reference pool needs at least two outcomes")
    rng = random.Random(seed)
    correlations: list[float] = []
    for _ in range(replicates):
        first: list[float] = []
        second: list[float] = []
        for child, state in reference_pools:
            child_shuffled = list(child)
            state_shuffled = list(state)
            rng.shuffle(child_shuffled)
            rng.shuffle(state_shuffled)
            child_half = len(child_shuffled) // 2
            state_half = len(state_shuffled) // 2
            first.append(
                math.fsum(child_shuffled[:child_half]) / child_half
                - math.fsum(state_shuffled[:state_half]) / state_half
            )
            second.append(
                math.fsum(child_shuffled[child_half : 2 * child_half]) / child_half
                - math.fsum(state_shuffled[state_half : 2 * state_half]) / state_half
            )
        correlation = spearman_rank_correlation(first, second)
        if correlation is not None:
            correlations.append(correlation)
    if not correlations:
        return None
    return float(median(correlations))


@dataclass(frozen=True, slots=True)
class ConclusionRobustnessGate:
    minimum_reference_stability: float = 0.5
    minimum_directional_support: float = 0.8
    minimum_method_rank_agreement: float = 0.7

    def evaluate(
        self,
        *,
        reference_stability: float | None,
        constrained_support: DirectionalSupport,
        deliberative_support: DirectionalSupport,
        method_rank_agreement: float | None,
    ) -> RobustnessDecision:
        """Return GO, NO-GO, or INCONCLUSIVE with the reasons behind it.

        An unstable sampled reference yields INCONCLUSIVE rather than NO-GO,
        because it is evidence about our sample size and not about whether the
        two setups disagree.
        """
        if (
            reference_stability is None
            or reference_stability < self.minimum_reference_stability
        ):
            return RobustnessDecision(
                decision="INCONCLUSIVE",
                reference_stability=reference_stability,
                constrained_support=constrained_support,
                deliberative_support=deliberative_support,
                method_rank_agreement=method_rank_agreement,
                reasons=(
                    "Sampled deliberative reference is not stable enough to rank "
                    "actions; estimator comparison is uninterpretable.",
                ),
            )

        reasons: list[str] = []
        for label, support in (
            ("constrained", constrained_support),
            ("deliberative", deliberative_support),
        ):
            if support.k16_beats_uniform < self.minimum_directional_support:
                reasons.append(
                    f"{label}: MC-reset K=16 does not beat Uniform with "
                    f"{self.minimum_directional_support:.0%} bootstrap support"
                )
            if support.positive_log2k_slope < self.minimum_directional_support:
                reasons.append(
                    f"{label}: fidelity does not increase with K at "
                    f"{self.minimum_directional_support:.0%} bootstrap support"
                )
        if (
            method_rank_agreement is None
            or method_rank_agreement < self.minimum_method_rank_agreement
        ):
            reasons.append(
                "method-level fidelity rankings disagree between the two setups"
            )
        return RobustnessDecision(
            decision="GO" if not reasons else "NO-GO",
            reference_stability=reference_stability,
            constrained_support=constrained_support,
            deliberative_support=deliberative_support,
            method_rank_agreement=method_rank_agreement,
            reasons=tuple(reasons),
        )
