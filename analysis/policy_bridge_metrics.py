"""Pure, dependency-free analysis functions for the staged policy bridge."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import median
from typing import Hashable, Mapping, Sequence, TypeVar

ActionKey = TypeVar("ActionKey", bound=Hashable)


def _validate_distribution(distribution: Mapping[ActionKey, float]) -> None:
    if not distribution:
        raise ValueError("Action distribution must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in distribution.values()):
        raise ValueError("Action probabilities must be finite and nonnegative")
    if not math.isclose(math.fsum(distribution.values()), 1.0, abs_tol=1e-6):
        raise ValueError("Action probabilities must sum to one")


def average_action_distributions(
    distributions: Sequence[Mapping[ActionKey, float]],
) -> dict[ActionKey, float]:
    if not distributions:
        raise ValueError("At least one distribution is required")
    first_keys = set(distributions[0])
    for distribution in distributions:
        _validate_distribution(distribution)
        if set(distribution) != first_keys:
            raise ValueError("All distributions must have the same legal action set")
    count = len(distributions)
    return {
        action: math.fsum(distribution[action] for distribution in distributions)
        / count
        for action in distributions[0]
    }


def total_variation_distance(
    first: Mapping[ActionKey, float],
    second: Mapping[ActionKey, float],
) -> float:
    _validate_distribution(first)
    _validate_distribution(second)
    if set(first) != set(second):
        raise ValueError("TV distance requires the same legal action set")
    return 0.5 * math.fsum(abs(first[action] - second[action]) for action in first)


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for original_index, _ in indexed[position:end]:
            ranks[original_index] = rank
        position = end
    return ranks


def spearman_rank_correlation(
    first: Sequence[float],
    second: Sequence[float],
) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman correlation requires equally sized sequences of 2+")
    if any(not math.isfinite(value) for value in (*first, *second)):
        raise ValueError("Spearman inputs must be finite")
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    first_mean = math.fsum(first_ranks) / len(first_ranks)
    second_mean = math.fsum(second_ranks) / len(second_ranks)
    numerator = math.fsum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first_ranks, second_ranks, strict=True)
    )
    left_norm = math.fsum((value - first_mean) ** 2 for value in first_ranks)
    right_norm = math.fsum((value - second_mean) ** 2 for value in second_ranks)
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / math.sqrt(left_norm * right_norm)


def solution_preserving_probability(
    probabilities: Mapping[ActionKey, float],
    q_star: Mapping[ActionKey, int],
) -> float:
    _validate_distribution(probabilities)
    if set(probabilities) != set(q_star):
        raise ValueError("Q* must cover the same legal action set")
    if any(value not in (0, 1) for value in q_star.values()):
        raise ValueError("Q* values must be binary")
    return math.fsum(
        probability
        for action, probability in probabilities.items()
        if q_star[action] == 1
    )


def top_actions(distribution: Mapping[ActionKey, float]) -> tuple[ActionKey, ...]:
    _validate_distribution(distribution)
    maximum = max(distribution.values())
    return tuple(action for action, value in distribution.items() if value == maximum)


@dataclass(frozen=True, slots=True)
class GateDecision:
    passed: bool
    valid_rate: float
    median_spearman: float | None
    median_tv: float | None


@dataclass(frozen=True, slots=True)
class BridgeStageAGate:
    minimum_valid_rate: float = 0.9
    minimum_median_spearman: float = 0.5
    maximum_median_tv: float = 0.35

    def evaluate(
        self,
        *,
        valid_rate: float,
        spearman_values: Sequence[float | None],
        tv_values: Sequence[float | None],
    ) -> GateDecision:
        if not 0.0 <= valid_rate <= 1.0:
            raise ValueError("valid_rate must be in [0, 1]")
        median_spearman = (
            float(median(value for value in spearman_values if value is not None))
            if spearman_values and all(value is not None for value in spearman_values)
            else None
        )
        median_tv = (
            float(median(value for value in tv_values if value is not None))
            if tv_values and all(value is not None for value in tv_values)
            else None
        )
        return GateDecision(
            passed=(
                valid_rate >= self.minimum_valid_rate
                and median_spearman is not None
                and median_spearman >= self.minimum_median_spearman
                and median_tv is not None
                and median_tv <= self.maximum_median_tv
            ),
            valid_rate=valid_rate,
            median_spearman=median_spearman,
            median_tv=median_tv,
        )


@dataclass(frozen=True, slots=True)
class BridgeStageBGate:
    minimum_median_advantage_correlation: float = 0.5
    minimum_sign_agreement: float = 0.7

    def evaluate(
        self,
        *,
        advantage_correlations: Sequence[float | None],
        sign_agreement: float | None,
    ) -> "StageBGateDecision":
        median_correlation = (
            float(median(value for value in advantage_correlations if value is not None))
            if advantage_correlations
            and all(value is not None for value in advantage_correlations)
            else None
        )
        return StageBGateDecision(
            passed=(
                median_correlation is not None
                and median_correlation >= self.minimum_median_advantage_correlation
                and sign_agreement is not None
                and sign_agreement >= self.minimum_sign_agreement
            ),
            median_advantage_correlation=median_correlation,
            sign_agreement=sign_agreement,
        )


@dataclass(frozen=True, slots=True)
class StageBGateDecision:
    passed: bool
    median_advantage_correlation: float | None
    sign_agreement: float | None


def _canonical_state_key(row: Mapping[str, object]) -> tuple[int, tuple[int, ...]]:
    state = row["state"]
    if not isinstance(state, Mapping):
        raise ValueError("Root row must contain a state object")
    return int(state["target"]), tuple(int(value) for value in state["values"])


def select_stage_b_roots(
    root_results: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    if len(root_results) < 3:
        raise ValueError("Stage B requires at least three Stage A roots")
    ordered = sorted(
        root_results,
        key=lambda row: (float(row["tv_distance"]), _canonical_state_key(row)),
    )
    return ordered[0], ordered[(len(ordered) - 1) // 2], ordered[-1]


def bootstrap_response_counts(
    responses: Sequence[Mapping[str, object]],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str | None, int], ...]:
    if replicates < 1:
        raise ValueError("replicates must be at least one")
    by_ordering: dict[str, list[Mapping[str, object]]] = {}
    for response in responses:
        ordering_id = response.get("ordering_id")
        if not isinstance(ordering_id, str):
            raise ValueError("Every response needs an ordering_id")
        by_ordering.setdefault(ordering_id, []).append(response)
    if not by_ordering:
        raise ValueError("At least one response is required")
    rng = random.Random(seed)
    result = []
    for _ in range(replicates):
        counts: dict[str | None, int] = {}
        for values in by_ordering.values():
            for response in rng.choices(values, k=len(values)):
                action = response.get("action")
                if action is not None and not isinstance(action, str):
                    raise ValueError("Response action must be a string or null")
                counts[action] = counts.get(action, 0) + 1
        result.append(counts)
    return tuple(result)
