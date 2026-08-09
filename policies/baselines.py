"""Three state-only, oracle-free baseline policies, all implementing
Policy[Fraction] so they can be evaluated exactly by
oracle.evaluation.ExactPolicyEvaluator.

None of these policies may consult V* or Q* -- they exist to stand in for
an imperfect real model (e.g. a small LLM), not to play optimally.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping

from env.countdown import Action, State, transition


class UniformPolicy:
    """Equal probability on every allowed action."""

    def action_probs(
        self, state: State, actions: tuple[Action, ...]
    ) -> Mapping[Action, Fraction]:
        n = len(actions)
        equal_probability = Fraction(1, n)
        return {a: equal_probability for a in actions}


def _distance_score(state: State, action: Action) -> int:
    """After taking `action`, how close does the CLOSEST resulting number
    get to the target? Lower is better. An action that wins immediately
    scores 0 (the target itself has distance 0 from itself).
    """
    child = transition(state, action)
    return min(abs(v - child.target) for v in child.values)


class GreedyDistancePolicy:
    """Deterministic-ish: always moves toward the target. Ties (multiple
    actions achieving the same best distance score) split probability
    uniformly among the tied actions; non-tied actions get zero mass.
    """

    def action_probs(
        self, state: State, actions: tuple[Action, ...]
    ) -> Mapping[Action, Fraction]:
        scores = {a: _distance_score(state, a) for a in actions}
        best = min(scores.values())
        winners = [a for a in actions if scores[a] == best]
        tied_winner_probability = Fraction(1, len(winners))
        return {
            a: (tied_winner_probability if a in winners else Fraction(0))
            for a in actions
        }


class EpsilonGreedyDistancePolicy:
    """Mixes (1 - epsilon) greedy-distance mass with epsilon uniform mass.
    Greedy (tied-best) actions receive mass from BOTH terms; every other
    action receives only its share of the uniform term. Probabilities sum
    to exactly (1 - epsilon) + epsilon == 1.
    """

    def __init__(self, epsilon: Fraction = Fraction(1, 10)) -> None:
        if not (Fraction(0) <= epsilon <= Fraction(1)):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = epsilon

    def action_probs(
        self, state: State, actions: tuple[Action, ...]
    ) -> Mapping[Action, Fraction]:
        scores = {a: _distance_score(state, a) for a in actions}
        best = min(scores.values())
        winners = {a for a in actions if scores[a] == best}

        n = len(actions)
        eps = self.epsilon
        greedy_share = (Fraction(1) - eps) * Fraction(1, len(winners))
        uniform_share = eps * Fraction(1, n)

        probs: dict[Action, Fraction] = {}
        for a in actions:
            action_probability = uniform_share
            if a in winners:
                action_probability += greedy_share
            probs[a] = action_probability
        return probs
