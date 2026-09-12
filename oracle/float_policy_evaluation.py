"""Floating-point policy values over an existing complete reasoning DAG.

This evaluator is intentionally separate from ExactPolicyEvaluator. Model
softmax probabilities are floating point, while the Phase 1 evaluator preserves
literal Fraction equality for hand-coded policies.
"""

from __future__ import annotations

import math
from typing import Mapping

from env.countdown import (
    DEFAULT_RULES,
    Action,
    RuleSet,
    State,
    allowed_actions,
    is_success,
    is_terminal,
    transition,
)
from oracle.evaluation import Policy, PolicyValidationError, ReasoningDag

PROBABILITY_TOLERANCE = 1e-6
NUMERICAL_TOLERANCE = 1e-9


class FloatPolicyEvaluator:
    """Numerically evaluate a stationary Policy[float] over one complete DAG."""

    def __init__(
        self,
        policy: Policy[float],
        dag: ReasoningDag,
        rules: RuleSet = DEFAULT_RULES,
    ) -> None:
        self._policy = policy
        self._dag = dag
        self._rules = rules
        self._v_pi_cache: dict[State, float] = {}
        self._action_probs_cache: dict[State, Mapping[Action, float]] = {}
        self._compute_all()

    def _get_action_probs(self, state: State) -> Mapping[Action, float]:
        cached = self._action_probs_cache.get(state)
        if cached is not None:
            return cached

        actions = allowed_actions(state, self._rules)
        if not actions:
            return {}

        probabilities = dict(self._policy.action_probs(state, actions))
        if set(probabilities) != set(actions):
            raise PolicyValidationError(
                f"Policy returned actions {set(probabilities)!r}; "
                f"expected {set(actions)!r} at {state!r}"
            )
        if any(not math.isfinite(value) for value in probabilities.values()):
            raise PolicyValidationError(
                f"Policy returned a non-finite probability at {state!r}"
            )
        if any(value < 0.0 for value in probabilities.values()):
            raise PolicyValidationError(
                f"Policy returned a negative probability at {state!r}"
            )

        total = math.fsum(probabilities.values())
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            raise PolicyValidationError(
                f"Policy probabilities at {state!r} sum to {total}, not 1"
            )

        self._action_probs_cache[state] = probabilities
        return probabilities

    def _compute_all(self) -> None:
        for state in reversed(self._dag.topological_order()):
            if is_success(state):
                self._v_pi_cache[state] = 1.0
            elif is_terminal(state, self._rules):
                self._v_pi_cache[state] = 0.0
            else:
                probabilities = self._get_action_probs(state)
                value = math.fsum(
                    probabilities[edge.action]
                    * self._v_pi_cache[edge.target_state]
                    for edge in self._dag.out_edges(state)
                )
                if not -NUMERICAL_TOLERANCE <= value <= 1.0 + NUMERICAL_TOLERANCE:
                    raise PolicyValidationError(
                        f"Computed Vpi={value} outside [0, 1] at {state!r}"
                    )
                self._v_pi_cache[state] = value

    def v_pi(self, state: State) -> float:
        return self._v_pi_cache[state]

    def q_pi(self, state: State, action: Action) -> float:
        return self._v_pi_cache[transition(state, action, self._rules)]

    def a_pi(self, state: State, action: Action) -> float:
        return self.q_pi(state, action) - self.v_pi(state)

    def advantages(self, state: State) -> dict[Action, float]:
        return {
            action: self.a_pi(state, action)
            for action in allowed_actions(state, self._rules)
        }

    def action_probs(self, state: State) -> Mapping[Action, float]:
        if is_terminal(state, self._rules):
            return {}
        return self._get_action_probs(state)
