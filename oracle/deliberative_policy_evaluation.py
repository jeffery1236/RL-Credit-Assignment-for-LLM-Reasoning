"""DAG evaluator for sampled policies whose invalid output ends the episode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from env.countdown import DEFAULT_RULES, Action, RuleSet, State, is_success, is_terminal
from oracle.evaluation import PolicyValidationError, ReasoningDag
from oracle.float_policy_evaluation import NUMERICAL_TOLERANCE, PROBABILITY_TOLERANCE


@dataclass(frozen=True, slots=True)
class DeliberativeStateProbabilities:
    action_probabilities: Mapping[Action, float]
    invalid_probability: float


class DeliberativePolicyEvaluator:
    """Evaluate legal action mass; invalid output has immediate terminal value zero."""

    def __init__(
        self,
        policy: Mapping[State, Mapping[str, object] | DeliberativeStateProbabilities],
        dag: ReasoningDag,
        rules: RuleSet = DEFAULT_RULES,
    ) -> None:
        self._dag = dag
        self._rules = rules
        self._policy = {
            state: self._coerce_probabilities(value) for state, value in policy.items()
        }
        self._action_probs_cache: dict[State, Mapping[Action, float]] = {}
        self._invalid_probs: dict[State, float] = {}
        self._values: dict[State, float] = {}
        self._compute_all()

    @staticmethod
    def _coerce_probabilities(
        value: Mapping[str, object] | DeliberativeStateProbabilities,
    ) -> DeliberativeStateProbabilities:
        if isinstance(value, DeliberativeStateProbabilities):
            return value
        try:
            return DeliberativeStateProbabilities(
                action_probabilities=value["action_probabilities"],  # type: ignore[arg-type]
                invalid_probability=float(value["invalid_probability"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PolicyValidationError(
                f"Malformed deliberative policy probabilities: {error}"
            ) from error

    def _get_probabilities(
        self, state: State
    ) -> tuple[Mapping[Action, float], float]:
        if state in self._action_probs_cache:
            return self._action_probs_cache[state], self._invalid_probs[state]
        expected_actions = {edge.action for edge in self._dag.out_edges(state)}
        try:
            record = self._policy[state]
        except KeyError as error:
            raise PolicyValidationError(
                f"No deliberative response probabilities for {state!r}"
            ) from error
        probabilities = dict(record.action_probabilities)
        if set(probabilities) != expected_actions:
            raise PolicyValidationError("Deliberative policy action set does not match DAG")
        values = (*probabilities.values(), record.invalid_probability)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise PolicyValidationError(
                "Deliberative policy probabilities must be finite and nonnegative"
            )
        if abs(math.fsum(values) - 1.0) > PROBABILITY_TOLERANCE:
            raise PolicyValidationError(
                "Legal and invalid deliberative probability must sum to one"
            )
        self._action_probs_cache[state] = probabilities
        self._invalid_probs[state] = record.invalid_probability
        return probabilities, record.invalid_probability

    def _compute_all(self) -> None:
        for state in reversed(self._dag.topological_order()):
            if is_success(state):
                self._values[state] = 1.0
            elif is_terminal(state, self._rules):
                self._values[state] = 0.0
            else:
                probabilities, _ = self._get_probabilities(state)
                self._values[state] = math.fsum(
                    probabilities[edge.action] * self._values[edge.target_state]
                    for edge in self._dag.out_edges(state)
                )

    def v_pi(self, state: State) -> float:
        return self._values[state]

    def q_pi(self, state: State, action: Action) -> float:
        for edge in self._dag.out_edges(state):
            if edge.action == action:
                return self._values[edge.target_state]
        raise PolicyValidationError(f"Action {action!r} is illegal at {state!r}")

    def a_pi(self, state: State, action: Action) -> float:
        return self.q_pi(state, action) - self.v_pi(state)

    def invalid_output_advantage(self, state: State) -> float:
        if is_terminal(state, self._rules):
            raise PolicyValidationError("Terminal states have no invalid-output action")
        return -self.v_pi(state)

    def action_probs(self, state: State) -> Mapping[Action, float]:
        if is_terminal(state, self._rules):
            return {}
        return self._get_probabilities(state)[0]

    def invalid_probability(self, state: State) -> float:
        if is_terminal(state, self._rules):
            return 0.0
        return self._get_probabilities(state)[1]

    def validate_invariants(self) -> None:
        for state in self._dag.topological_order():
            if is_terminal(state, self._rules):
                continue
            probabilities, _ = self._get_probabilities(state)
            value = self.v_pi(state)
            bellman = math.fsum(
                probabilities[edge.action] * self.v_pi(edge.target_state)
                for edge in self._dag.out_edges(state)
            )
            if abs(value - bellman) > NUMERICAL_TOLERANCE:
                raise PolicyValidationError("Deliberative Bellman invariant failed")
