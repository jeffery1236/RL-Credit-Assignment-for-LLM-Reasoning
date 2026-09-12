"""State-only prompt and strict response parsing for deliberative actions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from env.countdown import DEFAULT_RULES, Action, RuleSet, State
from policies.allowed_action_llm_policy import MenuOption

DELIBERATIVE_PROMPT_VERSION = "policy-bridge-deliberative-v1"
_COMPLETE_ANSWER_TAG = re.compile(
    r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL
)
_ANY_ANSWER_TAG = re.compile(r"</?answer\b[^>]*>", flags=re.IGNORECASE)


class AnswerStatus(StrEnum):
    VALID = "valid"
    MISSING_ANSWER = "missing_answer"
    MALFORMED_ANSWER = "malformed_answer"
    ILLEGAL_ANSWER = "illegal_answer"


@dataclass(frozen=True, slots=True)
class ParsedDeliberativeResponse:
    raw_response: str
    status: AnswerStatus
    normalized_answer: str | None
    action: Action | None

    @property
    def is_valid(self) -> bool:
        return self.status is AnswerStatus.VALID


def build_deliberative_messages(
    state: State,
    menu: Sequence[MenuOption],
    rules: RuleSet = DEFAULT_RULES,
) -> tuple[dict[str, str], ...]:
    """Render one state and its exact labeled menu, with no episode history."""
    operators = ", ".join(operator.value for operator in rules.operators)
    menu_text = "\n".join(
        f"{option.label}) {option.action.render()}" for option in menu
    )
    return (
        {
            "role": "system",
            "content": (
                "You are choosing the next move in a multi-step Countdown "
                "arithmetic game. Reason before answering, then give the final "
                "move exactly in the requested answer tag."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Target: {state.target}\n"
                f"Current numbers: {', '.join(str(value) for value in state.values)}\n"
                f"Allowed operators: {operators}\n"
                "Game rules:\n"
                "- Choose exactly two current numbers and one allowed operation.\n"
                "- The chosen numbers are removed and the operation result is added.\n"
                "- Subtraction must use the larger number minus the smaller number.\n"
                "- Every result must be positive.\n"
                "- Future moves continue with the new numbers until the target appears.\n"
                "- Remaining numbers may be unused once the target appears.\n"
                "Allowed actions:\n"
                f"{menu_text}\n"
                "You may write ordinary reasoning first. End with exactly one final "
                "tag containing the semantic action, for example "
                "<answer>10 + 25 = 35</answer>."
            ),
        },
    )


def _normalize_answer(value: str) -> str:
    return " ".join(value.split())


def parse_deliberative_response(
    raw_response: str,
    menu: Sequence[MenuOption],
) -> ParsedDeliberativeResponse:
    """Accept only a final, closed answer tag matching one legal rendering."""
    complete_matches = list(_COMPLETE_ANSWER_TAG.finditer(raw_response))
    if not complete_matches:
        status = (
            AnswerStatus.MALFORMED_ANSWER
            if _ANY_ANSWER_TAG.search(raw_response)
            else AnswerStatus.MISSING_ANSWER
        )
        return ParsedDeliberativeResponse(raw_response, status, None, None)

    final = complete_matches[-1]
    if raw_response[final.end() :].strip():
        return ParsedDeliberativeResponse(
            raw_response, AnswerStatus.MALFORMED_ANSWER, None, None
        )
    normalized = _normalize_answer(final.group(1))
    if not normalized:
        return ParsedDeliberativeResponse(
            raw_response, AnswerStatus.MALFORMED_ANSWER, None, None
        )
    action_by_rendering = {
        option.action.render(): option.action
        for option in menu
    }
    action = action_by_rendering.get(normalized)
    if action is None:
        return ParsedDeliberativeResponse(
            raw_response, AnswerStatus.ILLEGAL_ANSWER, normalized, None
        )
    return ParsedDeliberativeResponse(
        raw_response, AnswerStatus.VALID, normalized, action
    )
