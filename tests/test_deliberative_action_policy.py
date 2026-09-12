"""Tests for state-only deliberative action prompts and strict answer parsing."""

import json

import pytest

from env.countdown import State, allowed_actions
from policies.allowed_action_llm_policy import build_allowed_action_menu
from policies.deliberative_action_policy import (
    AnswerStatus,
    build_deliberative_messages,
    parse_deliberative_response,
)


LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def _menu():
    state = State(values=(3, 7, 10, 25), target=245)
    return state, build_allowed_action_menu(state, allowed_actions(state), LABELS)


def test_deliberative_prompt_uses_current_state_menu_and_no_history():
    state, menu = _menu()

    rendered = json.dumps(build_deliberative_messages(state, menu))

    assert "245" in rendered
    assert "3, 7, 10, 25" in rendered
    assert all(option.action.render() in rendered for option in menu)
    assert "<answer>10 + 25 = 35</answer>" in rendered
    assert "reason before answering" in rendered.lower()
    assert "previous state" not in rendered.lower()
    assert "previous action" not in rendered.lower()
    assert "trajectory" not in rendered.lower()
    assert "chain-of-thought" not in rendered.lower()


def test_parser_accepts_only_final_tag_with_normalized_legal_action():
    _, menu = _menu()

    result = parse_deliberative_response(
        "<answer>bad</answer>\nReasoning...\n<answer> 10   +  25 = 35 </answer>",
        menu,
    )

    assert result.status is AnswerStatus.VALID
    assert result.action is not None
    assert result.action.render() == "10 + 25 = 35"
    assert result.normalized_answer == "10 + 25 = 35"


@pytest.mark.parametrize(
    ("response", "status"),
    [
        ("I choose 10 + 25 = 35", AnswerStatus.MISSING_ANSWER),
        ("<answer>10 + 25 = 35", AnswerStatus.MALFORMED_ANSWER),
        ("<answer></answer>", AnswerStatus.MALFORMED_ANSWER),
        ("<answer>10 + 25 = 34</answer>", AnswerStatus.ILLEGAL_ANSWER),
        ("<answer>25 + 10 = 35</answer>", AnswerStatus.ILLEGAL_ANSWER),
        ("<answer>10 / 25 = 0.4</answer>", AnswerStatus.ILLEGAL_ANSWER),
        ("<answer>10 + 25 = 35</answer> trailing", AnswerStatus.MALFORMED_ANSWER),
    ],
)
def test_parser_records_every_invalid_answer_category(response, status):
    _, menu = _menu()

    result = parse_deliberative_response(response, menu)

    assert result.status is status
    assert result.action is None
