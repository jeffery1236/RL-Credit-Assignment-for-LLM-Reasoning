"""Tests for the project-rule full-solution completion prompt and scoring."""

import json

import pytest

from env.countdown import State
from policies.countdown_completion_policy import (
    CompletionStatus,
    build_completion_messages,
    score_completion,
)

STATE = State(values=(3, 7, 10, 25), target=245)


def test_prompt_states_project_rules_and_carries_no_history():
    rendered = json.dumps(build_completion_messages(STATE))

    assert "245" in rendered
    assert "3, 7, 10, 25" in rendered
    assert "+, -, *" in rendered
    assert "at most once" in rendered
    assert "do not need to use every number" in rendered.lower()
    assert "larger number minus the smaller" in rendered
    assert "<answer>" in rendered
    # Division and the mandatory-use-every-number rule belong to the
    # literature-style capability task, not to this project-rule task.
    assert "Allowed operators: +, -, *" in rendered
    assert " / " not in rendered
    assert "exactly once" not in rendered


def test_scores_target_reaching_expression_using_a_subset_of_numbers():
    scored = score_completion(
        "Let me try.\n<answer>(3 + 7) * 25 - 10 + 5</answer>",
        State(values=(3, 5, 7, 10, 25), target=245),
    )

    assert scored.status is CompletionStatus.SOLVED
    assert scored.is_success is True
    assert scored.value == 245


def test_allows_leaving_numbers_unused():
    scored = score_completion("<answer>7 * 10</answer>", State(values=(3, 7, 10), target=70))

    assert scored.status is CompletionStatus.SOLVED
    assert scored.expression == "7 * 10"


def test_accepts_bare_target_when_state_already_contains_it():
    scored = score_completion("<answer>35</answer>", State(values=(2, 35), target=35))

    assert scored.is_success is True


def test_uses_the_final_answer_tag():
    scored = score_completion(
        "<answer>3 + 7</answer> wait, actually <answer>7 * 10</answer>",
        State(values=(3, 7, 10), target=70),
    )

    assert scored.expression == "7 * 10"
    assert scored.is_success is True


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("no tag at all", CompletionStatus.MISSING_ANSWER),
        ("reasoning then <answer>7 * 10", CompletionStatus.MALFORMED_ANSWER),
        ("<answer></answer>", CompletionStatus.MALFORMED_ANSWER),
        ("<answer>7 * *</answer>", CompletionStatus.MALFORMED_ANSWER),
    ],
)
def test_classifies_missing_and_malformed_answers(response, expected):
    scored = score_completion(response, State(values=(3, 7, 10), target=70))

    assert scored.status is expected
    assert scored.is_success is False


@pytest.mark.parametrize(
    "expression",
    [
        "70 / 1",  # division is outside the project rules
        "3 - 7",  # subtraction must be larger minus smaller
        "7 * 10 * 3 * 3",  # 3 is only available once
        "7 * 99",  # 99 is not in the state
        "-7 * -10",  # negation cannot produce positive intermediates
        "open('x')",  # not arithmetic at all
    ],
)
def test_rejects_expressions_that_violate_project_rules(expression):
    scored = score_completion(
        f"<answer>{expression}</answer>", State(values=(3, 7, 10), target=70)
    )

    assert scored.status in {
        CompletionStatus.ILLEGAL_EXPRESSION,
        CompletionStatus.MALFORMED_ANSWER,
    }
    assert scored.is_success is False


def test_distinguishes_a_legal_expression_with_the_wrong_value():
    scored = score_completion(
        "<answer>3 + 7</answer>", State(values=(3, 7, 10), target=70)
    )

    assert scored.status is CompletionStatus.WRONG_VALUE
    assert scored.value == 10
    assert scored.is_success is False


def test_truncated_reasoning_without_an_answer_is_a_failed_rollout():
    scored = score_completion(
        "Let me enumerate: 7 * 10 = 70, then 70 * 3 = 210, then",
        State(values=(3, 7, 10), target=70),
    )

    assert scored.status is CompletionStatus.MISSING_ANSWER
    assert scored.is_success is False
