"""Tests for literature-style free-form Countdown capability evaluation."""

import json
from pathlib import Path

import pytest

from experiments.evaluate_countdown_capability import (
    CapabilityProblem,
    build_capability_messages,
    load_capability_problems,
    score_response,
    summarize_problem_results,
)


def test_loads_frozen_tinyzero_test_boundary_fixture():
    path = Path("data/countdown-capability/tinyzero-test-12.jsonl")

    problems = load_capability_problems(path)

    assert len(problems) == 12
    assert problems[0] == CapabilityProblem(
        numbers=(79, 17, 60),
        target=36,
        source_row_index=327680,
    )
    assert problems[-1].source_row_index == 327691


def test_prompt_matches_literature_style_countdown_contract():
    problem = CapabilityProblem(
        numbers=(79, 17, 60),
        target=36,
        source_row_index=327680,
    )

    rendered = json.dumps(build_capability_messages(problem))

    assert "each number exactly once" in rendered.lower()
    assert "+, -, *, /" in rendered
    assert "<think>" in rendered
    assert "<answer>" in rendered
    assert "do not include an equals sign" in rendered.lower()
    assert "36" in rendered


def test_scores_correct_expression_using_every_number_once():
    problem = CapabilityProblem(
        numbers=(79, 17, 60),
        target=36,
        source_row_index=327680,
    )

    score = score_response(
        "<think>Subtract and add.</think>\n<answer>(79 - 60) + 17</answer>",
        problem,
    )

    assert score.format_valid is True
    assert score.equation_valid is True
    assert score.correct is True
    assert score.equation == "(79 - 60) + 17"


def test_accepts_exact_fractional_intermediate_values():
    problem = CapabilityProblem(
        numbers=(8, 3, 8, 3),
        target=24,
        source_row_index=0,
    )

    score = score_response(
        "<answer>8 / (3 - 8 / 3)</answer>",
        problem,
    )

    assert score.correct is True


@pytest.mark.parametrize(
    "response",
    [
        "<answer>79 - 60</answer>",
        "<answer>79 + 17 + 60 + 60</answer>",
        "<answer>79 ** 2 - 60</answer>",
        "<answer>open('secret')</answer>",
        "<answer>79.0 - 60 + 17</answer>",
    ],
)
def test_rejects_missing_repeated_or_unsafe_numbers_and_operations(response):
    problem = CapabilityProblem(
        numbers=(79, 17, 60),
        target=36,
        source_row_index=327680,
    )

    score = score_response(response, problem)

    assert score.correct is False
    assert score.equation_valid is False


def test_uses_last_answer_tag_and_distinguishes_wrong_answer():
    problem = CapabilityProblem(
        numbers=(79, 17, 60),
        target=36,
        source_row_index=327680,
    )

    score = score_response(
        "<answer>79 + 17 - 60</answer>\n<answer>79 + 60 - 17</answer>",
        problem,
    )

    assert score.format_valid is True
    assert score.equation_valid is True
    assert score.correct is False
    assert score.equation == "79 + 60 - 17"


def test_summary_reports_accuracy_pass_at_k_all_correct_and_format_rate():
    problem_a = CapabilityProblem((1, 2), 3, 1)
    problem_b = CapabilityProblem((2, 3), 6, 2)
    results = [
        {
            "problem": problem_a,
            "scores": [
                score_response("<answer>1 + 2</answer>", problem_a),
                score_response("no answer", problem_a),
            ],
        },
        {
            "problem": problem_b,
            "scores": [
                score_response("<answer>2 * 3</answer>", problem_b),
                score_response("<answer>3 * 2</answer>", problem_b),
            ],
        },
    ]

    summary = summarize_problem_results(results)

    assert summary["problem_count"] == 2
    assert summary["sample_count"] == 4
    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["pass_at_k"] == pytest.approx(1.0)
    assert summary["all_correct_at_k"] == pytest.approx(0.5)
    assert summary["format_rate"] == pytest.approx(0.75)
