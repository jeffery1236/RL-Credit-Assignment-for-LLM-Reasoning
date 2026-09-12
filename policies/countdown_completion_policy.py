"""State-only full-solution prompt and strict project-rule expression scoring.

Key Terms:
- Completion: one free-form model response that reasons and then emits a single
  final arithmetic expression for the current state.
- Project rules: the V1 environment rules (+, -, * only; subtraction is larger
  minus smaller; every intermediate value is a positive integer; each current
  number is used at most once; unused numbers are allowed).
- Rollout outcome: whether a completion reached the target under project rules.

This module is deliberately separate from ``evaluate_countdown_capability``,
whose literature-style prompt permits division and requires consuming every
number. Reusing that contract would change the task as well as the policy, so
rollout outcomes would no longer be comparable with exact constrained credit.
Scoring is split from generation so every parse decision is testable without a
model.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from env.countdown import DEFAULT_RULES, Operator, RuleSet, State

COMPLETION_PROMPT_VERSION = "policy-bridge-completion-v1"

_COMPLETE_ANSWER_TAG = re.compile(
    r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL
)
_ANY_ANSWER_TAG = re.compile(r"</?answer\b[^>]*>", flags=re.IGNORECASE)

# Bounds the parsed tree so a pathological response cannot dominate scoring.
_MAX_EXPRESSION_NODES = 100

_OPERATOR_BY_AST_TYPE = {
    ast.Add: Operator.ADD,
    ast.Sub: Operator.SUB,
    ast.Mult: Operator.MUL,
}


class CompletionStatus(StrEnum):
    SOLVED = "solved"
    MISSING_ANSWER = "missing_answer"
    MALFORMED_ANSWER = "malformed_answer"
    ILLEGAL_EXPRESSION = "illegal_expression"
    WRONG_VALUE = "wrong_value"


@dataclass(frozen=True, slots=True)
class ScoredCompletion:
    """One completion's parse outcome and whether it reached the target."""

    raw_response: str
    status: CompletionStatus
    expression: str | None
    value: int | None  # None whenever the expression never evaluated legally

    @property
    def is_success(self) -> bool:
        return self.status is CompletionStatus.SOLVED


class _IllegalExpression(Exception):
    """The answer parsed as Python but violates the project rules."""


def build_completion_messages(
    state: State,
    rules: RuleSet = DEFAULT_RULES,
) -> tuple[dict[str, str], ...]:
    """Render one state as a solve-the-rest request, with no episode history."""
    operators = ", ".join(operator.value for operator in rules.operators)
    numbers = ", ".join(str(value) for value in state.values)
    return (
        {
            "role": "system",
            "content": (
                "You are a careful mathematical reasoner solving a Countdown "
                "arithmetic problem. Follow the stated rules and the requested "
                "output format exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Target: {state.target}\n"
                f"Current numbers: {numbers}\n"
                f"Allowed operators: {operators}\n"
                "Rules:\n"
                f"- Combine the current numbers using only {operators}.\n"
                "- Subtraction must be the larger number minus the smaller "
                "number.\n"
                f"- Every intermediate result must be a whole number of at "
                f"least {rules.min_intermediate}.\n"
                "- Use each of the current numbers at most once.\n"
                "- You do not need to use every number.\n"
                "Reason about the problem, then give one final arithmetic "
                f"expression equal to {state.target} inside a single answer "
                "tag, for example <answer>(25 + 10) * 2</answer>. The answer "
                "tag must contain only the expression, with no equals sign and "
                "no target value."
            ),
        },
    )


def _evaluate_under_project_rules(
    node: ast.AST,
    leaves: list[int],
    rules: RuleSet,
) -> int:
    """Evaluate one node, rejecting anything the environment would refuse."""
    if isinstance(node, ast.Expression):
        return _evaluate_under_project_rules(node.body, leaves, rules)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise _IllegalExpression("Only integer literals are allowed")
        if node.value < rules.min_intermediate:
            raise _IllegalExpression("Literals must be positive")
        leaves.append(node.value)
        return node.value
    if not isinstance(node, ast.BinOp):
        raise _IllegalExpression(f"Unsupported node {type(node).__name__}")

    operator = _OPERATOR_BY_AST_TYPE.get(type(node.op))
    if operator is None or operator not in rules.operators:
        raise _IllegalExpression(f"Unsupported operator {type(node.op).__name__}")

    left = _evaluate_under_project_rules(node.left, leaves, rules)
    right = _evaluate_under_project_rules(node.right, leaves, rules)
    if operator is Operator.SUB and left <= right:
        raise _IllegalExpression("Subtraction must be larger minus smaller")

    result = (
        left + right
        if operator is Operator.ADD
        else left - right
        if operator is Operator.SUB
        else left * right
    )
    if result < rules.min_intermediate:
        raise _IllegalExpression("Intermediate results must stay positive")
    return result


def _consumes_available_numbers(leaves: Sequence[int], state: State) -> bool:
    remaining = list(state.values)
    for leaf in leaves:
        if leaf not in remaining:
            return False
        remaining.remove(leaf)
    return True


def score_completion(
    raw_response: str,
    state: State,
    rules: RuleSet = DEFAULT_RULES,
) -> ScoredCompletion:
    """Accept only a final closed answer tag holding a target-reaching expression.

    Steps:
    1. Require exactly one usable final answer tag.
    2. Parse its contents as a single arithmetic expression.
    3. Evaluate it under the project rules, rejecting illegal structure.
    4. Require the consumed literals to be available in the current state.
    5. Require the value to equal the target.
    """
    # Step 1: locate the final complete answer tag.
    matches = list(_COMPLETE_ANSWER_TAG.finditer(raw_response))
    if not matches:
        status = (
            CompletionStatus.MALFORMED_ANSWER
            if _ANY_ANSWER_TAG.search(raw_response)
            else CompletionStatus.MISSING_ANSWER
        )
        return ScoredCompletion(raw_response, status, None, None)

    expression = " ".join(matches[-1].group(1).split())
    if not expression:
        return ScoredCompletion(
            raw_response, CompletionStatus.MALFORMED_ANSWER, expression, None
        )

    # Step 2: parse the expression itself.
    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return ScoredCompletion(
            raw_response, CompletionStatus.MALFORMED_ANSWER, expression, None
        )
    if sum(1 for _ in ast.walk(parsed)) > _MAX_EXPRESSION_NODES:
        return ScoredCompletion(
            raw_response, CompletionStatus.ILLEGAL_EXPRESSION, expression, None
        )

    # Steps 3 and 4: evaluate under project rules and check number availability.
    leaves: list[int] = []
    try:
        value = _evaluate_under_project_rules(parsed, leaves, rules)
    except (_IllegalExpression, RecursionError):
        return ScoredCompletion(
            raw_response, CompletionStatus.ILLEGAL_EXPRESSION, expression, None
        )
    if not leaves or not _consumes_available_numbers(leaves, state):
        return ScoredCompletion(
            raw_response, CompletionStatus.ILLEGAL_EXPRESSION, expression, None
        )

    # Step 5: only an exact target match counts as a successful rollout.
    if value != state.target:
        return ScoredCompletion(
            raw_response, CompletionStatus.WRONG_VALUE, expression, value
        )
    return ScoredCompletion(
        raw_response, CompletionStatus.SOLVED, expression, value
    )
