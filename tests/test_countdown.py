"""Tests for the immutable arithmetic kernel in env/countdown.py."""

import itertools

import pytest

from env.countdown import (
    Action,
    CountdownError,
    IllegalActionError,
    InvalidProblemError,
    Operator,
    Problem,
    State,
    TerminalStateError,
    is_success,
    is_terminal,
    allowed_actions,
    reward,
    transition,
)


def make_state(values, target):
    return State(values=tuple(sorted(values)), target=target)


# ---------------------------------------------------------------------------
# Success / terminality / reward
# ---------------------------------------------------------------------------


def test_early_success_target_alone():
    s = make_state([1, 2], 3)
    actions = allowed_actions(s)
    add = Action(Operator.ADD, 1, 2)
    assert add in actions
    result = transition(s, add)
    assert result.values == (3,)
    assert reward(result) == 1
    assert is_terminal(result)
    assert allowed_actions(result) == ()


def test_success_mid_bag_does_not_require_using_all_numbers():
    # Target 3 appears alongside 50 -- this IS success, and no further
    # moves are offered even though 50 remains unused.
    s = make_state([1, 2, 50], 3)
    add = Action(Operator.ADD, 1, 2)
    result = transition(s, add)
    assert result.values == (3, 50)
    assert is_success(result)
    assert reward(result) == 1
    assert is_terminal(result)
    assert allowed_actions(result) == ()


def test_terminal_failure_no_moves_left():
    s = make_state([5], 3)
    assert is_terminal(s)
    assert reward(s) == 0
    assert allowed_actions(s) == ()


def test_every_success_state_is_terminal_and_actionless():
    for values, target in [((3,), 3), ((3, 50), 3), ((1, 2, 3), 3)]:
        s = make_state(values, target)
        assert is_success(s)
        assert is_terminal(s)
        assert allowed_actions(s) == ()


# ---------------------------------------------------------------------------
# Legal action enumeration and deduplication
# ---------------------------------------------------------------------------


def test_duplicate_operands_deduplicated():
    # (3, 3, 7): pairs are (3,3), (3,7), (3,7).
    # From (3,3): 3+3=6, 3x3=9 (no subtraction, equal operands).
    # From (3,7) x2 (deduped to 1): 3+7=10, 3x7=21, 7-3=4.
    s = make_state([3, 3, 7], 99)
    actions = allowed_actions(s)
    expected = {
        Action(Operator.ADD, 3, 3),
        Action(Operator.MUL, 3, 3),
        Action(Operator.ADD, 3, 7),
        Action(Operator.MUL, 3, 7),
        Action(Operator.SUB, 7, 3),
    }
    assert set(actions) == expected
    assert len(actions) == 5


def test_duplicate_consumption_removes_both_copies():
    s = make_state([3, 3, 7], 99)
    result = transition(s, Action(Operator.ADD, 3, 3))
    assert result.values == (6, 7)


def test_permutation_invariance_of_state_and_actions():
    s1 = State(values=tuple(sorted([10, 3, 25, 7])), target=245)
    s2 = State(values=tuple(sorted([3, 7, 10, 25])), target=245)
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert allowed_actions(s1) == allowed_actions(s2)


def test_deterministic_action_ordering_is_stable_across_calls():
    s = make_state([3, 7, 10, 25], 245)
    first = allowed_actions(s)
    second = allowed_actions(s)
    assert first == second
    # sorted by (operator symbol, left, right)
    keys = [a.sort_key() for a in first]
    assert keys == sorted(keys)


def test_subtraction_requires_strictly_larger_minus_smaller():
    s = make_state([4, 4], 99)
    actions = allowed_actions(s)
    # Equal operands: no SUB action should be generated (would yield 0).
    assert all(a.operator is not Operator.SUB for a in actions)
    for a in actions:
        if a.operator is Operator.SUB:
            assert a.left > a.right


def test_subtraction_action_construction_rejects_non_positive_result():
    with pytest.raises(InvalidProblemError):
        Action(Operator.SUB, 4, 4)
    with pytest.raises(InvalidProblemError):
        Action(Operator.SUB, 3, 5)  # wrong order: left must be larger


def test_cardinality_invariant_holds_for_all_legal_actions():
    s = make_state([2, 3, 5, 7, 11], 999)  # unsolvable target -> no early exit
    for action in allowed_actions(s):
        result = transition(s, action)
        assert len(result.values) == len(s.values) - 1


# ---------------------------------------------------------------------------
# Illegal-move / error-path rejection
# ---------------------------------------------------------------------------


def test_stale_action_rejected():
    s = make_state([2, 3, 5, 7, 11], 999)
    stale = Action(Operator.ADD, 100, 200)
    with pytest.raises(IllegalActionError):
        transition(s, stale)


def test_acting_on_terminal_state_raises():
    s = make_state([3], 3)  # already success
    with pytest.raises(TerminalStateError):
        transition(s, Action(Operator.ADD, 1, 2))

    s2 = make_state([5], 3)  # terminal failure, no allowed actions
    with pytest.raises(TerminalStateError):
        transition(s2, Action(Operator.ADD, 1, 2))


def test_duplicate_operand_multiplicity_enforced():
    # Bag has only one 2 -- 2x2 must be rejected even though 2 IS present.
    s = make_state([2, 7], 99)
    with pytest.raises(IllegalActionError):
        transition(s, Action(Operator.ADD, 2, 2))


def test_state_must_be_constructed_sorted():
    with pytest.raises(InvalidProblemError):
        State(values=(7, 3, 10), target=245)


def test_problem_requires_at_least_two_values():
    with pytest.raises(InvalidProblemError):
        Problem(values=(5,), target=10)


def test_problem_rejects_non_positive_values_and_target():
    with pytest.raises(InvalidProblemError):
        Problem(values=(3, -1), target=10)
    with pytest.raises(InvalidProblemError):
        Problem(values=(3, 4), target=0)


# ---------------------------------------------------------------------------
# Worked example from the plan: (3,7,10,25) -> 245
# ---------------------------------------------------------------------------


def test_worked_example_25_times_10_reachable():
    s = make_state([3, 7, 10, 25], 245)
    actions = allowed_actions(s)
    mul_25_10 = Action(Operator.MUL, 10, 25)
    assert mul_25_10 in actions
    result = transition(s, mul_25_10)
    assert result.values == (3, 7, 250)
