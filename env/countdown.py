"""Core Countdown-style arithmetic reasoning environment.

This module owns everything about V1 game dynamics: the immutable data types
(Problem, RuleSet, State, Operator, Action, StepResult), rule enforcement,
legal-action generation, pure state transitions, and a thin stateful
CountdownEnv wrapper.

V1 rules (locked, see docs/project_plan.md and the Phase 1 plan):
  - Operators: +, -, x only. No division.
  - All intermediate values are strictly positive integers. a - b is legal
    only when a > b (so results are never zero or negative).
  - Each number is consumed at most once per action.
  - Success is immediate: the episode ends the moment the target value
    appears anywhere in the current bag. The remaining numbers need not be
    used.

State canonicalization is the single most important correctness property in
this codebase: a state is (target, sorted tuple of values), duplicates
preserved. Two bags that are permutations of each other MUST compare equal
and hash equally, or the DAG built in oracle/evaluation.py degenerates into
an exponentially larger tree and every downstream oracle computation becomes
either wrong or intractable.

Actions are identified by (operator, operand VALUES) -- never by list
index. This is what makes duplicate-number choices collapse into a single
action (e.g. in bag (3, 3, 7), "add either 3 to the 7" is one action, not
two) while still keeping two operators that reach the same child as two
distinct actions (e.g. in bag (2, 2), 2+2=4 and 2x2=4 are different
decisions that both land on child state (4,) -- this is why the reasoning
graph is a multigraph, not a simple graph).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum


class CountdownError(Exception):
    """Base class for all Countdown environment errors."""


class InvalidProblemError(CountdownError):
    """A Problem or State was constructed with invalid values."""


class IllegalActionError(CountdownError):
    """An action was proposed or applied that violates the V1 rules."""


class TerminalStateError(CountdownError):
    """An action was applied to (or requested from) a terminal state."""


class Operator(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"


@dataclass(frozen=True, slots=True)
class RuleSet:
    """The V1 rule configuration. There is deliberately no `allow_div` flag:
    division is out of scope for V1, not merely disabled by default.
    """

    operators: tuple[Operator, ...] = (Operator.ADD, Operator.SUB, Operator.MUL)
    min_intermediate: int = 1

    def __post_init__(self) -> None:
        if self.min_intermediate < 1:
            raise InvalidProblemError("min_intermediate must be >= 1")
        if not self.operators:
            raise InvalidProblemError("RuleSet must allow at least one operator")


DEFAULT_RULES = RuleSet()


@dataclass(frozen=True, slots=True)
class Problem:
    """A puzzle: a starting bag of numbers plus a target."""

    values: tuple[int, ...]
    target: int

    def __post_init__(self) -> None:
        if len(self.values) < 2:
            raise InvalidProblemError("Problem needs at least 2 starting values")
        if any(v < 1 for v in self.values):
            raise InvalidProblemError("Problem values must be strictly positive")
        if self.target < 1:
            raise InvalidProblemError("Problem target must be strictly positive")
        if tuple(sorted(self.values)) != self.values:
            object.__setattr__(self, "values", tuple(sorted(self.values)))

    def to_state(self) -> "State":
        return State(values=self.values, target=self.target)


@dataclass(frozen=True, slots=True)
class State:
    """A point mid-puzzle: the current bag of numbers plus the (unchanging)
    target. `values` is ALWAYS sorted ascending; duplicates are preserved.
    This canonical form is what allows two different action sequences that
    reach "the same" bag to be recognized as the identical state.
    """

    values: tuple[int, ...]
    target: int

    def __post_init__(self) -> None:
        if tuple(sorted(self.values)) != self.values:
            raise InvalidProblemError(
                f"State.values must be sorted ascending, got {self.values}"
            )
        if any(v < 1 for v in self.values):
            raise InvalidProblemError("State values must be strictly positive")
        if self.target < 1:
            raise InvalidProblemError("State target must be strictly positive")

    @property
    def depth(self) -> int:
        """Not meaningful in isolation -- see DagNode.depth, which is
        relative to a specific root. Provided here only as a convenience
        alias for len(values), which callers sometimes want directly.
        """
        return len(self.values)


@dataclass(frozen=True, slots=True)
class Action:
    """An action is identified by (operator, operand VALUES), never by
    list indices. `result` is a derived property, not a stored field --
    storing it would let a corrupted Action claim e.g. 3 + 7 == 11, and
    equality/hashing would then treat two contradictory actions as
    different rather than catching the corruption.

    Operand ordering is canonicalized so that logically identical actions
    always construct to equal objects:
      - ADD, MUL: left <= right (commutative, so order doesn't matter --
        pick the smaller operand as `left` by convention).
      - SUB: left is the larger operand, right is the smaller. b - a is
        represented as Action(SUB, left=b, right=a) with b > a, and the
        result b - a is always >= 1.
    """

    operator: Operator
    left: int
    right: int

    def __post_init__(self) -> None:
        if self.operator in (Operator.ADD, Operator.MUL):
            if self.left > self.right:
                raise InvalidProblemError(
                    f"{self.operator} requires left <= right, got "
                    f"left={self.left}, right={self.right}"
                )
        elif self.operator is Operator.SUB:
            if self.left <= self.right:
                raise InvalidProblemError(
                    f"SUB requires left > right (larger minus smaller), got "
                    f"left={self.left}, right={self.right}"
                )
        else:
            raise InvalidProblemError(f"Unsupported operator: {self.operator}")

    @property
    def result(self) -> int:
        if self.operator is Operator.ADD:
            return self.left + self.right
        if self.operator is Operator.MUL:
            return self.left * self.right
        if self.operator is Operator.SUB:
            return self.left - self.right
        raise InvalidProblemError(f"Unsupported operator: {self.operator}")

    @property
    def operands(self) -> tuple[int, int]:
        """The two consumed values, in no particular arithmetic role --
        used by transition() to remove them from the bag regardless of
        which one is "left" vs "right" for the operator's semantics.
        """
        return (self.left, self.right)

    def render(self) -> str:
        return f"{self.left} {self.operator.value} {self.right} = {self.result}"

    def sort_key(self) -> tuple[str, int, int]:
        """Deterministic ordering key: (operator symbol, left, right)."""
        return (self.operator.value, self.left, self.right)


@dataclass(frozen=True, slots=True)
class StepResult:
    state: State
    reward: int
    terminal: bool


def is_success(state: State) -> bool:
    """The target has been reached: it appears anywhere in the bag. Success
    is checked independently of remaining bag size -- a bag holding exactly
    one value equal to the target is a success, not a (coincidental)
    terminal failure.
    """
    return state.target in state.values


def allowed_actions(state: State, rules: RuleSet = DEFAULT_RULES) -> tuple[Action, ...]:
    """What moves can be made right now? All legal, deduplicated actions
    from `state`, in a fixed deterministic order (sorted by operator
    symbol, then left, then right).

    A won game has no moves: if `state` is already a success, this returns
    an empty tuple even if len(values) >= 2. Likewise a bag of fewer than 2
    values has no moves.

    Deduplication: iterating unordered index pairs (i < j) over the sorted
    values tuple naturally produces the same Action object for two pairs
    that select the same operand values (e.g. both (0,2) and (1,2) in
    (3, 3, 7) produce Action(ADD, 3, 7)) -- collecting into a set merges
    them. Two different operators reaching the same result value (e.g.
    2+2 and 2x2, both =4) remain separate Action objects because operator
    is part of the identity, which is exactly what keeps the DAG a
    multigraph rather than losing an edge.
    """
    if is_success(state):
        return ()
    values = state.values
    n = len(values)
    if n < 2:
        return ()

    actions: set[Action] = set()
    for i in range(n):
        for j in range(i + 1, n):
            a, b = values[i], values[j]  # a <= b, since values is sorted
            if Operator.ADD in rules.operators:
                actions.add(Action(Operator.ADD, a, b))
            if Operator.MUL in rules.operators:
                actions.add(Action(Operator.MUL, a, b))
            if Operator.SUB in rules.operators and b > a:
                actions.add(Action(Operator.SUB, b, a))

    return tuple(sorted(actions, key=Action.sort_key))


def is_terminal(state: State, rules: RuleSet = DEFAULT_RULES) -> bool:
    """Terminal means the episode is over: either success, or no legal
    actions remain.
    """
    return is_success(state) or len(allowed_actions(state, rules)) == 0


def reward(state: State) -> int:
    return 1 if is_success(state) else 0


def transition(
    state: State, action: Action, rules: RuleSet = DEFAULT_RULES
) -> State:
    """Apply `action` to `state`, returning the resulting State.

    Raises TerminalStateError if `state` is already terminal (acting after
    the game ends is a programming error, not a no-op). Raises
    IllegalActionError if the action's operands aren't actually available
    in the bag with sufficient multiplicity (a stale action constructed
    against a different state, or a bag containing only one copy of a
    value an action claims to consume twice), or if the result would
    violate the positivity rule.
    """
    if is_terminal(state, rules):
        raise TerminalStateError(
            f"Cannot act on terminal state {state!r} (is_success="
            f"{is_success(state)}, allowed_actions={allowed_actions(state, rules)})"
        )

    left, right = action.operands
    counts = Counter(state.values)
    needed = Counter([left, right])
    for value, count in needed.items():
        if counts[value] < count:
            raise IllegalActionError(
                f"Action {action.render()} requires {needed} but state "
                f"{state.values} only has {dict(counts)}"
            )

    result = action.result
    if result < rules.min_intermediate:
        raise IllegalActionError(
            f"Action {action.render()} produces {result}, which violates "
            f"min_intermediate={rules.min_intermediate}"
        )

    remaining = counts.copy()
    remaining[left] -= 1
    remaining[right] -= 1
    remaining[result] += 1

    new_values: list[int] = []
    for value, count in remaining.items():
        if count > 0:
            new_values.extend([value] * count)
    new_values.sort()

    new_state = State(values=tuple(new_values), target=state.target)

    if len(new_state.values) != len(state.values) - 1:
        raise CountdownError(
            "Internal invariant violated: transition must reduce bag size "
            f"by exactly 1, got {len(state.values)} -> {len(new_state.values)}"
        )

    return new_state


class CountdownEnv:
    """Thin stateful wrapper over the pure functions above, matching the
    project plan's requested API. Kept deliberately minimal: it holds
    `current_state` and delegates everything else. The oracle in
    oracle/evaluation.py calls the pure functions directly and never
    goes through this class -- this wrapper exists for rollout-style code
    (e.g. a future LLM policy loop), not for oracle computation.
    """

    def __init__(self, rules: RuleSet = DEFAULT_RULES) -> None:
        self.rules = rules
        self._state: State | None = None

    def reset(self, problem: Problem) -> State:
        self._state = problem.to_state()
        return self._state

    @property
    def current_state(self) -> State:
        if self._state is None:
            raise CountdownError("reset() must be called before use")
        return self._state

    def allowed_actions(self) -> tuple[Action, ...]:
        return allowed_actions(self.current_state, self.rules)

    def is_terminal(self) -> bool:
        return is_terminal(self.current_state, self.rules)

    def reward(self) -> int:
        return reward(self.current_state)

    def step(self, action: Action) -> StepResult:
        next_state = transition(self.current_state, action, self.rules)
        self._state = next_state
        return StepResult(
            state=next_state,
            reward=reward(next_state),
            terminal=is_terminal(next_state, self.rules),
        )
