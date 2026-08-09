"""An independent, deliberately naive brute-force solver used ONLY to
cross-check oracle/evaluation.py's memoized computations.

This is NOT a test file itself (no test_ functions) -- it is imported by
test_oracle.py. It shares the environment's pure functions (is_success,
allowed_actions, transition), since those define what the game IS. It shares
NOTHING with oracle/evaluation.py's DAG-building or memoization code: no
shared cache, no shared recursion helpers. If oracle/evaluation.py has a
canonicalization or memo-key bug, this independent path is the thing that
catches it -- which is why it is the highest-value test in the project.

Only use this on small state spaces (2-4 starting numbers, small values):
it recomputes every subtree from scratch with no memoization at all, so it
is deliberately exponential.
"""

from fractions import Fraction

from env.countdown import DEFAULT_RULES, RuleSet, State, is_success, allowed_actions, transition


def brute_force_v_star(state: State, rules: RuleSet = DEFAULT_RULES) -> int:
    if is_success(state):
        return 1
    actions = allowed_actions(state, rules)
    if not actions:
        return 0
    return max(brute_force_v_star(transition(state, a, rules), rules) for a in actions)


def brute_force_v_pi(state: State, policy, rules: RuleSet = DEFAULT_RULES) -> Fraction:
    if is_success(state):
        return Fraction(1)
    actions = allowed_actions(state, rules)
    if not actions:
        return Fraction(0)
    probs = policy.action_probs(state, actions)
    total = Fraction(0)
    for a in actions:
        total += probs[a] * brute_force_v_pi(transition(state, a, rules), policy, rules)
    return total


def brute_force_min_steps_to_success(
    state: State, rules: RuleSet = DEFAULT_RULES
) -> int | None:
    if is_success(state):
        return 0
    actions = allowed_actions(state, rules)
    if not actions:
        return None
    child_steps = [
        brute_force_min_steps_to_success(transition(state, a, rules), rules)
        for a in actions
    ]
    solvable_child_steps = [c for c in child_steps if c is not None]
    if not solvable_child_steps:
        return None
    return 1 + min(solvable_child_steps)


def enumerate_successful_paths(
    state: State, rules: RuleSet = DEFAULT_RULES, prefix: tuple = ()
) -> list[tuple]:
    """All action sequences (as tuples of Action) from `state` that reach
    success. Used to independently confirm the shortest solution length.
    """
    if is_success(state):
        return [prefix]
    actions = allowed_actions(state, rules)
    paths: list[tuple] = []
    for a in actions:
        child = transition(state, a, rules)
        paths.extend(enumerate_successful_paths(child, rules, prefix + (a,)))
    return paths
