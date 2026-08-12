"""Credit-structure summaries built on top of the exact oracle.

Two structurally SEPARATE profile types, on purpose (see docs/project_plan.md
and the Phase 1 plan Part 2.5 -- this separation is the enforcement of the
project's single most important conceptual rule):

  - OptimalActionProfile: policy-INDEPENDENT, optimal-play facts about a
    state. "Was this move already a game-loser under perfect play?"
  - PolicyCreditProfile: policy-DEPENDENT advantage facts about a state,
    computed for one specific policy. "Was this move worse than what THIS
    policy usually does here?"

A move can score badly on one and fine on the other. Never merge these two
types, and never name a PolicyCreditProfile field as if it were measuring
optimal-play destructiveness.

ProblemProfile aggregates both, plus DAG-level structural statistics, into
one small summary per problem -- meant for reporting across hundreds of
problems (see experiments/profile_environment.py), not for re-deriving
individual Q*/Api values (those live in the full DAG/oracle artifact, see
analysis/artifacts.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

from env.countdown import (
    Action,
    DEFAULT_RULES,
    RuleSet,
    State,
    allowed_actions,
    is_success,
    is_terminal,
)
from oracle.evaluation import (
    ExactPolicyEvaluator,
    ReasoningDag,
    VisitProbabilities,
    is_pivotal,
)


# ---------------------------------------------------------------------------
# OptimalActionProfile: policy-independent, optimal-play facts about a state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OptimalActionProfile:
    """What does perfect play say about the choices available at this
    state? Computed once per state, and the same for every policy.
    """

    action_to_q_star: Mapping[Action, int]
    allowed_action_count: int
    solution_preserving_count: int
    ruining_actions: tuple[Action, ...]
    fraction_of_moves_that_ruin_solution: float | None
    min_steps_to_success: int | None


def has_at_least_one_ruining_move(
    state: State,
    v_star: Mapping[State, int],
    q_star_for_state: Mapping[Action, int],
) -> bool:
    """Is this a state where the choice genuinely matters? True iff the
    state is still winnable under optimal play (v_star[state] == 1) and at
    least one available action would irreversibly ruin that possibility.
    A policy-INDEPENDENT, optimal-play question -- never conflate this with
    whether a specific policy's advantage looks bad here.
    """
    if v_star[state] != 1:
        return False
    return any(
        is_pivotal(state, action, v_star, q_star_for_state)
        for action in q_star_for_state
    )


def compute_optimal_action_profile(
    state: State,
    v_star: Mapping[State, int],
    q_star: Mapping[State, Mapping[Action, int]],
    min_steps_to_success: Mapping[State, int | None],
) -> OptimalActionProfile:
    """Build the optimal-play profile for one state from already-computed
    oracle mappings (does not recompute V*/Q*/min_steps -- callers build
    those once per DAG and reuse them across every state).
    """
    q_star_for_state = q_star.get(state, {})
    allowed_count = len(q_star_for_state)
    solution_preserving_count = sum(1 for q in q_star_for_state.values() if q == 1)
    ruining_actions = tuple(
        sorted(
            (
                action
                for action in q_star_for_state
                if is_pivotal(state, action, v_star, q_star_for_state)
            ),
            key=Action.sort_key,
        )
    )

    # fraction_of_moves_that_ruin_solution is UNDEFINED (None), not 0.0, for
    # terminal states (no actions to take) and for already-unsolvable
    # non-terminal states (every action there is moot, not "safe").
    if allowed_count == 0 or v_star[state] == 0:
        fraction_ruining: float | None = None
    else:
        fraction_ruining = 1.0 - (solution_preserving_count / allowed_count)

    return OptimalActionProfile(
        action_to_q_star=dict(q_star_for_state),
        allowed_action_count=allowed_count,
        solution_preserving_count=solution_preserving_count,
        ruining_actions=ruining_actions,
        fraction_of_moves_that_ruin_solution=fraction_ruining,
        min_steps_to_success=min_steps_to_success[state],
    )


# ---------------------------------------------------------------------------
# PolicyCreditProfile: policy-dependent advantage facts about a state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyCreditProfile:
    """How does ONE specific policy's advantage look at this state? Kept as
    a separate type from OptimalActionProfile on purpose -- see the module
    docstring. Never add a field here named as if it measures optimal-play
    destructiveness (e.g. nothing resembling "ruining" or "destructive").
    """

    action_to_advantage: Mapping[Action, Fraction]
    action_probs: Mapping[Action, Fraction]
    negative_count: int
    zero_count: int
    positive_count: int
    mean_abs_advantage: Fraction
    max_abs_advantage: Fraction


_EMPTY_POLICY_CREDIT_PROFILE = PolicyCreditProfile(
    action_to_advantage={},
    action_probs={},
    negative_count=0,
    zero_count=0,
    positive_count=0,
    mean_abs_advantage=Fraction(0),
    max_abs_advantage=Fraction(0),
)


def compute_policy_credit_profile(
    state: State,
    evaluator: ExactPolicyEvaluator,
    rules: RuleSet = DEFAULT_RULES,
) -> PolicyCreditProfile:
    """Build one policy's credit profile for `state`. Terminal states have
    no actions and no advantages -- return the empty profile directly
    rather than asking the policy for probabilities over zero actions
    (which some policies, e.g. UniformPolicy, cannot answer: 1/0 is
    undefined).
    """
    if is_terminal(state, rules):
        return _EMPTY_POLICY_CREDIT_PROFILE

    advantages = evaluator.advantages(state)
    probs = evaluator.action_probs(state)

    negative_count = sum(1 for a in advantages.values() if a < 0)
    zero_count = sum(1 for a in advantages.values() if a == 0)
    positive_count = sum(1 for a in advantages.values() if a > 0)

    abs_advantages = [abs(a) for a in advantages.values()]
    mean_abs_advantage = sum(abs_advantages, Fraction(0)) / len(abs_advantages)
    max_abs_advantage = max(abs_advantages)

    return PolicyCreditProfile(
        action_to_advantage=dict(advantages),
        action_probs=dict(probs),
        negative_count=negative_count,
        zero_count=zero_count,
        positive_count=positive_count,
        mean_abs_advantage=mean_abs_advantage,
        max_abs_advantage=max_abs_advantage,
    )


# ---------------------------------------------------------------------------
# ProblemProfile: one small summary per problem, for reporting across many
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BranchingStats:
    mean: float
    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class PolicyAdvantageSummary:
    """Aggregate Api statistics for one policy across every non-terminal
    reachable state-action pair in a problem's DAG. Unweighted -- treats a
    state the policy would visit 0.01% of the time the same as one it
    visits 60% of the time. See OccupancyWeightedAdvantageSummary for the
    companion view that corrects for this.
    """

    negative_count: int
    zero_count: int
    positive_count: int
    mean_abs_advantage: float


@dataclass(frozen=True, slots=True)
class OccupancyWeightedAdvantageSummary:
    """Api sign/magnitude distribution weighted by how often the policy
    actually reaches each state and takes each action, rather than
    treating every reachable state-action pair as equally important. A
    credit structure that looks rich when every corner of the DAG is
    enumerated may be nearly flat in the region the policy actually
    visits -- and it's the visited region that determines what RL will
    experience. negative_mass/zero_mass/positive_mass are a proper
    distribution (sum to 1) over the visited state-action pairs.
    """

    negative_mass: float
    zero_mass: float
    positive_mass: float
    mean_abs_advantage: float


def compute_occupancy_weighted_advantage_summary(
    non_terminal_states: list[State],
    evaluator: ExactPolicyEvaluator,
    visit_probabilities: VisitProbabilities,
) -> OccupancyWeightedAdvantageSummary:
    """Weight each state-action pair's advantage by
    probability_of_taking_action, then normalize into a proper
    distribution over {negative, zero, positive}.
    """
    negative_mass = zero_mass = positive_mass = 0.0
    weighted_abs_sum = 0.0

    for state in non_terminal_states:
        state_mass = visit_probabilities.probability_of_reaching_state.get(
            state, Fraction(0)
        )
        if state_mass == 0:
            continue
        for action, advantage in evaluator.advantages(state).items():
            action_mass = float(
                visit_probabilities.probability_of_taking_action.get(
                    (state, action), Fraction(0)
                )
            )
            if action_mass == 0:
                continue
            if advantage < 0:
                negative_mass += action_mass
            elif advantage == 0:
                zero_mass += action_mass
            else:
                positive_mass += action_mass
            weighted_abs_sum += action_mass * float(abs(advantage))

    total_mass = negative_mass + zero_mass + positive_mass
    if total_mass == 0:
        return OccupancyWeightedAdvantageSummary(0.0, 0.0, 0.0, 0.0)
    return OccupancyWeightedAdvantageSummary(
        negative_mass=negative_mass / total_mass,
        zero_mass=zero_mass / total_mass,
        positive_mass=positive_mass / total_mass,
        mean_abs_advantage=weighted_abs_sum / total_mass,
    )


_FRAGILITY_BUCKETS = (
    ("0.0-0.2", 0.0, 0.2),
    ("0.2-0.4", 0.2, 0.4),
    ("0.4-0.6", 0.4, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.0 + 1e-9),
)


def _fragility_histogram(values: list[float]) -> dict[str, int]:
    counts = {label: 0 for label, _, _ in _FRAGILITY_BUCKETS}
    for value in values:
        for label, low, high in _FRAGILITY_BUCKETS:
            if low <= value < high:
                counts[label] += 1
                break
    return counts


@dataclass(frozen=True, slots=True)
class ProblemProfile:
    solvable: bool
    initially_solved: bool
    node_count: int
    edge_count: int
    branching: BranchingStats | None
    min_solution_depth: int | None
    solution_preserving_root_action_count: int
    ruining_move_state_count: int
    ruining_move_state_rate: float | None
    ruining_action_count: int
    ruining_action_rate: float | None
    fraction_of_moves_that_ruin_solution_mean: float | None
    fraction_of_moves_that_ruin_solution_histogram: Mapping[str, int]
    policy_advantage_summaries: Mapping[str, PolicyAdvantageSummary]
    occupancy_weighted_advantage_summaries: Mapping[str, OccupancyWeightedAdvantageSummary]


def compute_problem_profile(
    dag: ReasoningDag,
    v_star: Mapping[State, int],
    q_star: Mapping[State, Mapping[Action, int]],
    min_steps_to_success: Mapping[State, int | None],
    evaluators: Mapping[str, ExactPolicyEvaluator],
    rules: RuleSet = DEFAULT_RULES,
    visit_probabilities: Mapping[str, VisitProbabilities] | None = None,
) -> ProblemProfile:
    root = dag.root
    non_terminal_states = [s for s in dag.nodes if not is_terminal(s, rules)]
    solvable_non_terminal_states = [s for s in non_terminal_states if v_star[s] == 1]

    if non_terminal_states:
        branching_counts = [len(allowed_actions(s, rules)) for s in non_terminal_states]
        branching = BranchingStats(
            mean=sum(branching_counts) / len(branching_counts),
            minimum=min(branching_counts),
            maximum=max(branching_counts),
        )
    else:
        branching = None

    root_q_star = q_star.get(root, {})
    solution_preserving_root_action_count = sum(
        1 for q in root_q_star.values() if q == 1
    )

    ruining_move_state_count = sum(
        1
        for s in solvable_non_terminal_states
        if has_at_least_one_ruining_move(s, v_star, q_star[s])
    )
    ruining_move_state_rate = (
        ruining_move_state_count / len(solvable_non_terminal_states)
        if solvable_non_terminal_states
        else None
    )

    ruining_action_count = 0
    total_actions_from_solvable_states = 0
    fragility_values: list[float] = []
    for s in solvable_non_terminal_states:
        q_star_for_state = q_star[s]
        total_actions_from_solvable_states += len(q_star_for_state)
        solution_preserving = sum(1 for q in q_star_for_state.values() if q == 1)
        fragility_values.append(1.0 - solution_preserving / len(q_star_for_state))
        ruining_action_count += sum(
            1
            for action in q_star_for_state
            if is_pivotal(s, action, v_star, q_star_for_state)
        )
    ruining_action_rate = (
        ruining_action_count / total_actions_from_solvable_states
        if total_actions_from_solvable_states
        else None
    )
    fraction_of_moves_that_ruin_solution_mean = (
        sum(fragility_values) / len(fragility_values) if fragility_values else None
    )

    policy_advantage_summaries: dict[str, PolicyAdvantageSummary] = {}
    for name, evaluator in evaluators.items():
        negative_count = zero_count = positive_count = 0
        abs_advantages: list[Fraction] = []
        for s in non_terminal_states:
            for advantage in evaluator.advantages(s).values():
                if advantage < 0:
                    negative_count += 1
                elif advantage == 0:
                    zero_count += 1
                else:
                    positive_count += 1
                abs_advantages.append(abs(advantage))
        mean_abs_advantage = (
            float(sum(abs_advantages, Fraction(0)) / len(abs_advantages))
            if abs_advantages
            else 0.0
        )
        policy_advantage_summaries[name] = PolicyAdvantageSummary(
            negative_count=negative_count,
            zero_count=zero_count,
            positive_count=positive_count,
            mean_abs_advantage=mean_abs_advantage,
        )

    occupancy_weighted_advantage_summaries: dict[str, OccupancyWeightedAdvantageSummary] = {}
    if visit_probabilities:
        for name, evaluator in evaluators.items():
            visit_probs_for_policy = visit_probabilities.get(name)
            if visit_probs_for_policy is None:
                continue
            occupancy_weighted_advantage_summaries[name] = (
                compute_occupancy_weighted_advantage_summary(
                    non_terminal_states, evaluator, visit_probs_for_policy
                )
            )

    return ProblemProfile(
        solvable=bool(v_star[root]),
        initially_solved=is_success(root),
        node_count=dag.node_count,
        edge_count=dag.edge_count,
        branching=branching,
        min_solution_depth=min_steps_to_success[root],
        solution_preserving_root_action_count=solution_preserving_root_action_count,
        ruining_move_state_count=ruining_move_state_count,
        ruining_move_state_rate=ruining_move_state_rate,
        ruining_action_count=ruining_action_count,
        ruining_action_rate=ruining_action_rate,
        fraction_of_moves_that_ruin_solution_mean=fraction_of_moves_that_ruin_solution_mean,
        fraction_of_moves_that_ruin_solution_histogram=_fragility_histogram(
            fragility_values
        ),
        policy_advantage_summaries=policy_advantage_summaries,
        occupancy_weighted_advantage_summaries=occupancy_weighted_advantage_summaries,
    )
