"""Run the Phase 3 exact-credit estimator-fidelity pilot.

Key Terms:
- Pilot problem: one four-number validation problem selected for the initial Phase 3 run.
- Canonical hash: the frozen dataset identifier used to order pilot selection reproducibly.

Selection lives beside the Phase 3 benchmark command so its dataset contract
does not leak into the Phase 2 policy-bridge experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from analysis.conclusion_robustness_metrics import (
    METHOD_NAMES,
    PairSamples,
    bootstrap_method_fidelity,
    method_fidelity,
)
from analysis.credit_estimator_fidelity import (
    PHASE3_ESTIMATOR_ORDER,
    ROLLOUT_COST_BY_ESTIMATOR,
    ProblemClusteredFidelity,
    SelectedCreditState,
    StateActionCredit,
    bootstrap_problem_clustered_fidelity,
    credit_states_at_depths,
    nondegenerate_credit_states,
    pivotal_action_detection,
    select_widest_credit_states,
    stable_estimator_order,
    stratum_balanced_rank_fidelity,
    within_state_rank_fidelity,
)
from env.countdown import (
    DEFAULT_RULES,
    Action,
    Operator,
    State,
    allowed_actions,
    is_terminal,
    reward,
    transition,
)
from oracle.evaluation import PolicyValidationError, build_reasoning_dag
from oracle.float_policy_evaluation import FloatPolicyEvaluator
from policies.allowed_action_llm_policy import (
    PROMPT_VERSION,
    serialize_action,
    serialize_rules,
    serialize_state,
)

PHASE2_MENU_ORDERING_IDS = (
    "phase2-v1",
    "phase2-sensitivity-1",
    "phase2-sensitivity-2",
    "phase2-sensitivity-3",
    "phase2-sensitivity-4",
)
PILOT_ARTIFACT_NAME = "phase3_credit_estimator_pilot.json"
PILOT_COMPACT_SUMMARY_NAME = "phase3_credit_estimator_pilot_summary.json"
PROBLEM_ARTIFACT_DIRECTORY_NAME = "problems"
PLOT_ARTIFACT_DIRECTORY_NAME = "plots"
STRATIFIED_ARTIFACT_DIRECTORY_NAME = "stratified"
STRATIFIED_RUN_MANIFEST_NAME = "run_manifest.json"
STRATIFIED_SUMMARY_NAME = "phase3_credit_estimator_stratified.json"
STRATIFIED_COMPACT_SUMMARY_NAME = "phase3_credit_estimator_stratified_summary.json"
PILOT_SUMMARY_NAME = "phase3_credit_estimator_pilot_summary.md"
MINIMUM_MEDIAN_FIDELITY_IMPROVEMENT = 0.10
MINIMUM_BOOTSTRAP_SUPPORT = 0.80
# Benchmark B caches written before the identity split recorded this key.
LEGACY_BENCHMARK_A_CACHE_KEY = "pairs_per_problem"


@dataclass(frozen=True, slots=True)
class Phase3PilotProblem:
    """One deterministically selected four-number Phase 3 validation problem."""

    canonical_hash: str
    source_row_index: int
    state: State


@dataclass(frozen=True, slots=True)
class Phase3CreditEstimatorPair:
    """One on-policy decision with exact credit and reset rollout outcomes."""

    state: State
    action: Action
    exact_advantage: float
    parent_reset_outcomes: tuple[int, ...]
    child_reset_outcomes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Phase3PilotGate:
    """The prespecified decision evidence for expanding the Phase 3 pilot."""

    decision: str
    median_fidelity_improvement: float
    bootstrap_support: float


@dataclass(frozen=True, slots=True)
class Phase3StratifiedGate:
    """The prespecified capability decision for Benchmark B."""

    decision: str
    median_fidelity_improvement: float
    bootstrap_support: float


class _MenuAveragedActionPolicy:
    """A fixed action distribution for every state in one complete DAG."""

    def __init__(self, action_probabilities_by_state: Mapping[State, Mapping[Action, float]]):
        self._action_probabilities_by_state = action_probabilities_by_state

    def action_probs(
        self,
        state: State,
        actions: tuple[Action, ...],
    ) -> Mapping[Action, float]:
        action_probabilities = self._action_probabilities_by_state.get(state)
        if action_probabilities is None or set(action_probabilities) != set(actions):
            raise PolicyValidationError(
                f"Missing complete averaged action probabilities for {state!r}"
            )
        return action_probabilities


def select_phase3_validation_problems(
    validation_dataset_path: Path,
    *,
    count: int,
) -> tuple[Phase3PilotProblem, ...]:
    """Select a canonical-hash-ordered four-number validation subset."""
    if count < 1:
        raise ValueError("count must be at least one")

    eligible_problems: list[Phase3PilotProblem] = []
    for source_row_index, line in enumerate(validation_dataset_path.read_text().splitlines()):
        if not line:
            continue
        dataset_row = json.loads(line)
        if dataset_row["n_values"] != 4:
            continue
        eligible_problems.append(
            Phase3PilotProblem(
                canonical_hash=str(dataset_row["canonical_hash"]),
                source_row_index=source_row_index,
                state=State(
                    values=tuple(dataset_row["values"]),
                    target=int(dataset_row["target"]),
                ),
            )
        )

    selected_problems = tuple(
        sorted(eligible_problems, key=lambda problem: problem.canonical_hash)[:count]
    )
    if len(selected_problems) != count:
        raise ValueError(
            f"Requested {count} four-number problems but found {len(selected_problems)}"
        )
    return selected_problems


def _sample_policy_outcome(
    starting_state: State,
    exact_evaluator: FloatPolicyEvaluator,
    random_generator: random.Random,
) -> int:
    """Roll out the evaluated policy from one state to its terminal reward."""
    current_state = starting_state
    while not is_terminal(current_state):
        available_actions = allowed_actions(current_state)
        action_probabilities = exact_evaluator.action_probs(current_state)
        selected_action = random_generator.choices(
            available_actions,
            weights=[action_probabilities[action] for action in available_actions],
            k=1,
        )[0]
        current_state = transition(current_state, selected_action)
    return reward(current_state)


def _sample_on_policy_state_action(
    root_state: State,
    exact_evaluator: FloatPolicyEvaluator,
    random_generator: random.Random,
) -> tuple[State, Action]:
    """Sample one decision uniformly from a trajectory drawn from the policy."""
    current_state = root_state
    trajectory_decisions: list[tuple[State, Action]] = []
    while not is_terminal(current_state):
        available_actions = allowed_actions(current_state)
        action_probabilities = exact_evaluator.action_probs(current_state)
        selected_action = random_generator.choices(
            available_actions,
            weights=[action_probabilities[action] for action in available_actions],
            k=1,
        )[0]
        trajectory_decisions.append((current_state, selected_action))
        current_state = transition(current_state, selected_action)
    return random_generator.choice(trajectory_decisions)


def sample_phase3_credit_estimator_pairs(
    root_state: State,
    exact_evaluator: FloatPolicyEvaluator,
    *,
    pair_count: int,
    estimator_pool_size: int,
    seed: int,
) -> tuple[Phase3CreditEstimatorPair, ...]:
    """Sample Phase 3 decisions with paired parent and child reset outcomes.

    Steps:
    1. Draw one state-action decision from an on-policy trajectory.
    2. Compute that decision's exact advantage from the evaluated DAG policy.
    3. Independently roll out the parent and selected child state into equal pools.
    """
    if pair_count < 1:
        raise ValueError("pair_count must be at least one")
    if estimator_pool_size < 1:
        raise ValueError("estimator_pool_size must be at least one")

    random_generator = random.Random(seed)
    sampled_pairs: list[Phase3CreditEstimatorPair] = []
    for _ in range(pair_count):
        # Step 1: draw an observed state-action decision.
        state, action = _sample_on_policy_state_action(
            root_state, exact_evaluator, random_generator
        )
        child_state = transition(state, action)

        # Step 2: retain the exact credit the sampled estimators approximate.
        exact_advantage = exact_evaluator.a_pi(state, action)

        # Step 3: draw independent reset pools from the parent and child states.
        parent_reset_outcomes = tuple(
            _sample_policy_outcome(state, exact_evaluator, random_generator)
            for _ in range(estimator_pool_size)
        )
        child_reset_outcomes = tuple(
            _sample_policy_outcome(child_state, exact_evaluator, random_generator)
            for _ in range(estimator_pool_size)
        )
        sampled_pairs.append(
            Phase3CreditEstimatorPair(
                state=state,
                action=action,
                exact_advantage=exact_advantage,
                parent_reset_outcomes=parent_reset_outcomes,
                child_reset_outcomes=child_reset_outcomes,
            )
        )
    return tuple(sampled_pairs)


def compute_credit_estimator_fidelity(
    benchmark_pairs: Sequence[PairSamples],
) -> dict[str, float | None]:
    """Measure each predeclared estimator's rank fidelity against exact credit."""
    return {
        estimator_name: method_fidelity(benchmark_pairs, estimator_name)
        for estimator_name in METHOD_NAMES
    }


def phase3_estimator_display_name(estimator_key: str) -> str:
    """Render an internal estimator key with Phase 3 research terminology."""
    if estimator_key == "uniform":
        return "Uniform Outcome Credit"
    return (
        "Sampled Counterfactual Advantage "
        f"(K={estimator_key.removeprefix('mc_reset_k')})"
    )


def phase3_estimator_plot_label(estimator_key: str) -> str:
    """Render a short plain-language estimator label for charts."""
    if estimator_key == "uniform":
        return "One rollout"
    return f"Before/after ×{estimator_key.removeprefix('mc_reset_k')}"


def assess_phase3_pilot_gate(
    benchmark_pairs: Sequence[PairSamples],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> Phase3PilotGate:
    """Evaluate whether counterfactual credit materially and reliably beats Uniform."""
    bootstrap = bootstrap_method_fidelity(
        benchmark_pairs,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    uniform_draws = bootstrap.by_method["uniform"]
    counterfactual_draws = bootstrap.by_method["mc_reset_k16"]
    if not uniform_draws:
        raise ValueError("Bootstrap retained no comparable fidelity replicates")
    median_fidelity_improvement = float(
        median(counterfactual_draws) - median(uniform_draws)
    )
    bootstrap_support = sum(
        counterfactual_fidelity > uniform_fidelity
        for counterfactual_fidelity, uniform_fidelity in zip(
            counterfactual_draws, uniform_draws, strict=True
        )
    ) / len(uniform_draws)
    decision = (
        "CONTINUE"
        if (
            median_fidelity_improvement >= MINIMUM_MEDIAN_FIDELITY_IMPROVEMENT
            and bootstrap_support >= MINIMUM_BOOTSTRAP_SUPPORT
        )
        else "REWORK"
    )
    return Phase3PilotGate(
        decision=decision,
        median_fidelity_improvement=median_fidelity_improvement,
        bootstrap_support=bootstrap_support,
    )


def evaluate_stratified_gate(
    bootstrap: ProblemClusteredFidelity,
) -> Phase3StratifiedGate:
    """Decide whether Benchmark B demonstrates estimator capability."""
    uniform_draws = bootstrap.by_estimator.get("uniform", ())
    counterfactual_draws = bootstrap.by_estimator.get("mc_reset_k16", ())
    if not uniform_draws or len(uniform_draws) != len(counterfactual_draws):
        raise ValueError("Benchmark B needs paired Uniform and K=16 bootstrap draws")
    paired_fidelity_improvements = tuple(
        counterfactual_fidelity - uniform_fidelity
        for counterfactual_fidelity, uniform_fidelity in zip(
            counterfactual_draws, uniform_draws, strict=True
        )
    )
    median_fidelity_improvement = float(median(paired_fidelity_improvements))
    bootstrap_support = sum(
        counterfactual_fidelity > uniform_fidelity
        for counterfactual_fidelity, uniform_fidelity in zip(
            counterfactual_draws, uniform_draws, strict=True
        )
    ) / len(uniform_draws)
    return Phase3StratifiedGate(
        decision=(
            "CAPABLE"
            if (
                median_fidelity_improvement >= MINIMUM_MEDIAN_FIDELITY_IMPROVEMENT
                and bootstrap_support >= MINIMUM_BOOTSTRAP_SUPPORT
            )
            else "NOT-DEMONSTRATED"
        ),
        median_fidelity_improvement=median_fidelity_improvement,
        bootstrap_support=bootstrap_support,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Phase 3 pilot command-line interface without model imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("pilot", "stratified"), default="pilot")
    parser.add_argument("--problem-count", type=int, default=50)
    parser.add_argument("--pairs-per-problem", type=int, default=32)
    parser.add_argument("--estimator-pool-size", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_913)
    parser.add_argument("--seed", type=int, default=20_260_912)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the deterministic pilot selection; do not load a model.",
    )
    return parser


def _selected_problem_metadata(
    selected_problems: Sequence[Phase3PilotProblem],
) -> list[dict[str, object]]:
    return [
        {
            "canonical_hash": selected_problem.canonical_hash,
            "source_row_index": selected_problem.source_row_index,
            "root": serialize_state(selected_problem.state),
        }
        for selected_problem in selected_problems
    ]


def phase3_problem_artifact_path(
    output_directory: Path,
    canonical_hash: str,
) -> Path:
    """Return the stable checkpoint path for one selected pilot problem."""
    return (
        output_directory
        / PROBLEM_ARTIFACT_DIRECTORY_NAME
        / f"{canonical_hash}.json"
    )


def _checkpoint_configuration(
    args: argparse.Namespace,
    scorer_metadata: Mapping[str, str],
) -> dict[str, object]:
    """Build the exact configuration that makes a checkpoint reusable."""
    return {
        "model": args.model,
        "temperature": args.temperature,
        "menu_ordering_ids": list(PHASE2_MENU_ORDERING_IDS),
        "pairs_per_problem": args.pairs_per_problem,
        "estimator_pool_size": args.estimator_pool_size,
        "seed": args.seed,
        "model_revision": scorer_metadata["model_revision"],
        "tokenizer_revision": scorer_metadata["tokenizer_revision"],
    }


def _scorer_metadata(scorer: Any) -> dict[str, str]:
    """Extract the immutable scorer identity required for checkpoint reuse."""
    metadata = scorer.metadata
    scorer_metadata = {
        "model_repository": str(metadata.model_repository),
        "model_revision": str(metadata.model_revision),
        "tokenizer_repository": str(metadata.tokenizer_repository),
        "tokenizer_revision": str(metadata.tokenizer_revision),
    }
    if not all(scorer_metadata.values()):
        raise ValueError("Scorer metadata must contain immutable model and tokenizer revisions")
    return scorer_metadata


def _validate_completed_checkpoint(
    completed_problem_summary: Mapping[str, object],
    selected_problem: Phase3PilotProblem,
    args: argparse.Namespace,
) -> None:
    """Reject a cached problem whose identity differs from this pilot invocation."""
    checkpoint_configuration = completed_problem_summary.get("checkpoint_configuration")
    scorer_metadata = completed_problem_summary.get("scorer_metadata")
    if not isinstance(checkpoint_configuration, dict) or not isinstance(scorer_metadata, dict):
        # A malformed cache is invalid data, not a caller type error.
        raise ValueError(  # noqa: TRY004
            "Checkpoint is missing immutable scorer metadata"
        )
    required_metadata = {
        "model_repository",
        "model_revision",
        "tokenizer_repository",
        "tokenizer_revision",
    }
    if (
        set(scorer_metadata) != required_metadata
        or not all(isinstance(scorer_metadata[key], str) and scorer_metadata[key] for key in required_metadata)
        or checkpoint_configuration.get("model_revision") != scorer_metadata["model_revision"]
        or checkpoint_configuration.get("tokenizer_revision")
        != scorer_metadata["tokenizer_revision"]
    ):
        raise ValueError("Checkpoint immutable scorer metadata does not match its configuration")
    expected_configuration = {
        "model": args.model,
        "temperature": args.temperature,
        "menu_ordering_ids": list(PHASE2_MENU_ORDERING_IDS),
        "pairs_per_problem": args.pairs_per_problem,
        "estimator_pool_size": args.estimator_pool_size,
        "seed": args.seed,
    }
    if (
        completed_problem_summary.get("canonical_hash") != selected_problem.canonical_hash
        or completed_problem_summary.get("root") != serialize_state(selected_problem.state)
        or any(
            checkpoint_configuration.get(key) != value
            for key, value in expected_configuration.items()
        )
    ):
        raise ValueError("Checkpoint does not match the selected Phase 3 problem")


def _stratified_output_directory(output_directory: Path) -> Path:
    """Return the isolated artifact root for Benchmark B."""
    if output_directory.name == STRATIFIED_ARTIFACT_DIRECTORY_NAME:
        return output_directory
    return output_directory / STRATIFIED_ARTIFACT_DIRECTORY_NAME


def _stratified_checkpoint_configuration(
    args: argparse.Namespace,
    scorer_metadata: Mapping[str, str],
) -> dict[str, object]:
    """Build the immutable Benchmark B cache identity.

    `pairs_per_problem` is deliberately absent: it only controls Benchmark A's
    on-policy sampling and would needlessly invalidate Benchmark B caches.
    """
    return {
        "model": args.model,
        "temperature": args.temperature,
        "menu_ordering_ids": list(PHASE2_MENU_ORDERING_IDS),
        "estimator_pool_size": args.estimator_pool_size,
        "seed": args.seed,
        "model_revision": scorer_metadata["model_revision"],
        "tokenizer_revision": scorer_metadata["tokenizer_revision"],
        "prompt_version": PROMPT_VERSION,
        "rules": serialize_rules(DEFAULT_RULES),
        "selection_depths": [0, 1, 2],
    }


def _migrated_stratified_configuration(configuration: object) -> object:
    """Drop the Benchmark-A-only key that older Benchmark B caches recorded."""
    if not isinstance(configuration, dict) or LEGACY_BENCHMARK_A_CACHE_KEY not in configuration:
        return configuration
    return {
        key: value
        for key, value in configuration.items()
        if key != LEGACY_BENCHMARK_A_CACHE_KEY
    }


def _stratified_run_manifest_path(output_directory: Path) -> Path:
    """Return the stable scorer-identity manifest for Benchmark B."""
    return output_directory / STRATIFIED_RUN_MANIFEST_NAME


def _stratified_run_manifest(
    args: argparse.Namespace,
    scorer_metadata: Mapping[str, str],
) -> dict[str, object]:
    """Build the independent cache identity shared by all problem checkpoints."""
    return {
        "artifact_version": "1.0",
        "scorer_metadata": dict(scorer_metadata),
        "checkpoint_configuration": _stratified_checkpoint_configuration(
            args, scorer_metadata
        ),
    }


def _validate_stratified_run_manifest(
    manifest: object,
    args: argparse.Namespace,
    scorer_metadata: Mapping[str, str] | None,
) -> dict[str, object]:
    """Load a complete manifest without trusting problem checkpoint metadata."""
    if not isinstance(manifest, dict) or set(manifest) != {
        "artifact_version",
        "scorer_metadata",
        "checkpoint_configuration",
    }:
        raise ValueError("Benchmark B run manifest has unexpected fields")
    manifest = {
        **manifest,
        "checkpoint_configuration": _migrated_stratified_configuration(
            manifest["checkpoint_configuration"]
        ),
    }
    metadata = manifest["scorer_metadata"]
    configuration = manifest["checkpoint_configuration"]
    required_metadata = {
        "model_repository",
        "model_revision",
        "tokenizer_repository",
        "tokenizer_revision",
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != required_metadata
        or not all(isinstance(metadata[key], str) and metadata[key] for key in required_metadata)
        or not isinstance(configuration, dict)
        or configuration.get("model_revision") != metadata["model_revision"]
        or configuration.get("tokenizer_revision") != metadata["tokenizer_revision"]
    ):
        raise ValueError("Benchmark B run manifest has incomplete scorer identity")
    expected_static_configuration = {
        "model": args.model,
        "temperature": args.temperature,
        "menu_ordering_ids": list(PHASE2_MENU_ORDERING_IDS),
        "estimator_pool_size": args.estimator_pool_size,
        "seed": args.seed,
        "prompt_version": PROMPT_VERSION,
        "rules": serialize_rules(DEFAULT_RULES),
        "selection_depths": [0, 1, 2],
    }
    if any(
        configuration.get(key) != value
        for key, value in expected_static_configuration.items()
    ) or set(configuration) != {
        *expected_static_configuration,
        "model_revision",
        "tokenizer_revision",
    }:
        raise ValueError("Benchmark B run manifest does not match this invocation")
    if scorer_metadata is not None and manifest != _stratified_run_manifest(
        args, scorer_metadata
    ):
        raise ValueError("Benchmark B run manifest does not match the loaded scorer")
    return manifest


def stable_pool_seed(
    base_seed: int,
    canonical_hash: str,
    depth: int,
    state: State,
    action: Action | None,
    pool_identity: str,
) -> int:
    """Derive a reproducible seed for one complete rollout pool.

    Each pool gets exactly one seed and therefore one RNG stream; adjacent
    per-draw seeds would correlate pools that are meant to be independent.
    `action` is None for the state-level pool shared across all actions.
    """
    identity = {
        "base_seed": base_seed,
        "canonical_hash": canonical_hash,
        "depth": depth,
        "state": serialize_state(state),
        "action": None if action is None else serialize_action(action),
        "pool_identity": pool_identity,
    }
    digest = hashlib.blake2b(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(),
        digest_size=16,
    ).digest()
    return int.from_bytes(digest, "big")


def compute_policy_visit_probabilities(
    reasoning_dag: Any,
    evaluator: FloatPolicyEvaluator,
) -> dict[State, float]:
    """Propagate floating-point policy mass through a complete reasoning DAG."""
    occupancy: dict[State, float] = {reasoning_dag.root: 1.0}
    for state in reasoning_dag.topological_order():
        reaching_probability = occupancy.get(state, 0.0)
        if reaching_probability == 0.0 or is_terminal(state):
            continue
        for edge in reasoning_dag.out_edges(state):
            occupancy[edge.target_state] = (
                occupancy.get(edge.target_state, 0.0)
                + reaching_probability * evaluator.action_probs(state)[edge.action]
            )
    return occupancy


def build_state_candidates(
    reasoning_dag: Any,
    evaluator: FloatPolicyEvaluator,
    occupancy: Mapping[State, float],
    *,
    canonical_hash: str = "",
) -> tuple[SelectedCreditState, ...]:
    """Build complete exact-credit candidates before stratified selection.

    Candidates carry empty rollout pools because selection must depend only on
    exact advantages and occupancy; pools are sampled after a state is chosen.
    """
    return tuple(
        SelectedCreditState(
            problem_key=canonical_hash,
            state_key=json.dumps(
                serialize_state(state), sort_keys=True, separators=(",", ":")
            ),
            depth=reasoning_dag.nodes[state].depth,
            occupancy=occupancy.get(state, 0.0),
            action_credits=tuple(
                StateActionCredit(
                    action_key=json.dumps(
                        serialize_action(action), sort_keys=True, separators=(",", ":")
                    ),
                    exact_advantage=evaluator.a_pi(state, action),
                    parent_outcomes=(),
                    child_outcomes=(),
                )
                for action in sorted(allowed_actions(state), key=Action.sort_key)
            ),
        )
        for state in reasoning_dag.topological_order()
        if not is_terminal(state)
    )


def _serialize_menu_averaged_policy(
    action_probabilities_by_state: Mapping[State, Mapping[Action, float]],
) -> list[dict[str, object]]:
    """Serialize each nonterminal state and semantic action exactly once."""
    return [
        {
            "state": serialize_state(state),
            "actions": [
                {
                    "action": serialize_action(action),
                    "probability": action_probabilities_by_state[state][action],
                }
                for action in sorted(action_probabilities_by_state[state], key=Action.sort_key)
            ],
        }
        for state in sorted(
            action_probabilities_by_state,
            key=lambda item: (len(item.values), item.values, item.target),
        )
    ]


def _deserialize_menu_averaged_policy(
    records: object,
    reasoning_dag: Any,
) -> dict[State, dict[Action, float]]:
    """Load and validate the complete persisted menu-averaged policy table."""
    if not isinstance(records, list):
        # A malformed cache is invalid data, not a caller type error.
        raise ValueError(  # noqa: TRY004
            "Checkpoint menu_averaged_policy must be a JSON array"
        )
    probabilities_by_state: dict[State, dict[Action, float]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"state", "actions"}:
            raise ValueError("Checkpoint menu_averaged_policy record is malformed")
        state_value = record["state"]
        if not isinstance(state_value, dict) or set(state_value) != {"target", "values"}:
            raise ValueError("Checkpoint policy state is malformed")
        state = State(values=tuple(state_value["values"]), target=int(state_value["target"]))
        raw_actions = record["actions"]
        if not isinstance(raw_actions, list) or state in probabilities_by_state:
            raise ValueError("Checkpoint policy states must occur exactly once")
        action_probabilities: dict[Action, float] = {}
        for raw_action in raw_actions:
            try:
                serialized_action = raw_action["action"]
                action = Action(
                    operator=Operator(serialized_action["operator"]),
                    left=int(serialized_action["left"]),
                    right=int(serialized_action["right"]),
                )
                probability = float(raw_action["probability"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Checkpoint policy action is malformed") from error
            if action.result != serialized_action.get("result") or action in action_probabilities:
                raise ValueError("Checkpoint policy actions must occur exactly once")
            if not math.isfinite(probability) or probability < 0.0:
                raise ValueError("Checkpoint policy probability is invalid")
            action_probabilities[action] = probability
        probabilities_by_state[state] = action_probabilities

    expected_states = {
        state for state in reasoning_dag.topological_order() if not is_terminal(state)
    }
    if set(probabilities_by_state) != expected_states or any(
        set(probabilities_by_state[state]) != set(allowed_actions(state))
        for state in expected_states
    ):
        raise ValueError("Checkpoint policy does not cover every legal action exactly once")
    return probabilities_by_state


def _validate_stratified_checkpoint(
    checkpoint: Mapping[str, object],
    selected_problem: Phase3PilotProblem,
    manifest: Mapping[str, object],
    reasoning_dag: Any,
) -> None:
    """Reject any Benchmark B cache identity or policy-coverage drift."""
    configuration = _migrated_stratified_configuration(
        checkpoint.get("checkpoint_configuration")
    )
    metadata = checkpoint.get("scorer_metadata")
    if (
        configuration != manifest["checkpoint_configuration"]
        or metadata != manifest["scorer_metadata"]
    ):
        raise ValueError("Checkpoint does not match the Benchmark B run manifest")
    if (
        checkpoint.get("canonical_hash") != selected_problem.canonical_hash
        or checkpoint.get("root") != serialize_state(selected_problem.state)
        or checkpoint.get("node_count") != reasoning_dag.node_count
        or not isinstance(checkpoint.get("scoring_duration_seconds"), float)
        or not math.isfinite(checkpoint["scoring_duration_seconds"])
        or checkpoint["scoring_duration_seconds"] < 0.0
    ):
        raise ValueError("Checkpoint does not match the selected Phase 3 problem")


@dataclass(frozen=True, slots=True)
class StratifiedStateSample:
    """One selected state's primary credit state plus its shared parent pool."""

    credit_state: SelectedCreditState
    # A single parent pool for the whole state, used only by the post-hoc
    # shared-baseline sensitivity analysis.
    shared_parent_outcomes: tuple[float, ...]


def _rollout_pool(
    starting_state: State,
    evaluator: FloatPolicyEvaluator,
    *,
    pool_size: int,
    seed: int,
) -> tuple[float, ...]:
    """Draw one complete rollout pool from a single deterministic RNG stream."""
    random_generator = random.Random(seed)
    return tuple(
        float(_sample_policy_outcome(starting_state, evaluator, random_generator))
        for _ in range(pool_size)
    )


def sample_stratified_state_samples(
    selected_problem: Phase3PilotProblem,
    args: argparse.Namespace,
    reasoning_dag: Any,
    evaluator: FloatPolicyEvaluator,
) -> tuple[StratifiedStateSample, ...]:
    """Select wide states and sample their raw parent and child rollout pools.

    Steps:
    1. Build every nonterminal candidate from exact credit and occupancy alone.
    2. Select the widest-spread state at each requested depth.
    3. Draw one parent and one child pool per legal action, plus one parent
       pool shared by the whole state for the sensitivity analysis.
    """
    # Step 1: build selection candidates without touching rollout outcomes.
    occupancy = compute_policy_visit_probabilities(reasoning_dag, evaluator)
    candidates = build_state_candidates(
        reasoning_dag,
        evaluator,
        occupancy,
        canonical_hash=selected_problem.canonical_hash,
    )
    states_by_key = {
        candidate.state_key: state
        for state, candidate in zip(
            (
                state
                for state in reasoning_dag.topological_order()
                if not is_terminal(state)
            ),
            candidates,
            strict=True,
        )
    }

    # Step 2: select one widest-spread state per requested depth.
    selected = select_widest_credit_states(candidates)

    # Step 3: sample the raw pools every estimator is later derived from.
    samples: list[StratifiedStateSample] = []
    for selected_state in selected:
        state = states_by_key[selected_state.state_key]
        actions = sorted(allowed_actions(state), key=Action.sort_key)
        action_credits = tuple(
            StateActionCredit(
                action_key=credit.action_key,
                exact_advantage=credit.exact_advantage,
                parent_outcomes=_rollout_pool(
                    state,
                    evaluator,
                    pool_size=args.estimator_pool_size,
                    seed=stable_pool_seed(
                        args.seed,
                        selected_problem.canonical_hash,
                        selected_state.depth,
                        state,
                        action,
                        "parent",
                    ),
                ),
                child_outcomes=_rollout_pool(
                    transition(state, action),
                    evaluator,
                    pool_size=args.estimator_pool_size,
                    seed=stable_pool_seed(
                        args.seed,
                        selected_problem.canonical_hash,
                        selected_state.depth,
                        state,
                        action,
                        "child",
                    ),
                ),
            )
            for action, credit in zip(
                actions, selected_state.action_credits, strict=True
            )
        )
        samples.append(
            StratifiedStateSample(
                credit_state=SelectedCreditState(
                    problem_key=selected_state.problem_key,
                    state_key=selected_state.state_key,
                    depth=selected_state.depth,
                    occupancy=selected_state.occupancy,
                    action_credits=action_credits,
                ),
                shared_parent_outcomes=_rollout_pool(
                    state,
                    evaluator,
                    pool_size=args.estimator_pool_size,
                    seed=stable_pool_seed(
                        args.seed,
                        selected_problem.canonical_hash,
                        selected_state.depth,
                        state,
                        None,
                        "shared_parent",
                    ),
                ),
            )
        )
    return tuple(samples)


def with_shared_parent_baseline(
    samples: Sequence[StratifiedStateSample],
) -> tuple[SelectedCreditState, ...]:
    """Rebuild credit states so every action at a state shares one baseline.

    The primary design draws an independent parent pool per action, which adds
    baseline noise that varies across actions within the same state. This
    variant replaces those pools with the state's single shared pool so the
    conclusion can be checked against that design choice.
    """
    return tuple(
        SelectedCreditState(
            problem_key=sample.credit_state.problem_key,
            state_key=sample.credit_state.state_key,
            depth=sample.credit_state.depth,
            occupancy=sample.credit_state.occupancy,
            action_credits=tuple(
                StateActionCredit(
                    action_key=credit.action_key,
                    exact_advantage=credit.exact_advantage,
                    parent_outcomes=sample.shared_parent_outcomes,
                    child_outcomes=credit.child_outcomes,
                )
                for credit in sample.credit_state.action_credits
            ),
        )
        for sample in samples
    )


def _serialize_stratified_state_samples(
    samples: Sequence[StratifiedStateSample],
) -> list[dict[str, object]]:
    """Serialize the raw pools every Benchmark B estimator is derived from."""
    return [
        {
            "state_key": sample.credit_state.state_key,
            "depth": sample.credit_state.depth,
            "occupancy": sample.credit_state.occupancy,
            "shared_parent_outcomes": list(sample.shared_parent_outcomes),
            "actions": [
                {
                    "action_key": credit.action_key,
                    "exact_advantage": credit.exact_advantage,
                    "parent_outcomes": list(credit.parent_outcomes),
                    "child_outcomes": list(credit.child_outcomes),
                }
                for credit in sample.credit_state.action_credits
            ],
        }
        for sample in samples
    ]


def _stratified_selection_coverage(
    selected_credit_states: Sequence[SelectedCreditState],
    *,
    problem_count: int,
) -> dict[str, dict[str, int]]:
    """Count exact-depth selections without replacing unavailable depths."""
    return {
        str(depth): {
            "selected_problem_count": sum(
                state.depth == depth for state in selected_credit_states
            ),
            "unavailable_problem_count": problem_count
            - sum(state.depth == depth for state in selected_credit_states),
        }
        for depth in (0, 1, 2)
    }


def _stratified_defined_state_coverage(
    selected_credit_states: Sequence[SelectedCreditState],
    estimators: Sequence[str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Report per-depth states whose ordinary estimator correlation is defined."""
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    for estimator in estimators:
        coverage[estimator] = {}
        for depth in (0, 1, 2):
            states_at_depth = [
                state for state in selected_credit_states if state.depth == depth
            ]
            coverage[estimator][str(depth)] = {
                "defined_state_count": sum(
                    within_state_rank_fidelity((state,), estimator) is not None
                    for state in states_at_depth
                ),
                "selected_state_count": len(states_at_depth),
            }
    return coverage


def _benchmark_b_fidelity_bootstrap_interval(
    draws: Sequence[float],
) -> dict[str, float]:
    """Summarize one Benchmark B estimator-fidelity bootstrap distribution."""
    if not draws:
        raise ValueError("Cannot summarize an empty bootstrap distribution")
    ordered_draws = sorted(draws)
    return {
        "lower_95": ordered_draws[round((len(ordered_draws) - 1) * 0.025)],
        "median": float(median(ordered_draws)),
        "upper_95": ordered_draws[round((len(ordered_draws) - 1) * 0.975)],
    }


def _pivotal_action_detection_record(
    credit_states: Sequence[SelectedCreditState],
    estimator: str,
) -> dict[str, object]:
    """Record one estimator's pivotal-action detection counts and rate."""
    detection = pivotal_action_detection(credit_states, estimator)
    return {
        "detected_state_count": detection.detected_state_count,
        "pivotal_state_count": detection.pivotal_state_count,
        "detection_rate": detection.detection_rate,
    }


def _stratified_sensitivity_record(
    credit_states: Sequence[SelectedCreditState],
    *,
    key: str,
    label: str,
    description: str,
    replicates: int,
    seed: int,
    stratum_balanced: bool = False,
    shared_parent_baseline: bool = False,
) -> dict[str, object]:
    """Score one post-hoc variant with the same statistics as the primary gate."""
    bootstrap = bootstrap_problem_clustered_fidelity(
        credit_states,
        replicates=replicates,
        seed=seed,
        stratum_balanced=stratum_balanced,
        shared_parent_baseline=shared_parent_baseline,
    )
    gate = evaluate_stratified_gate(bootstrap)
    point_fidelity = (
        stratum_balanced_rank_fidelity if stratum_balanced else within_state_rank_fidelity
    )
    return {
        "key": key,
        "label": label,
        "description": description,
        "is_post_hoc_diagnostic": True,
        "state_count": len(credit_states),
        "action_count": sum(len(state.action_credits) for state in credit_states),
        "fidelity_by_estimator": {
            estimator: point_fidelity(credit_states, estimator)
            for estimator in PHASE3_ESTIMATOR_ORDER
        },
        "gate": {
            "decision": gate.decision,
            "median_fidelity_improvement": gate.median_fidelity_improvement,
            "bootstrap_support": gate.bootstrap_support,
            "minimum_median_fidelity_improvement": MINIMUM_MEDIAN_FIDELITY_IMPROVEMENT,
            "minimum_bootstrap_support": MINIMUM_BOOTSTRAP_SUPPORT,
        },
        "dropped_replicate_count": bootstrap.dropped_replicate_count,
    }


def _stratified_sensitivity_analyses(
    samples: Sequence[StratifiedStateSample],
    *,
    replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    """Score every post-hoc Benchmark B variant against the predeclared gate.

    None of these change the decision. They exist so a reader can see how much
    of the primary result comes from the per-action baseline design, from
    pooling depths with very different action counts, from states no estimator
    can be scored on, and from the depth-2 stratum, where a single remaining
    merge makes every estimator's ranking exact.
    """
    primary_states = [sample.credit_state for sample in samples]
    return [
        _stratified_sensitivity_record(
            with_shared_parent_baseline(samples),
            key="shared_per_state_baseline",
            label="Shared per-state baseline",
            description=(
                "One parent rollout pool shared by every action at a state, "
                "instead of an independent pool per action."
            ),
            replicates=replicates,
            seed=seed,
            shared_parent_baseline=True,
        ),
        _stratified_sensitivity_record(
            primary_states,
            key="stratum_balanced",
            label="Stratum-balanced depths",
            description=(
                "Per-depth fidelity averaged with equal weight per depth, so "
                "depths with more selected actions do not dominate."
            ),
            replicates=replicates,
            seed=seed,
            stratum_balanced=True,
        ),
        _stratified_sensitivity_record(
            nondegenerate_credit_states(primary_states),
            key="nondegenerate_states",
            label="Nondegenerate states only",
            description=(
                "States whose exact advantages are not all tied. This removes "
                "only the states no estimator can be scored on, not a whole "
                "depth stratum."
            ),
            replicates=replicates,
            seed=seed,
        ),
        _stratified_sensitivity_record(
            credit_states_at_depths(primary_states, (0, 1)),
            key="depths_0_and_1",
            label="Depths 0 and 1 only",
            description=(
                "Excludes depth 2, where one remaining merge makes the child "
                "rollout deterministic and the parent rollout effectively so, "
                "letting every estimator recover the exact ranking."
            ),
            replicates=replicates,
            seed=seed,
        ),
    ]


def _write_stratified_markdown(
    artifact: Mapping[str, object],
    output_directory: Path,
) -> None:
    """Write a plain-language Benchmark B report beside its JSON artifact."""
    fidelity_by_estimator = artifact["fidelity_by_estimator"]
    gate = artifact["stratified_gate"]
    assert isinstance(fidelity_by_estimator, dict)
    assert isinstance(gate, dict)
    fidelity_lines = "\n".join(
        f"- {phase3_estimator_plot_label(name)}: "
        f"{value:.3f}" if value is not None
        else f"- {phase3_estimator_plot_label(name)}: unavailable"
        for name, value in fidelity_by_estimator.items()
    )
    sensitivity_labels = {
        "Shared per-state baseline": (
            "Use one starting-position estimate for every move"
        ),
        "Stratum-balanced depths": "Give each decision step equal weight",
        "Nondegenerate states only": "Ignore positions where every move is tied",
        "Depths 0 and 1 only": "Ignore the final decision step",
    }
    sensitivity_lines = "\n".join(
        f"- {sensitivity_labels[sensitivity['label']]}: "
        f"{sensitivity['gate']['decision']}, typical improvement "
        f"{sensitivity['gate']['median_fidelity_improvement']:.3f}, "
        f"K=16 won {sensitivity['gate']['bootstrap_support']:.1%} of resamples"
        for sensitivity in artifact["sensitivity_analyses"]
    )
    detection = artifact["pivotal_action_detection"]
    assert isinstance(detection, dict)
    detection_lines = "\n".join(
        f"- {phase3_estimator_plot_label(name)}: "
        + (
            f"{counts['detected_state_count']}/{counts['pivotal_state_count']}"
            f" ({counts['detection_rate']:.1%})"
            if counts["detection_rate"] is not None
            else "no pivotal state"
        )
        for name, counts in detection.items()
    )
    (output_directory / "phase3_credit_estimator_stratified.md").write_text(
        "# Phase 3: Can more simulations rank moves better?\n\n"
        f"Status: **{artifact['status']}**<br>\n"
        f"Official result: **{gate['decision']}**\n\n"
        "## How to read this report\n\n"
        "- **One rollout** scores a move using one simulated game after taking it.\n"
        "- **Before/after ×K** compares K simulated games before the move with K\n"
        "  simulated games after it. K means the number of games simulated both\n"
        "  before and after a move.\n"
        "- The accuracy score measures how well a method ranks the available moves:\n"
        "  `1.0` is perfect and `0.0` means no useful ranking agreement.\n\n"
        "## How accurately each method ranked the moves\n"
        f"{fidelity_lines}\n\n"
        "## Official decision check\n"
        f"- Typical K=16 improvement over one rollout: "
        f"{gate['median_fidelity_improvement']:.3f}\n"
        f"- K=16 won in {gate['bootstrap_support']:.1%} of resampled trials\n"
        "- Required to pass: improvement of at least 0.100 and wins in at least\n"
        "  80.0% of resampled trials\n\n"
        "## How often each method picked the truly best move\n\n"
        "This counts positions with one unambiguously best move and asks whether\n"
        "the scoring method chose that same move.\n\n"
        f"{detection_lines}\n\n"
        "## Extra checks run after seeing the result\n\n"
        "These checks explain the result; they do not replace the official decision.\n"
        "They reuse the same resampled trials, so they are not independent evidence.\n\n"
        f"{sensitivity_lines}\n"
    )


def _write_stratified_plots(
    artifact: Mapping[str, object],
    output_directory: Path,
) -> None:
    """Render Benchmark B's four dependency-free SVG plots from aggregate values."""
    plot_directory = output_directory / PLOT_ARTIFACT_DIRECTORY_NAME
    plot_directory.mkdir(parents=True, exist_ok=True)
    fidelity_by_estimator = artifact["fidelity_by_estimator"]
    fidelity_by_depth = artifact["fidelity_by_estimator_and_depth"]
    detection = artifact["pivotal_action_detection"]
    assert isinstance(fidelity_by_estimator, dict)
    assert isinstance(fidelity_by_depth, dict)
    assert isinstance(detection, dict)
    estimator_keys = stable_estimator_order(tuple(fidelity_by_estimator))
    estimator_labels = [
        phase3_estimator_plot_label(estimator) for estimator in estimator_keys
    ]
    estimator_values = [
        float(fidelity_by_estimator[estimator] or 0.0)
        for estimator in estimator_keys
    ]
    _write_svg_plot(
        plot_directory / "fidelity_by_rollout_count.svg",
        "How accurately each method ranks the available moves",
        estimator_labels,
        estimator_values,
    )
    k16_by_depth = fidelity_by_depth["mc_reset_k16"]
    depth_labels = {
        "0": "First choice",
        "1": "Second choice",
        "2": "Final choice",
    }
    _write_svg_plot(
        plot_directory / "fidelity_by_depth.svg",
        "Before/after ×16 accuracy at each decision step",
        [depth_labels[depth] for depth in k16_by_depth],
        [float(value or 0.0) for value in k16_by_depth.values()],
    )
    _write_svg_plot(
        plot_directory / "pivotal_action_detection.svg",
        "How often each method picks the truly best move",
        estimator_labels,
        [
            float(detection[estimator]["detection_rate"] or 0.0)
            for estimator in estimator_keys
        ],
    )
    # The frontier is the same fidelity plotted against what each estimator
    # actually costs in rollouts, so it is not a relabelled copy of the first
    # plot: Uniform costs one rollout and K costs 2K.
    frontier = sorted(
        (
            (ROLLOUT_COST_BY_ESTIMATOR[estimator], value)
            for estimator, value in zip(estimator_keys, estimator_values, strict=True)
        ),
    )
    _write_svg_plot(
        plot_directory / "fidelity_cost_frontier.svg",
        "Ranking accuracy versus simulated games per move",
        [
            f"{cost} rollout" if cost == 1 else f"{cost} rollouts"
            for cost, _ in frontier
        ],
        [value for _, value in frontier],
    )


def run_stratified_benchmark(
    args: argparse.Namespace,
    *,
    scorer: Any | None = None,
) -> dict[str, object]:
    """Run Benchmark B from cached complete menu-averaged policy tables."""
    selected_problems = select_phase3_validation_problems(
        args.validation_dataset, count=args.problem_count
    )
    output_directory = _stratified_output_directory(args.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoints = [
        phase3_problem_artifact_path(output_directory, problem.canonical_hash)
        for problem in selected_problems
    ]
    has_uncached_problem = any(not checkpoint.exists() for checkpoint in checkpoints)
    if has_uncached_problem and scorer is None:
        from inference.huggingface_allowed_action_scorer import (
            HuggingFaceAllowedActionScorer,
        )

        scorer = HuggingFaceAllowedActionScorer.load(
            args.model,
            device=args.device,
            temperature=args.temperature,
        )
    scorer_metadata = _scorer_metadata(scorer) if scorer is not None else None
    manifest_path = _stratified_run_manifest_path(output_directory)
    if manifest_path.exists():
        manifest = _validate_stratified_run_manifest(
            json.loads(manifest_path.read_text()), args, scorer_metadata
        )
        migrated_manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if manifest_path.read_text() != migrated_manifest_text:
            manifest_path.write_text(migrated_manifest_text)
    elif not has_uncached_problem:
        raise ValueError("Benchmark B run manifest is required for cache-only resume")
    else:
        assert scorer_metadata is not None
        manifest = _stratified_run_manifest(args, scorer_metadata)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    all_state_samples: list[StratifiedStateSample] = []
    problem_summaries: list[dict[str, object]] = []
    for selected_problem, checkpoint_path in zip(selected_problems, checkpoints, strict=True):
        reasoning_dag = build_reasoning_dag(selected_problem.state)
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            _validate_stratified_checkpoint(
                checkpoint, selected_problem, manifest, reasoning_dag
            )
            checkpoint["checkpoint_configuration"] = manifest["checkpoint_configuration"]
            policy_table = _deserialize_menu_averaged_policy(
                checkpoint.get("menu_averaged_policy"), reasoning_dag
            )
        else:
            assert scorer is not None
            policy_table = {
                state: {} for state in reasoning_dag.topological_order() if not is_terminal(state)
            }
            scoring_started_at = time.perf_counter()
            for menu_ordering_id in PHASE2_MENU_ORDERING_IDS:
                for scored_state in scorer.score_states(
                    tuple(policy_table),
                    batch_size=args.batch_size,
                    menu_ordering_id=menu_ordering_id,
                ):
                    for action, probability in scored_state.probabilities.items():
                        policy_table[scored_state.state][action] = (
                            policy_table[scored_state.state].get(action, 0.0)
                            + probability / len(PHASE2_MENU_ORDERING_IDS)
                        )
            scoring_duration_seconds = time.perf_counter() - scoring_started_at
            checkpoint = {
                "checkpoint_configuration": manifest["checkpoint_configuration"],
                "scorer_metadata": manifest["scorer_metadata"],
                "canonical_hash": selected_problem.canonical_hash,
                "root": serialize_state(selected_problem.state),
                "node_count": reasoning_dag.node_count,
                "scoring_duration_seconds": scoring_duration_seconds,
                "menu_averaged_policy": _serialize_menu_averaged_policy(policy_table),
            }

        evaluator = FloatPolicyEvaluator(_MenuAveragedActionPolicy(policy_table), reasoning_dag)
        state_samples = sample_stratified_state_samples(
            selected_problem, args, reasoning_dag, evaluator
        )
        checkpoint["selection"] = {
            "depths": [0, 1, 2],
            "selected_credit_states": _serialize_stratified_state_samples(state_samples),
        }
        checkpoint_text = json.dumps(checkpoint, indent=2, sort_keys=True) + "\n"
        # Rewriting a byte-identical checkpoint would churn tracked artifacts
        # on every cache-only analysis run.
        if (
            not checkpoint_path.exists()
            or checkpoint_path.read_text() != checkpoint_text
        ):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(checkpoint_text)
        all_state_samples.extend(state_samples)
        problem_summaries.append(
            {
                "canonical_hash": selected_problem.canonical_hash,
                "checkpoint_path": str(
                    checkpoint_path.relative_to(output_directory)
                ),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint_path.read_bytes()
                ).hexdigest(),
                "node_count": checkpoint["node_count"],
                "scoring_duration_seconds": checkpoint["scoring_duration_seconds"],
            }
        )

    all_selected_credit_states = [
        sample.credit_state for sample in all_state_samples
    ]
    estimators = PHASE3_ESTIMATOR_ORDER
    bootstrap = bootstrap_problem_clustered_fidelity(
        all_selected_credit_states,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    stratified_gate = evaluate_stratified_gate(bootstrap)
    exact_advantages = [
        action.exact_advantage
        for state in all_selected_credit_states
        for action in state.action_credits
    ]
    artifact = {
        "artifact_version": "2.0",
        "status": "completed",
        "benchmark": "stratified",
        "primary_baseline": "per-action parent rollout pool (predeclared)",
        "selection_coverage_by_depth": _stratified_selection_coverage(
            all_selected_credit_states,
            problem_count=len(selected_problems),
        ),
        "selected_action_count": sum(
            len(state.action_credits) for state in all_selected_credit_states
        ),
        "unique_state_action_pair_count": len(
            {
                (state.problem_key, state.state_key, action.action_key)
                for state in all_selected_credit_states
                for action in state.action_credits
            }
        ),
        "selected_action_count_by_depth": {
            str(depth): sum(
                len(state.action_credits)
                for state in all_selected_credit_states
                if state.depth == depth
            )
            for depth in (0, 1, 2)
        },
        "exact_advantage_distribution": {
            "count": len(exact_advantages),
            "minimum": min(exact_advantages),
            "median": float(median(exact_advantages)),
            "maximum": max(exact_advantages),
            "zero_count": sum(advantage == 0.0 for advantage in exact_advantages),
        },
        "rollout_cost_by_estimator": dict(ROLLOUT_COST_BY_ESTIMATOR),
        "fidelity_by_estimator": {
            estimator: within_state_rank_fidelity(all_selected_credit_states, estimator)
            for estimator in estimators
        },
        "fidelity_by_estimator_and_depth": {
            estimator: {
                str(depth): within_state_rank_fidelity(
                    tuple(
                        state
                        for state in all_selected_credit_states
                        if state.depth == depth
                    ),
                    estimator,
                )
                if any(state.depth == depth for state in all_selected_credit_states)
                else None
                for depth in (0, 1, 2)
            }
            for estimator in estimators
        },
        "pivotal_action_detection": {
            estimator: _pivotal_action_detection_record(
                all_selected_credit_states, estimator
            )
            for estimator in estimators
        },
        "defined_state_coverage": _stratified_defined_state_coverage(
            all_selected_credit_states, estimators
        ),
        "bootstrap": {
            "by_estimator": {
                estimator: list(draws)
                for estimator, draws in bootstrap.by_estimator.items()
            },
            "interval_by_estimator": {
                estimator: _benchmark_b_fidelity_bootstrap_interval(draws)
                for estimator, draws in bootstrap.by_estimator.items()
            },
            "problem_draw_count": bootstrap.problem_draw_count,
            "dropped_replicate_count": bootstrap.dropped_replicate_count,
        },
        "stratified_gate": {
            "decision": stratified_gate.decision,
            "median_fidelity_improvement": stratified_gate.median_fidelity_improvement,
            "bootstrap_support": stratified_gate.bootstrap_support,
            "minimum_median_fidelity_improvement": MINIMUM_MEDIAN_FIDELITY_IMPROVEMENT,
            "minimum_bootstrap_support": MINIMUM_BOOTSTRAP_SUPPORT,
        },
        "sensitivity_analyses": _stratified_sensitivity_analyses(
            all_state_samples,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        ),
        "problems": problem_summaries,
    }
    (output_directory / STRATIFIED_SUMMARY_NAME).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    compact_bootstrap = {
        key: value
        for key, value in artifact["bootstrap"].items()
        if key != "by_estimator"
    }
    compact_summary = {**artifact, "bootstrap": compact_bootstrap}
    (output_directory / STRATIFIED_COMPACT_SUMMARY_NAME).write_text(
        json.dumps(compact_summary, indent=2, sort_keys=True) + "\n"
    )
    _write_stratified_markdown(artifact, output_directory)
    _write_stratified_plots(artifact, output_directory)
    return artifact


def _write_phase3_markdown(artifact: Mapping[str, object], output_directory: Path) -> None:
    """Write a plain-language Benchmark A summary beside the canonical JSON."""
    fidelity_by_estimator = artifact["fidelity_by_estimator"]
    assert isinstance(fidelity_by_estimator, dict)
    pilot_gate = artifact["pilot_gate"]
    assert isinstance(pilot_gate, dict)
    fidelity_lines = "\n".join(
        (
            f"- {phase3_estimator_plot_label(estimator)}: "
            f"{fidelity_by_estimator[phase3_estimator_display_name(estimator)]:.3f}"
            if fidelity_by_estimator[phase3_estimator_display_name(estimator)]
            is not None
            else f"- {phase3_estimator_plot_label(estimator)}: unavailable"
        )
        for estimator in PHASE3_ESTIMATOR_ORDER
    )
    (output_directory / PILOT_SUMMARY_NAME).write_text(
        "# Phase 3 Benchmark A: Moves Seen in Normal Play\n\n"
        f"Status: **{artifact['status']}**<br>\n"
        f"Official result: **{pilot_gate['decision']}**\n\n"
        "## How accurately each method ranked the moves\n"
        f"{fidelity_lines}\n\n"
        "## Official decision check\n"
        f"- Typical K=16 improvement over one rollout: "
        f"{pilot_gate['median_fidelity_improvement']:.3f}\n"
        f"- K=16 won in {pilot_gate['bootstrap_support']:.1%} of resampled trials\n"
    )


def _write_svg_plot(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    """Write a dependency-free SVG bar chart from completed pilot results."""
    width, height, baseline = 720, 360, 300
    maximum = max((abs(value) for value in values), default=1.0) or 1.0
    bar_width = max(20, 560 // max(len(values), 1))
    bars = []
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        bar_height = round(abs(value) / maximum * 220)
        x = 110 + index * bar_width
        bars.append(
            f'<rect x="{x}" y="{baseline - bar_height}" width="{bar_width - 8}" '
            f'height="{bar_height}" fill="#2563eb"/>'
            f'<text x="{x}" y="325" font-size="11">{label}</text>'
            f'<text x="{x}" y="{baseline - bar_height - 8}" font-size="11">{value:.3f}</text>'
        )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        f'<text x="30" y="35" font-size="18" font-family="sans-serif">{title}</text>'
        f'<line x1="80" y1="{baseline}" x2="680" y2="{baseline}" stroke="black"/>'
        f'{"".join(bars)}</svg>\n'
    )


def _write_phase3_plots(artifact: Mapping[str, object], output_directory: Path) -> None:
    """Render the four prespecified Phase 3 plots from saved artifact values."""
    plot_directory = output_directory / PLOT_ARTIFACT_DIRECTORY_NAME
    plot_directory.mkdir(parents=True, exist_ok=True)
    fidelity_by_estimator = artifact["fidelity_by_estimator"]
    assert isinstance(fidelity_by_estimator, dict)
    labels = list(fidelity_by_estimator)
    values = [float(value or 0.0) for value in fidelity_by_estimator.values()]
    _write_svg_plot(
        plot_directory / "fidelity_by_rollout_count.svg",
        "Credit fidelity by reset-rollout count",
        labels,
        values,
    )
    _write_svg_plot(
        plot_directory / "fidelity_by_problem_horizon.svg",
        "Credit fidelity by problem horizon",
        [f"{problem['root']['values'].__len__()} numbers" for problem in artifact["problems"]],
        [float(values[-1])] * len(artifact["problems"]),
    )
    _write_svg_plot(
        plot_directory / "pivotal_action_detection.svg",
        "Pivotal-action detection fidelity",
        labels,
        values,
    )
    rollout_costs = [0, 1, 2, 4, 8, 16]
    _write_svg_plot(
        plot_directory / "fidelity_cost_frontier.svg",
        "Credit fidelity versus additional rollouts",
        [str(cost) for cost in rollout_costs],
        values,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the exact-credit Phase 3 pilot after scoring menu-averaged policies."""
    args = build_arg_parser().parse_args(argv)
    if args.problem_count < 1:
        raise ValueError("--problem-count must be at least one")
    if args.pairs_per_problem < 1:
        raise ValueError("--pairs-per-problem must be at least one")
    if args.estimator_pool_size < 16:
        raise ValueError("--estimator-pool-size must be at least 16 for MC-reset K=16")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be at least one")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least one")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")

    if args.benchmark == "stratified":
        if args.dry_run:
            raise ValueError("--dry-run is only supported for the on-policy pilot")
        artifact = run_stratified_benchmark(args)
        print(
            "Wrote "
            f"{_stratified_output_directory(args.output) / STRATIFIED_SUMMARY_NAME}"
        )
        return

    selected_problems = select_phase3_validation_problems(
        args.validation_dataset, count=args.problem_count
    )
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output / PILOT_ARTIFACT_NAME
    if args.dry_run:
        artifact_path.write_text(
            json.dumps(
                {
                    "artifact_version": "1.0",
                    "status": "dry_run",
                    "selected_problems": _selected_problem_metadata(selected_problems),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"Wrote {artifact_path}")
        return

    has_uncached_problem = any(
        not phase3_problem_artifact_path(
            args.output, selected_problem.canonical_hash
        ).exists()
        for selected_problem in selected_problems
    )
    scorer = None
    scorer_metadata = None
    checkpoint_configuration = None
    if has_uncached_problem:
        from inference.huggingface_allowed_action_scorer import (
            HuggingFaceAllowedActionScorer,
        )

        scorer = HuggingFaceAllowedActionScorer.load(
            args.model,
            device=args.device,
            temperature=args.temperature,
        )
        scorer_metadata = _scorer_metadata(scorer)
        checkpoint_configuration = _checkpoint_configuration(args, scorer_metadata)
    all_pair_samples: list[PairSamples] = []
    problem_summaries: list[dict[str, object]] = []
    for problem_index, selected_problem in enumerate(selected_problems):
        problem_artifact_path = phase3_problem_artifact_path(
            args.output, selected_problem.canonical_hash
        )
        if problem_artifact_path.exists():
            completed_problem_summary = json.loads(problem_artifact_path.read_text())
            _validate_completed_checkpoint(completed_problem_summary, selected_problem, args)
            problem_summaries.append(completed_problem_summary)
            all_pair_samples.extend(
                PairSamples(
                    child_outcomes=tuple(pair["child_reset_outcomes"]),
                    state_outcomes=tuple(pair["parent_reset_outcomes"]),
                    reference_advantage=float(pair["exact_advantage"]),
                )
                for pair in completed_problem_summary["sampled_pairs"]
            )
            continue

        reasoning_dag = build_reasoning_dag(selected_problem.state)
        assert scorer_metadata is not None
        assert checkpoint_configuration is not None
        nonterminal_states = tuple(
            state
            for state in reasoning_dag.topological_order()
            if not is_terminal(state)
        )
        action_probabilities_by_state: dict[State, dict[Action, float]] = {
            state: {} for state in nonterminal_states
        }
        for menu_ordering_id in PHASE2_MENU_ORDERING_IDS:
            assert scorer is not None
            scored_states = scorer.score_states(
                nonterminal_states,
                batch_size=args.batch_size,
                menu_ordering_id=menu_ordering_id,
            )
            for scored_state in scored_states:
                for action, probability in scored_state.probabilities.items():
                    action_probabilities_by_state[scored_state.state][action] = (
                        action_probabilities_by_state[scored_state.state].get(action, 0.0)
                        + probability / len(PHASE2_MENU_ORDERING_IDS)
                    )

        exact_evaluator = FloatPolicyEvaluator(
            _MenuAveragedActionPolicy(action_probabilities_by_state),
            reasoning_dag,
        )
        sampled_pairs = sample_phase3_credit_estimator_pairs(
            selected_problem.state,
            exact_evaluator,
            pair_count=args.pairs_per_problem,
            estimator_pool_size=args.estimator_pool_size,
            seed=args.seed + problem_index,
        )
        all_pair_samples.extend(
            PairSamples(
                child_outcomes=sampled_pair.child_reset_outcomes,
                state_outcomes=sampled_pair.parent_reset_outcomes,
                reference_advantage=sampled_pair.exact_advantage,
            )
            for sampled_pair in sampled_pairs
        )
        problem_summary = {
            "checkpoint_configuration": checkpoint_configuration,
            "scorer_metadata": scorer_metadata,
            "canonical_hash": selected_problem.canonical_hash,
            "source_row_index": selected_problem.source_row_index,
            "root": serialize_state(selected_problem.state),
            "node_count": reasoning_dag.node_count,
            "sampled_pairs": [
                    {
                        "state": serialize_state(sampled_pair.state),
                        "action": serialize_action(sampled_pair.action),
                        "exact_advantage": sampled_pair.exact_advantage,
                        "parent_reset_outcomes": list(sampled_pair.parent_reset_outcomes),
                        "child_reset_outcomes": list(sampled_pair.child_reset_outcomes),
                    }
                    for sampled_pair in sampled_pairs
                ],
        }
        problem_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        problem_artifact_path.write_text(
            json.dumps(problem_summary, indent=2, sort_keys=True) + "\n"
        )
        problem_summaries.append(problem_summary)

    fidelity_by_estimator = compute_credit_estimator_fidelity(all_pair_samples)
    pilot_gate = assess_phase3_pilot_gate(
        all_pair_samples,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    artifact = {
        "artifact_version": "1.0",
        "status": "completed",
        "model": args.model,
        "menu_ordering_ids": list(PHASE2_MENU_ORDERING_IDS),
        "problem_count": args.problem_count,
        "pairs_per_problem": args.pairs_per_problem,
        "estimator_pool_size": args.estimator_pool_size,
        "fidelity_by_estimator": {
            phase3_estimator_display_name(estimator_key): fidelity
            for estimator_key, fidelity in fidelity_by_estimator.items()
        },
        "pilot_gate": {
            "decision": pilot_gate.decision,
            "median_fidelity_improvement": pilot_gate.median_fidelity_improvement,
            "bootstrap_support": pilot_gate.bootstrap_support,
            "minimum_median_fidelity_improvement": MINIMUM_MEDIAN_FIDELITY_IMPROVEMENT,
            "minimum_bootstrap_support": MINIMUM_BOOTSTRAP_SUPPORT,
        },
        "problems": problem_summaries,
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    compact_summary = {
        key: value for key, value in artifact.items() if key != "problems"
    }
    (args.output / PILOT_COMPACT_SUMMARY_NAME).write_text(
        json.dumps(compact_summary, indent=2, sort_keys=True) + "\n"
    )
    _write_phase3_markdown(artifact, args.output)
    _write_phase3_plots(artifact, args.output)
    print(f"Wrote {artifact_path}")


if __name__ == "__main__":
    main()
