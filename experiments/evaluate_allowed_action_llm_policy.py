"""Score and evaluate a stationary allowed-action LLM policy.

The ``score`` command loads the model and writes a resumable probability cache.
The ``evaluate`` command uses only that cache to compute policy values, validate
them against Monte Carlo rollouts, and produce mixed-credit trajectory artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Mapping, Sequence

from env.countdown import (
    DEFAULT_RULES,
    Action,
    Operator,
    Problem,
    RuleSet,
    State,
    allowed_actions,
    is_terminal,
    reward,
    transition,
)
from oracle.evaluation import Policy, ReasoningDag, build_reasoning_dag
from oracle.float_policy_evaluation import (
    NUMERICAL_TOLERANCE,
    FloatPolicyEvaluator,
)
from policies.allowed_action_llm_policy import (
    ALLOWED_ACTION_POLICY_CACHE_NAME,
    POLICY_CACHE_SCHEMA_VERSION,
    PROMPT_VERSION,
    AllowedActionPolicyCache,
    CachedAllowedActionLLMPolicy,
    MenuOption,
    PolicyCacheMetadata,
    StateActionProbabilities,
    load_cache_snapshot,
    save_cache_snapshot,
    serialize_action,
    serialize_rules,
    serialize_state,
    validate_cache_coverage,
)

THROUGHPUT_FILE = "throughput_benchmark.json"
EVALUATION_FILE = "policy_evaluation_summary.json"
TRAJECTORY_JSON_FILE = "mixed_credit_trajectory.json"
TRAJECTORY_MARKDOWN_FILE = "mixed_credit_trajectory.md"


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    state: State
    menu: tuple[MenuOption, ...]
    action_probabilities: Mapping[Action, float]
    action: Action
    action_probability: float
    next_state: State
    v_pi: float
    q_pi: float
    advantage: float


@dataclass(frozen=True, slots=True)
class EvaluatedTrajectory:
    steps: tuple[TrajectoryStep, ...]
    final_state: State
    final_reward: int
    sample_index: int


def sample_evaluated_trajectory(
    root: State,
    policy: CachedAllowedActionLLMPolicy,
    evaluator: FloatPolicyEvaluator,
    rng: random.Random,
    sample_index: int,
    rules: RuleSet = DEFAULT_RULES,
) -> EvaluatedTrajectory:
    state = root
    steps: list[TrajectoryStep] = []
    while not is_terminal(state, rules):
        actions = allowed_actions(state, rules)
        record = policy.state_action_probabilities(state, actions)
        probabilities = record.probabilities
        action = rng.choices(
            actions,
            weights=[probabilities[candidate] for candidate in actions],
            k=1,
        )[0]
        next_state = transition(state, action, rules)
        steps.append(
            TrajectoryStep(
                state=state,
                menu=record.menu,
                action_probabilities=dict(probabilities),
                action=action,
                action_probability=probabilities[action],
                next_state=next_state,
                v_pi=evaluator.v_pi(state),
                q_pi=evaluator.q_pi(state, action),
                advantage=evaluator.a_pi(state, action),
            )
        )
        state = next_state
    return EvaluatedTrajectory(
        steps=tuple(steps),
        final_state=state,
        final_reward=reward(state),
        sample_index=sample_index,
    )


def monte_carlo_success_rate(
    root: State,
    policy: Policy[float],
    *,
    rollout_count: int,
    seed: int,
    rules: RuleSet = DEFAULT_RULES,
) -> float:
    if rollout_count < 1:
        raise ValueError("rollout_count must be at least 1")
    rng = random.Random(seed)
    successes = 0
    for _ in range(rollout_count):
        state = root
        while not is_terminal(state, rules):
            actions = allowed_actions(state, rules)
            probabilities = policy.action_probs(state, actions)
            action = rng.choices(
                actions,
                weights=[probabilities[candidate] for candidate in actions],
                k=1,
            )[0]
            state = transition(state, action, rules)
        successes += reward(state)
    return successes / rollout_count


def validate_numerical_policy_invariants(
    dag: ReasoningDag,
    evaluator: FloatPolicyEvaluator,
    rules: RuleSet = DEFAULT_RULES,
) -> None:
    for state in dag.topological_order():
        value = evaluator.v_pi(state)
        if not -NUMERICAL_TOLERANCE <= value <= 1.0 + NUMERICAL_TOLERANCE:
            raise RuntimeError(f"Vpi={value} is outside [0, 1] at {state!r}")
        if is_terminal(state, rules):
            continue

        probabilities = evaluator.action_probs(state)
        bellman_value = math.fsum(
            probabilities[edge.action] * evaluator.v_pi(edge.target_state)
            for edge in dag.out_edges(state)
        )
        if abs(value - bellman_value) > NUMERICAL_TOLERANCE:
            raise RuntimeError(f"Bellman residual is nonzero at {state!r}")

        advantages = evaluator.advantages(state)
        weighted_advantage = math.fsum(
            probabilities[action] * advantages[action]
            for action in advantages
        )
        if abs(weighted_advantage) > NUMERICAL_TOLERANCE:
            raise RuntimeError(
                f"Policy-weighted advantage is nonzero at {state!r}"
            )

        for edge in dag.out_edges(state):
            q_pi = evaluator.q_pi(state, edge.action)
            if (
                abs(q_pi - evaluator.v_pi(edge.target_state))
                > NUMERICAL_TOLERANCE
            ):
                raise RuntimeError(f"Qpi child-value mismatch at {state!r}")
            expected_advantage = q_pi - value
            if (
                abs(evaluator.a_pi(state, edge.action) - expected_advantage)
                > NUMERICAL_TOLERANCE
            ):
                raise RuntimeError(f"Api definition mismatch at {state!r}")


def select_mixed_credit_trajectory(
    root: State,
    policy: CachedAllowedActionLLMPolicy,
    evaluator: FloatPolicyEvaluator,
    *,
    seed: int,
    max_attempts: int,
    rules: RuleSet = DEFAULT_RULES,
) -> EvaluatedTrajectory:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    rng = random.Random(seed)
    for sample_index in range(max_attempts):
        trajectory = sample_evaluated_trajectory(
            root,
            policy,
            evaluator,
            rng,
            sample_index,
            rules,
        )
        advantages = [step.advantage for step in trajectory.steps]
        if (
            trajectory.final_reward == 0
            and any(value > NUMERICAL_TOLERANCE for value in advantages)
            and any(value < -NUMERICAL_TOLERANCE for value in advantages)
        ):
            return trajectory
    raise RuntimeError(
        f"No failed mixed-credit trajectory found in {max_attempts} attempts"
    )


def _selected_label(step: TrajectoryStep) -> str:
    for option in step.menu:
        if option.action == step.action:
            return option.label
    raise ValueError(f"Selected action {step.action!r} is absent from its menu")


def trajectory_to_dict(
    trajectory: EvaluatedTrajectory,
    *,
    seed: int,
) -> dict[str, object]:
    outcome_credit = 1 if trajectory.final_reward == 1 else -1
    steps = []
    for index, step in enumerate(trajectory.steps):
        steps.append(
            {
                "step": index,
                "state": serialize_state(step.state),
                "menu": [
                    {
                        "label": option.label,
                        "action": serialize_action(option.action),
                        "probability": step.action_probabilities[option.action],
                        "selected": option.action == step.action,
                    }
                    for option in step.menu
                ],
                "selected_label": _selected_label(step),
                "selected_action": serialize_action(step.action),
                "selected_action_probability": step.action_probability,
                "next_state": serialize_state(step.next_state),
                "v_pi": step.v_pi,
                "q_pi": step.q_pi,
                "advantage": step.advantage,
                "outcome_only_display_credit": outcome_credit,
            }
        )
    return {
        "artifact_version": "1.0",
        "selection": {
            "seed": seed,
            "rule": (
                "first failed trajectory containing at least one positive and "
                "one negative advantage"
            ),
            "sample_index": trajectory.sample_index,
        },
        "final_state": serialize_state(trajectory.final_state),
        "final_environment_reward": trajectory.final_reward,
        "outcome_only_display_credit": outcome_credit,
        "steps": steps,
    }


def trajectory_to_markdown(
    trajectory: EvaluatedTrajectory,
    *,
    seed: int | None = None,
) -> str:
    outcome_credit = 1 if trajectory.final_reward == 1 else -1
    selection_text = f"sample index `{trajectory.sample_index}`"
    if seed is not None:
        selection_text += f" under seed `{seed}`"
    lines = [
        "# Mixed-Credit Trajectory",
        "",
        (
            f"Final environment reward: `{trajectory.final_reward}`. "
            f"Outcome-only display credit: `{outcome_credit}` at every step."
        ),
        "",
        (
            "The environment reward remains 0/1. “Outcome-only display credit” "
            "maps the final outcome to +1/-1 only for comparison with per-action "
            "Aπ. This trajectory was selected by the predeclared "
            "seed-and-first-match rule and does not estimate population frequency."
        ),
        "",
        f"Selection: {selection_text}.",
        "",
    ]
    for index, step in enumerate(trajectory.steps, start=1):
        lines.extend(
            [
                f"## Step {index}",
                "",
                (
                    f"State: numbers `{list(step.state.values)}`, "
                    f"target `{step.state.target}`"
                ),
                "",
                "Menu:",
            ]
        )
        for option in step.menu:
            marker = " **← selected**" if option.action == step.action else ""
            lines.append(
                f"- `{option.label}`: `{option.action.render()}` — "
                f"π={step.action_probabilities[option.action]:.8f}{marker}"
            )
        lines.extend(
            [
                "",
                f"Selected action: `{_selected_label(step)}` "
                f"(`{step.action.render()}`)",
                "",
                f"- `Vπ(s) = {step.v_pi:.8f}`",
                f"- `Qπ(s,a) = {step.q_pi:.8f}`",
                f"- `Aπ(s,a) = {step.advantage:+.8f}`",
                f"- Outcome-only display credit: `{outcome_credit}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value))


def _rules_from_metadata(metadata: PolicyCacheMetadata) -> RuleSet:
    try:
        operators = tuple(Operator(symbol) for symbol in metadata.rules["operators"])
        min_intermediate = int(metadata.rules["min_intermediate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed RuleSet in cache metadata: {error}") from error
    return RuleSet(operators=operators, min_intermediate=min_intermediate)


def _root_from_metadata(metadata: PolicyCacheMetadata) -> State:
    try:
        return State(
            values=tuple(metadata.root_problem["values"]),
            target=int(metadata.root_problem["target"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed root problem in cache metadata: {error}") from error


def _nonterminal_states(
    root: State,
    rules: RuleSet,
    *,
    max_nodes: int = 100_000,
) -> tuple[State, ...]:
    dag = build_reasoning_dag(root, rules, max_nodes=max_nodes)
    return tuple(
        state
        for state in dag.topological_order()
        if not is_terminal(state, rules)
    )


def run_score_command(args: argparse.Namespace) -> None:
    # Keep this import local: ``evaluate`` must not import model infrastructure.
    from inference.huggingface_allowed_action_scorer import (
        HuggingFaceAllowedActionScorer,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rules = DEFAULT_RULES
    root = Problem(values=tuple(args.numbers), target=args.target).to_state()
    nonterminal_states = _nonterminal_states(
        root,
        rules,
        max_nodes=args.max_nodes,
    )
    scorer = HuggingFaceAllowedActionScorer.load(
        args.model,
        device=args.device,
        temperature=args.temperature,
    )
    scorer_metadata = scorer.metadata
    metadata = PolicyCacheMetadata(
        schema_version=POLICY_CACHE_SCHEMA_VERSION,
        model_repository=scorer_metadata.model_repository,
        model_revision=scorer_metadata.model_revision,
        tokenizer_repository=scorer_metadata.tokenizer_repository,
        tokenizer_revision=scorer_metadata.tokenizer_revision,
        prompt_version=PROMPT_VERSION,
        enable_thinking=False,
        labels=scorer_metadata.label_inventory.labels,
        label_token_ids=scorer_metadata.label_inventory.token_ids,
        menu_ordering_id=args.menu_ordering_id,
        temperature=scorer_metadata.temperature,
        rules=serialize_rules(rules),
        device=scorer_metadata.device,
        dtype=scorer_metadata.dtype,
        software_versions={
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
        root_problem=serialize_state(root),
    )

    cache_path = output_dir / ALLOWED_ACTION_POLICY_CACHE_NAME
    if cache_path.exists():
        snapshot = load_cache_snapshot(cache_path, expected_metadata=metadata)
    else:
        snapshot = AllowedActionPolicyCache(metadata=metadata, records={})
        save_cache_snapshot(cache_path, snapshot)

    expected_states = set(nonterminal_states)
    extra_states = set(snapshot.records) - expected_states
    if extra_states:
        raise RuntimeError(
            f"Policy cache contains states outside this DAG: {extra_states!r}"
        )

    benchmark = scorer.benchmark(
        nonterminal_states,
        menu_ordering_id=args.menu_ordering_id,
        rules=rules,
    )
    measurements = benchmark.measurements
    if not measurements:
        raise RuntimeError("No throughput benchmark batch size succeeded")
    fastest_rate = max(measurement.states_per_second for measurement in measurements)
    estimated_seconds = len(nonterminal_states) / fastest_rate
    throughput = {
        "artifact_version": "1.0",
        "root_problem": serialize_state(root),
        "nonterminal_state_count": len(nonterminal_states),
        "representative_states": [
            serialize_state(state) for state in benchmark.states
        ],
        "measurements": [
            {
                "batch_size": measurement.batch_size,
                "state_count": measurement.state_count,
                "elapsed_seconds": measurement.elapsed_seconds,
                "states_per_second": measurement.states_per_second,
            }
            for measurement in measurements
        ],
        "fastest_states_per_second": fastest_rate,
        "estimated_full_scoring_seconds": estimated_seconds,
        "maximum_allowed_seconds": 30 * 60,
    }
    _write_json(output_dir / THROUGHPUT_FILE, throughput)
    if estimated_seconds > 30 * 60:
        raise RuntimeError(
            f"Estimated scoring time {estimated_seconds:.1f}s exceeds 30 minutes"
        )

    records = dict(snapshot.records)
    missing_states = [
        state for state in nonterminal_states if state not in records
    ]
    for start in range(0, len(missing_states), args.batch_size):
        state_batch = missing_states[start : start + args.batch_size]
        scored = scorer.score_states(
            state_batch,
            batch_size=args.batch_size,
            menu_ordering_id=args.menu_ordering_id,
            rules=rules,
        )
        if tuple(record.state for record in scored) != tuple(state_batch):
            raise RuntimeError("Scorer returned records in an unexpected state order")
        records.update({record.state: record for record in scored})
        snapshot = AllowedActionPolicyCache(
            metadata=metadata,
            records=records,
        )
        save_cache_snapshot(cache_path, snapshot)

    validate_cache_coverage(snapshot, expected_states)
    print(
        f"Cached {len(snapshot.records)} nonterminal states at {cache_path} "
        f"(estimated scoring time {estimated_seconds:.2f}s)"
    )


def run_evaluate_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    cache_path = output_dir / ALLOWED_ACTION_POLICY_CACHE_NAME
    snapshot = load_cache_snapshot(cache_path)
    rules = _rules_from_metadata(snapshot.metadata)
    root = _root_from_metadata(snapshot.metadata)
    dag = build_reasoning_dag(root, rules)
    expected_states = {
        state for state in dag.nodes if not is_terminal(state, rules)
    }
    validate_cache_coverage(snapshot, expected_states)

    policy = CachedAllowedActionLLMPolicy(snapshot.records)
    evaluator = FloatPolicyEvaluator(policy, dag, rules)
    validate_numerical_policy_invariants(dag, evaluator, rules)
    root_value = evaluator.v_pi(root)
    success_rate = monte_carlo_success_rate(
        root,
        policy,
        rollout_count=args.mc_rollouts,
        seed=args.seed,
        rules=rules,
    )
    standard_error = (
        math.sqrt(root_value * (1.0 - root_value) / args.mc_rollouts)
        if 0.0 < root_value < 1.0
        else 1e-9
    )
    margin = 5.0 * standard_error + 1e-6
    check_passed = abs(success_rate - root_value) <= margin
    summary: dict[str, object] = {
        "artifact_version": "1.0",
        "root_problem": serialize_state(root),
        "seed": args.seed,
        "monte_carlo_rollout_count": args.mc_rollouts,
        "artifact_attempt_limit": args.artifact_attempts,
        "numerical_invariants_passed": True,
        "numerical_root_v_pi": root_value,
        "monte_carlo_success_rate": success_rate,
        "monte_carlo_standard_error": standard_error,
        "monte_carlo_acceptance_margin": margin,
        "monte_carlo_check_passed": check_passed,
    }
    if not check_passed:
        summary["artifact_found"] = False
        _write_json(output_dir / EVALUATION_FILE, summary)
        raise RuntimeError(
            "Monte Carlo success rate disagrees with numerical Vpi: "
            f"estimate={success_rate}, value={root_value}, margin={margin}"
        )

    try:
        trajectory = select_mixed_credit_trajectory(
            root,
            policy,
            evaluator,
            seed=args.seed,
            max_attempts=args.artifact_attempts,
            rules=rules,
        )
    except RuntimeError:
        summary["artifact_found"] = False
        _write_json(output_dir / EVALUATION_FILE, summary)
        raise

    summary["artifact_found"] = True
    summary["selected_trajectory_sample_index"] = trajectory.sample_index
    _write_json(output_dir / EVALUATION_FILE, summary)
    _write_json(
        output_dir / TRAJECTORY_JSON_FILE,
        trajectory_to_dict(trajectory, seed=args.seed),
    )
    (output_dir / TRAJECTORY_MARKDOWN_FILE).write_text(
        trajectory_to_markdown(trajectory, seed=args.seed)
    )
    print(
        f"Vpi(root)={root_value:.8f}; Monte Carlo={success_rate:.8f}; "
        f"selected trajectory sample={trajectory.sample_index}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser(
        "score",
        help="Load the LLM and cache action probabilities for a complete DAG.",
    )
    score.add_argument("--numbers", type=int, nargs="+", required=True)
    score.add_argument("--target", type=int, required=True)
    score.add_argument("--output", type=str, required=True)
    score.add_argument("--model", default="Qwen/Qwen3-0.6B")
    score.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    score.add_argument("--temperature", type=float, default=1.0)
    score.add_argument("--menu-ordering-id", default="phase2-v1")
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--max-nodes", type=int, default=100_000)
    score.set_defaults(handler=run_score_command)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a completed policy cache without loading model weights.",
    )
    evaluate.add_argument("--output", type=str, required=True)
    evaluate.add_argument("--mc-rollouts", type=int, default=20_000)
    evaluate.add_argument("--artifact-attempts", type=int, default=10_000)
    evaluate.add_argument("--seed", type=int, default=20_260_815)
    evaluate.set_defaults(handler=run_evaluate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if getattr(args, "batch_size", 1) < 1:
        parser.error("--batch-size must be at least 1")
    if getattr(args, "mc_rollouts", 1) < 1:
        parser.error("--mc-rollouts must be at least 1")
    if getattr(args, "artifact_attempts", 1) < 1:
        parser.error("--artifact-attempts must be at least 1")
    args.handler(args)


if __name__ == "__main__":
    main()
