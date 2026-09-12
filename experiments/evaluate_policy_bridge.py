"""Staged Qwen3.5-4B bridge between constrained and deliberative policies.

Key Terms:
- Constrained policy: the next-token allowed-action policy, averaged over five
  predeclared menu orderings and matched by semantic action.
- Stage A: root-level agreement between the constrained policy and a
  single-action deliberative policy.
- Stage B: the conclusion-robustness benchmark. It asks whether credit
  estimators rank the same way under exact constrained credit and under
  full-solution completion rollouts.
- Arm: one state whose rollout outcomes estimate V(s), or one child state whose
  outcomes estimate Q(s,a).
- Reference pool / estimator pool: disjoint rollout sets per arm. The reference
  pool defines what estimators are scored against; the estimator pool is what
  Uniform and MC-reset consume.

Stage B is independent of Stage A: it measures estimator-level conclusions
rather than policy equality, so a failed Stage A neither blocks nor informs it.
Model imports stay local to their loaders so `--help` and the pure selection and
metric paths never import torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from analysis.conclusion_robustness_metrics import (
    ESTIMATOR_POOL_SIZE,
    K_VALUES,
    METHOD_NAMES,
    STRATA,
    ConclusionRobustnessGate,
    PairSamples,
    StateCandidate,
    bootstrap_method_fidelity,
    directional_support,
    method_credit,
    method_fidelity,
    method_rank_agreement,
    reference_split_half_stability,
    select_benchmark_pairs,
    state_occupancy,
)
from analysis.policy_bridge_metrics import (
    BridgeStageAGate,
    average_action_distributions,
    solution_preserving_probability,
    spearman_rank_correlation,
    top_actions,
    total_variation_distance,
)
from env.countdown import (
    DEFAULT_RULES,
    Action,
    Operator,
    State,
    allowed_actions,
    is_success,
    is_terminal,
    transition,
)
from oracle.evaluation import build_reasoning_dag, compute_q_star, compute_v_star
from oracle.float_policy_evaluation import FloatPolicyEvaluator
from policies.allowed_action_llm_policy import (
    MenuOption,
    PROMPT_VERSION,
    build_allowed_action_menu,
    serialize_action,
    serialize_rules,
    serialize_state,
)
from policies.countdown_completion_policy import (
    COMPLETION_PROMPT_VERSION,
    build_completion_messages,
    score_completion,
)
from policies.deliberative_action_policy import (
    DELIBERATIVE_PROMPT_VERSION,
    build_deliberative_messages,
    parse_deliberative_response,
)

MODEL_REPOSITORY = "Qwen/Qwen3.5-4B"
MENU_ORDERING_IDS = (
    "phase2-v1",
    "phase2-sensitivity-1",
    "phase2-sensitivity-2",
    "phase2-sensitivity-3",
    "phase2-sensitivity-4",
)
STAGE_A_FILE = "policy_bridge_stage_a.json"
STAGE_A_MARKDOWN = "policy_bridge_stage_a.md"
STAGE_B_FILE = "policy_bridge_stage_b.json"
STAGE_B_MARKDOWN = "policy_bridge_stage_b.md"
CALIBRATION_FILE = "policy_bridge_calibration.json"
CALIBRATION_MARKDOWN = "policy_bridge_calibration.md"
PREFLIGHT_FILE = "policy_bridge_preflight.json"
ARTIFACT_VERSION = "1.0"
# Bumped away from the complete-DAG Stage B so its artifacts cannot be resumed
# into the conclusion-robustness benchmark.
STAGE_B_SCHEMA_VERSION = "2.0-conclusion-robustness"
SELECTION_RULE = (
    "filter data/v1/test.jsonl rows with exactly four values, sort by canonical "
    "(target, sorted values, source row index), and take the first 12"
)
STATE_SELECTION_RULE = (
    "assign depths 0/1/2 cyclically by canonical root index; among states at "
    "the assigned depth with at least three legal actions take the highest "
    "constrained-policy occupancy, breaking ties by canonical state order; then "
    "take two of the top/middle/bottom constrained-advantage actions, rotating "
    "the omitted stratum by canonical root index"
)
CALIBRATION_ROOT_INDICES = (0, 5, 11)
CALIBRATION_TOKEN_CAPS = (1024, 2048, 3072)
CALIBRATION_SAMPLES_PER_CAP = 4
MAX_TRUNCATION_FRACTION = 0.25
# Ordered largest first so budget planning takes the most precise pool that
# fits. 32 and 64 were the originally predeclared sizes; the larger options
# exist because a reference advantage is a difference of two binomial means and
# 64 samples only resolve |A| >= 0.17, which is coarser than most advantages in
# these DAGs.
REFERENCE_POOL_SIZES = (256, 128, 64, 32)
BUDGET_CEILING_USD = 10.0
COST_CONTINGENCY = 0.20


@dataclass(frozen=True, slots=True)
class BridgeProblem:
    state: State
    source_row_index: int
    source_canonical_hash: str | None


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(_canonical_json(value))
    temporary.replace(path)


def select_bridge_problems(path: Path, *, count: int = 12) -> tuple[BridgeProblem, ...]:
    """Deterministically select four-number roots from the frozen test split."""
    if count < 1:
        raise ValueError("count must be at least one")
    candidates: list[BridgeProblem] = []
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            values = tuple(int(value) for value in row["values"])
            target = int(row["target"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Malformed dataset row {index}: {error}") from error
        if len(values) != 4:
            continue
        candidates.append(
            BridgeProblem(
                state=State(values=tuple(sorted(values)), target=target),
                source_row_index=index,
                source_canonical_hash=(
                    str(row["canonical_hash"]) if "canonical_hash" in row else None
                ),
            )
        )
    selected = tuple(
        sorted(
            candidates,
            key=lambda problem: (
                problem.state.target,
                problem.state.values,
                problem.source_row_index,
            ),
        )[:count]
    )
    if len(selected) != count:
        raise ValueError(
            f"Dataset provides only {len(selected)} four-number problems; need {count}"
        )
    return selected


def _action_from_dict(value: Mapping[str, object]) -> Action:
    return Action(
        Operator(str(value["operator"])),
        int(value["left"]),
        int(value["right"]),
    )


def _serialize_distribution(distribution: Mapping[Action, float]) -> list[dict[str, object]]:
    return [
        {"action": serialize_action(action), "probability": distribution[action]}
        for action in sorted(distribution, key=Action.sort_key)
    ]


def _deserialize_distribution(value: Sequence[Mapping[str, object]]) -> dict[Action, float]:
    return {
        _action_from_dict(item["action"]): float(item["probability"])
        for item in value
    }


def _serialize_menu(menu: Sequence[MenuOption]) -> list[dict[str, object]]:
    return [
        {"label": option.label, "action": serialize_action(option.action)}
        for option in menu
    ]


def _model_metadata(metadata: Any) -> dict[str, object]:
    return {
        "model_repository": metadata.model_repository,
        "model_revision": metadata.model_revision,
        "tokenizer_repository": metadata.tokenizer_repository,
        "tokenizer_revision": metadata.tokenizer_revision,
        "device": metadata.device,
        "dtype": metadata.dtype,
        "temperature": metadata.temperature,
    }


def _static_metadata(args: argparse.Namespace, problems: Sequence[BridgeProblem]) -> dict[str, object]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "execution_mode": "smoke" if getattr(args, "smoke", False) else "full",
        "model_repository": args.model,
        "constrained_temperature": args.constrained_temperature,
        "deliberative_temperature": args.deliberative_temperature,
        "samples_per_ordering": args.samples_per_ordering,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "rules": serialize_rules(DEFAULT_RULES),
        "menu_ordering_ids": list(MENU_ORDERING_IDS),
        "deliberative_prompt_version": DELIBERATIVE_PROMPT_VERSION,
        "constrained_prompt_version": PROMPT_VERSION,
        "chat_template": {
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "software_versions": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
        "dataset": str(args.dataset),
        "selection_rule": SELECTION_RULE,
        "selected_problems": [
            {
                "state": serialize_state(problem.state),
                "source_row_index": problem.source_row_index,
                "source_canonical_hash": problem.source_canonical_hash,
            }
            for problem in problems
        ],
    }


def load_stage_a_results(
    path: Path,
    *,
    expected_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read Stage A artifact: {error}") from error
    if not isinstance(result, dict) or not isinstance(result.get("metadata"), dict):
        raise ValueError("Stage A artifact must contain metadata")
    if expected_metadata is not None:
        metadata = result["metadata"]
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Stage A artifact metadata mismatch for {key!r}; stale results "
                    "must not be reused"
                )
    return result


def _response_seed(base_seed: int, state: State, ordering_id: str, sample_index: int) -> int:
    payload = json.dumps(
        {
            "seed": base_seed,
            "state": serialize_state(state),
            "ordering_id": ordering_id,
            "sample_index": sample_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big")


def _new_stage_a(static_metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "metadata": {
            **static_metadata,
            "constrained_model": None,
            "deliberative_model": None,
        },
        "selection": {"rule": SELECTION_RULE, "problems": static_metadata["selected_problems"]},
        "roots": [
            {
                "selection_index": index,
                **problem,
                "constrained_by_menu": {},
                "deliberative_responses": [],
            }
            for index, problem in enumerate(static_metadata["selected_problems"])
        ],
        "progress": {
            "completed_deliberative_responses": 0,
            "expected_deliberative_responses": (
                len(static_metadata["selected_problems"])
                * len(MENU_ORDERING_IDS)
                * int(static_metadata["samples_per_ordering"])
            ),
        },
        "aggregate": None,
    }


def _require_model(args: argparse.Namespace) -> None:
    if args.model != MODEL_REPOSITORY:
        raise ValueError(f"This experiment only supports {MODEL_REPOSITORY}")


def _load_constrained_scorer(args: argparse.Namespace) -> Any:
    from inference.huggingface_allowed_action_scorer import HuggingFaceAllowedActionScorer

    return HuggingFaceAllowedActionScorer.load(
        args.model, device=args.device, temperature=args.constrained_temperature
    )


def _load_deliberative_generator(args: argparse.Namespace) -> Any:
    from inference.huggingface_deliberative_generator import (
        HuggingFaceDeliberativeGenerator,
    )

    return HuggingFaceDeliberativeGenerator.load(
        args.model, device=args.device, temperature=args.deliberative_temperature
    )


def _score_missing_constrained(
    artifact: dict[str, object],
    artifact_path: Path,
    *,
    scorer: Any,
    rules=DEFAULT_RULES,
) -> None:
    roots = artifact["roots"]
    assert isinstance(roots, list)
    for root_result in roots:
        assert isinstance(root_result, dict)
        state_value = root_result["state"]
        state = State(
            values=tuple(state_value["values"]), target=int(state_value["target"])
        )
        constrained = root_result["constrained_by_menu"]
        assert isinstance(constrained, dict)
        for ordering_id in MENU_ORDERING_IDS:
            if ordering_id in constrained:
                continue
            record = scorer.score_states(
                [state], batch_size=1, menu_ordering_id=ordering_id, rules=rules
            )[0]
            constrained[ordering_id] = {
                "menu": _serialize_menu(record.menu),
                "distribution": _serialize_distribution(record.probabilities),
            }
            _write_json(artifact_path, artifact)


def _constrained_average(root_result: Mapping[str, object]) -> dict[Action, float]:
    by_menu = root_result["constrained_by_menu"]
    if not isinstance(by_menu, Mapping) or set(by_menu) != set(MENU_ORDERING_IDS):
        raise ValueError("Missing constrained menu-ordering distribution")
    return average_action_distributions(
        [
            _deserialize_distribution(by_menu[ordering_id]["distribution"])
            for ordering_id in MENU_ORDERING_IDS
        ]
    )


def _collect_missing_responses(
    root_result: dict[str, object],
    artifact: dict[str, object],
    artifact_path: Path,
    *,
    generator: Any,
    args: argparse.Namespace,
) -> None:
    state_value = root_result["state"]
    state = State(values=tuple(state_value["values"]), target=int(state_value["target"]))
    responses = root_result["deliberative_responses"]
    assert isinstance(responses, list)
    seen = {(row["ordering_id"], row["sample_index"]) for row in responses}
    expected = {
        (ordering_id, sample_index)
        for ordering_id in MENU_ORDERING_IDS
        for sample_index in range(args.samples_per_ordering)
    }
    if not seen <= expected or len(seen) != len(responses):
        raise ValueError("Deliberative response checkpoint is malformed or duplicated")
    constrained = root_result["constrained_by_menu"]
    assert isinstance(constrained, Mapping)
    for ordering_id, sample_index in sorted(expected - seen):
        menu = build_allowed_action_menu(
            state,
            allowed_actions(state),
            tuple(option["label"] for option in constrained[ordering_id]["menu"]),
            menu_ordering_id=ordering_id,
        )
        recorded_menu = constrained[ordering_id]["menu"]
        if _serialize_menu(menu) != recorded_menu:
            raise ValueError(
                "Deliberative menu does not exactly match its constrained menu"
            )
        seed = _response_seed(args.seed, state, ordering_id, sample_index)
        raw_response = generator.generate(
            [build_deliberative_messages(state, menu)],
            seeds=[seed],
            max_new_tokens=args.max_new_tokens,
        )[0]
        parsed = parse_deliberative_response(raw_response, menu)
        responses.append(
            {
                "ordering_id": ordering_id,
                "sample_index": sample_index,
                "seed": seed,
                "raw_response": raw_response,
                "status": parsed.status.value,
                "normalized_answer": parsed.normalized_answer,
                "action": serialize_action(parsed.action) if parsed.action else None,
            }
        )
        responses.sort(key=lambda row: (row["ordering_id"], row["sample_index"]))
        _update_stage_a_progress(artifact)
        _write_json(artifact_path, artifact)


def _update_stage_a_progress(artifact: dict[str, object]) -> None:
    roots = artifact.get("roots")
    if not isinstance(roots, list):
        return
    completed = sum(
        len(root["deliberative_responses"])
        for root in roots
        if isinstance(root, Mapping)
        and isinstance(root.get("deliberative_responses"), list)
    )
    progress = artifact.get("progress")
    if isinstance(progress, dict):
        progress["completed_deliberative_responses"] = completed


def _valid_distribution(
    responses: Sequence[Mapping[str, object]], actions: Sequence[Action]
) -> tuple[dict[Action, float] | None, float, dict[str, int]]:
    counts = {action: 0 for action in actions}
    statuses: dict[str, int] = {}
    for response in responses:
        status = str(response["status"])
        statuses[status] = statuses.get(status, 0) + 1
        if status == "valid":
            action = _action_from_dict(response["action"])
            if action not in counts:
                raise ValueError("Saved valid response has an illegal action")
            counts[action] += 1
    total = len(responses)
    valid_count = sum(counts.values())
    if total == 0:
        raise ValueError("No deliberative responses were collected")
    valid_distribution = (
        {action: count / valid_count for action, count in counts.items()}
        if valid_count
        else None
    )
    return valid_distribution, (total - valid_count) / total, statuses


def _finalize_stage_a(artifact: dict[str, object]) -> None:
    roots = artifact["roots"]
    assert isinstance(roots, list)
    valid_count = 0
    response_count = 0
    spearman_values: list[float | None] = []
    tv_values: list[float | None] = []
    for root_result in roots:
        assert isinstance(root_result, dict)
        average = _constrained_average(root_result)
        state_value = root_result["state"]
        state = State(values=tuple(state_value["values"]), target=int(state_value["target"]))
        dag = build_reasoning_dag(state)
        q_star = compute_q_star(dag, compute_v_star(dag))[state]
        deliberative, invalid_probability, statuses = _valid_distribution(
            root_result["deliberative_responses"], tuple(average)
        )
        response_count += len(root_result["deliberative_responses"])
        root_valid_count = statuses.get("valid", 0)
        valid_count += root_valid_count
        metrics: dict[str, object] = {
            "invalid_probability": invalid_probability,
            "invalid_response_frequency": invalid_probability,
            "status_counts": statuses,
            "valid_response_count": root_valid_count,
            "response_count": len(root_result["deliberative_responses"]),
            "constrained_solution_preserving_probability": solution_preserving_probability(
                average, q_star
            ),
            "deliberative_valid_conditioned_distribution": (
                _serialize_distribution(deliberative) if deliberative else None
            ),
        }
        root_result["averaged_constrained_distribution"] = _serialize_distribution(average)
        if deliberative is None:
            metrics.update(
                {
                    "tv_distance": None,
                    "spearman_rank_correlation": None,
                    "top_action_agreement": False,
                    "deliberative_solution_preserving_probability": None,
                }
            )
            spearman_values.append(None)
            tv_values.append(None)
        else:
            correlation = spearman_rank_correlation(
                [average[action] for action in average],
                [deliberative[action] for action in average],
            )
            tv = total_variation_distance(average, deliberative)
            metrics.update(
                {
                    "tv_distance": tv,
                    "spearman_rank_correlation": correlation,
                    "top_action_agreement": bool(
                        set(top_actions(average)) & set(top_actions(deliberative))
                    ),
                    "constrained_top_actions": [
                        action.render() for action in top_actions(average)
                    ],
                    "deliberative_top_actions": [
                        action.render() for action in top_actions(deliberative)
                    ],
                    "deliberative_solution_preserving_probability": (
                        solution_preserving_probability(deliberative, q_star)
                    ),
                }
            )
            spearman_values.append(correlation)
            tv_values.append(tv)
        root_result["metrics"] = metrics
    gate = BridgeStageAGate().evaluate(
        valid_rate=valid_count / response_count,
        spearman_values=spearman_values,
        tv_values=tv_values,
    )
    artifact["aggregate"] = {
        "total_response_count": response_count,
        "valid_response_count": valid_count,
        "valid_legal_response_rate": valid_count / response_count,
        "statewise_spearman": spearman_values,
        "statewise_tv_distance": tv_values,
        "gate_thresholds": asdict(BridgeStageAGate()),
        "gate": asdict(gate),
        "credit_interpretation": (
            "Constrained results are exact dynamic-programming credit conditional "
            "on next-token distributions; deliberative root distributions are "
            "sampled and condition on valid legal responses."
        ),
    }


def stage_a_markdown(artifact: Mapping[str, object]) -> str:
    aggregate = artifact["aggregate"]
    assert isinstance(aggregate, Mapping)
    gate = aggregate["gate"]
    return "\n".join(
        [
            "# Policy Bridge — Stage A",
            "",
            "Compares the menu-order-averaged constrained next-token policy with "
            "the sampled state-only deliberative policy at selected roots.",
            "",
            f"- Roots: `{len(artifact['roots'])}`",
            f"- Valid legal deliberative response rate: `{aggregate['valid_legal_response_rate']:.3%}`",
            f"- Median Spearman: `{gate['median_spearman']}`",
            f"- Median TV: `{gate['median_tv']}`",
            f"- Gate passed: `{gate['passed']}`",
            "",
            "Constrained-policy credit is exact conditional on cached/recorded "
            "next-token probabilities. Deliberative policy measurements are "
            "sampled; invalid responses are reported separately and are excluded "
            "only from this Stage A action-distribution comparison.",
            "",
        ]
    )


def run_stage_a(args: argparse.Namespace) -> None:
    _require_model(args)
    _validate_stage_contract(args, stage="stage-a")
    problems = select_bridge_problems(Path(args.dataset), count=args.problem_limit)
    static_metadata = _static_metadata(args, problems)
    output_dir = Path(args.output)
    artifact_path = output_dir / STAGE_A_FILE
    artifact = (
        load_stage_a_results(artifact_path, expected_metadata=static_metadata)
        if artifact_path.exists()
        else _new_stage_a(static_metadata)
    )
    _write_json(artifact_path, artifact)

    scorer = _load_constrained_scorer(args)
    constrained_metadata = _model_metadata(scorer.metadata)
    existing = artifact["metadata"].get("constrained_model")
    if existing not in (None, constrained_metadata):
        raise ValueError("Stage A constrained model metadata mismatch")
    artifact["metadata"]["constrained_model"] = constrained_metadata
    _write_json(artifact_path, artifact)
    _score_missing_constrained(artifact, artifact_path, scorer=scorer)
    del scorer

    generator = _load_deliberative_generator(args)
    deliberative_metadata = _model_metadata(generator.metadata)
    existing = artifact["metadata"].get("deliberative_model")
    if existing not in (None, deliberative_metadata):
        raise ValueError("Stage A deliberative model metadata mismatch")
    artifact["metadata"]["deliberative_model"] = deliberative_metadata
    _write_json(artifact_path, artifact)
    for root_result in artifact["roots"]:
        _collect_missing_responses(
            root_result, artifact, artifact_path, generator=generator, args=args
        )
    _finalize_stage_a(artifact)
    _write_json(artifact_path, artifact)
    (output_dir / STAGE_A_MARKDOWN).write_text(stage_a_markdown(artifact))
    print(f"Wrote {artifact_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("stage-a", "Run or resume root-level policy agreement."),
        (
            "calibrate",
            "Measure completion length locally to size the paid token cap.",
        ),
        (
            "preflight",
            "Measure paid-GPU throughput at the production cap and batch size.",
        ),
        (
            "stage-b",
            "Run or resume the conclusion-robustness estimator benchmark.",
        ),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--output", required=True)
        subparser.add_argument("--model", default=MODEL_REPOSITORY)
        subparser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        subparser.add_argument("--dataset", default="data/v1/test.jsonl")
        subparser.add_argument("--seed", type=int, default=20260904)
        subparser.add_argument(
            "--smoke",
            action="store_true",
            help="Permit reduced sample counts for a non-gating smoke artifact.",
        )
        if name == "stage-a":
            subparser.add_argument("--constrained-temperature", type=float, default=1.0)
            subparser.add_argument("--deliberative-temperature", type=float, default=0.6)
            subparser.add_argument("--samples-per-ordering", type=int, default=32)
            subparser.add_argument("--max-new-tokens", type=int, default=1024)
            subparser.add_argument("--problem-limit", type=int, default=12)
            subparser.set_defaults(handler=run_stage_a)
        elif name == "calibrate":
            subparser.add_argument("--completion-temperature", type=float, default=0.6)
            subparser.set_defaults(handler=run_calibrate)
        elif name == "preflight":
            subparser.add_argument("--completion-temperature", type=float, default=0.6)
            subparser.add_argument("--token-cap", type=int, required=True)
            subparser.add_argument(
                "--batch-sizes", type=int, nargs="+", default=[8, 16, 32]
            )
            subparser.add_argument("--batches", type=int, default=3)
            subparser.add_argument("--generating-arm-count", type=int, default=36)
            subparser.add_argument("--gpu-hourly-usd", type=float, required=True)
            subparser.add_argument(
                "--budget-ceiling-usd", type=float, default=BUDGET_CEILING_USD
            )
            subparser.set_defaults(handler=run_preflight)
        else:
            subparser.add_argument("--constrained-temperature", type=float, default=1.0)
            subparser.add_argument("--completion-temperature", type=float, default=0.6)
            subparser.add_argument("--problem-limit", type=int, default=12)
            subparser.add_argument(
                "--token-cap",
                type=int,
                required=True,
                help="Token allowance selected by the calibrate command.",
            )
            subparser.add_argument("--batch-size", type=int, default=8)
            subparser.add_argument(
                "--reference-pool-size",
                type=int,
                required=True,
                choices=REFERENCE_POOL_SIZES,
                help="Held-out reference samples per arm, chosen by --plan-budget.",
            )
            subparser.add_argument("--bootstrap-replicates", type=int, default=1000)
            subparser.add_argument("--bootstrap-seed", type=int, default=20260905)
            subparser.set_defaults(handler=run_stage_b)

    budget = subparsers.add_parser(
        "plan-budget",
        help="Project Stage B spend and pick the affordable reference pool size.",
    )
    budget.add_argument(
        "--seconds-per-response",
        type=float,
        required=True,
        help="Wall time per response measured by the paid RunPod preflight.",
    )
    budget.add_argument("--gpu-hourly-usd", type=float, required=True)
    budget.add_argument("--generating-arm-count", type=int, default=36)
    budget.add_argument(
        "--budget-ceiling-usd",
        type=float,
        default=BUDGET_CEILING_USD,
        help="Overrides the predeclared ceiling; record any override in the spec.",
    )
    budget.set_defaults(handler=run_plan_budget)
    return parser


def run_plan_budget(args: argparse.Namespace) -> None:
    """Report the largest reference pool that stays inside the stated ceiling."""
    chosen, projections = choose_reference_pool_size(
        generating_arm_count=args.generating_arm_count,
        seconds_per_response=args.seconds_per_response,
        gpu_hourly_usd=args.gpu_hourly_usd,
        budget_ceiling_usd=args.budget_ceiling_usd,
    )
    for projection in projections:
        print(
            f"reference_pool_size={projection['reference_pool_size']} "
            f"responses={projection['response_count']} "
            f"projected_usd_with_contingency="
            f"{projection['projected_usd_with_contingency']:.2f} "
            f"affordable={projection['affordable']}"
        )
    if chosen is None:
        raise ValueError(
            f"No reference pool size fits the ${args.budget_ceiling_usd:.2f} "
            "ceiling; refuse the full run rather than overspending"
        )
    print(f"Use --reference-pool-size {chosen}")


def _validate_stage_contract(args: argparse.Namespace, *, stage: str) -> None:
    if getattr(args, "smoke", False):
        return
    if stage == "stage-a":
        if args.samples_per_ordering != 32:
            raise ValueError(
                "Full Stage A requires exactly 32 responses per menu ordering; use "
                "--smoke only for a non-gating reduced run"
            )
        if args.problem_limit != 12:
            raise ValueError(
                "Full Stage A requires exactly 12 roots; use --smoke only for a "
                "non-gating reduced run"
            )
    if stage == "stage-b":
        if args.problem_limit != 12:
            raise ValueError(
                "Full Stage B requires exactly 12 roots; use --smoke only for a "
                "non-gating reduced run"
            )
        if args.bootstrap_replicates != 1000:
            raise ValueError(
                "Full Stage B requires exactly 1,000 bootstrap replicates; use "
                "--smoke only for a non-gating reduced run"
            )


class _MappingPolicy:
    def __init__(self, probabilities: Mapping[State, Mapping[Action, float]]) -> None:
        self._probabilities = probabilities

    def action_probs(
        self, state: State, actions: tuple[Action, ...]
    ) -> Mapping[Action, float]:
        probabilities = self._probabilities[state]
        if set(probabilities) != set(actions):
            raise ValueError("State action probabilities do not match legal actions")
        return probabilities


def _bootstrap_quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": ordered[len(ordered) // 2],
        "p025": ordered[round((len(ordered) - 1) * 0.025)],
        "p975": ordered[round((len(ordered) - 1) * 0.975)],
    }


def _arm_identifier(root_index: int, role: str, action: Action | None) -> str:
    """Name each of the 36 arms so its two seed ranges are independent.

    Arms are keyed by their role in the benchmark rather than by state, so two
    arms that happen to share a state still draw independent completions.
    """
    if role == "state":
        return f"root{root_index:02d}/state"
    if role == "child":
        assert action is not None
        return f"root{root_index:02d}/child/{action.render()}"
    raise ValueError(f"Unknown arm role: {role!r}")


def _completion_seed(base_seed: int, arm_id: str, pool: str, batch_index: int) -> int:
    payload = json.dumps(
        {
            "seed": base_seed,
            "arm_id": arm_id,
            "pool": pool,
            "batch_index": batch_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return int.from_bytes(
        hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big"
    )


def _deterministic_outcome(state: State) -> int | None:
    """Return a fixed rollout outcome when no generation can change it."""
    if is_success(state):
        return 1
    if is_terminal(state):
        return 0
    return None


def _root_selection_candidates(
    root: State,
    constrained_by_state: Mapping[State, Mapping[Action, float]],
    dag: Any,
) -> tuple[FloatPolicyEvaluator, tuple[StateCandidate, ...]]:
    """Derive occupancy and exact advantages for every nonterminal DAG state."""
    evaluator = FloatPolicyEvaluator(_MappingPolicy(constrained_by_state), dag)
    order = dag.topological_order()
    nonterminal = [state for state in order if not is_terminal(state)]
    occupancy = state_occupancy(
        topological_states=order,
        out_edges={
            state: [
                (edge.action.render(), edge.target_state)
                for edge in dag.out_edges(state)
            ]
            for state in nonterminal
        },
        action_probabilities={
            state: {
                action.render(): probability
                for action, probability in constrained_by_state[state].items()
            }
            for state in nonterminal
        },
        root=root,
    )
    candidates = tuple(
        StateCandidate(
            canonical_key=(state.target, state.values),
            depth=len(root.values) - len(state.values),
            occupancy=occupancy.get(state, 0.0),
            advantage_by_action={
                action.render(): evaluator.a_pi(state, action)
                for action in allowed_actions(state)
            },
        )
        for state in nonterminal
    )
    return evaluator, candidates


def choose_token_cap(
    truncation_fraction_by_cap: Mapping[int, float],
    *,
    maximum_truncation_fraction: float = MAX_TRUNCATION_FRACTION,
) -> int | None:
    """Smallest measured cap where few enough responses used the whole allowance.

    Returns None when every cap systematically truncates, which means the
    completion policy was not measurable within the predeclared range.
    """
    if not truncation_fraction_by_cap:
        raise ValueError("At least one token cap must be measured")
    for cap in sorted(truncation_fraction_by_cap):
        if truncation_fraction_by_cap[cap] <= maximum_truncation_fraction:
            return cap
    return None


def project_stage_b_cost(
    *,
    response_count: int,
    seconds_per_response: float,
    gpu_hourly_usd: float,
    contingency: float = COST_CONTINGENCY,
) -> dict[str, float | int]:
    """Spend implied by measured wall time per response.

    Cost is wall time, so the basis is seconds per response observed at the
    production batch size. That measurement already carries both the real token
    distribution and the fact that a batch runs until its longest member
    finishes. Projecting instead from a token cap and a token rate would apply
    the worst-case token count on top of a rate that already reflects realistic
    lengths, overstating the cost roughly twofold.
    """
    if response_count < 1:
        raise ValueError("response_count must be at least one")
    if seconds_per_response <= 0.0:
        raise ValueError("seconds_per_response must be positive")
    if gpu_hourly_usd <= 0.0:
        raise ValueError("gpu_hourly_usd must be positive")
    hours = response_count * seconds_per_response / 3600.0
    return {
        "response_count": response_count,
        "seconds_per_response": seconds_per_response,
        "projected_gpu_hours": hours,
        "projected_usd": hours * gpu_hourly_usd,
        "projected_usd_with_contingency": hours * gpu_hourly_usd * (1.0 + contingency),
    }


def choose_reference_pool_size(
    *,
    generating_arm_count: int,
    seconds_per_response: float,
    gpu_hourly_usd: float,
    budget_ceiling_usd: float = BUDGET_CEILING_USD,
    candidate_sizes: Sequence[int] = REFERENCE_POOL_SIZES,
) -> tuple[int | None, list[dict[str, object]]]:
    """Largest affordable reference pool, chosen from cost alone.

    Returns ``(None, projections)`` when even the smallest candidate exceeds the
    ceiling, so the caller refuses the run instead of overspending.
    """
    projections: list[dict[str, object]] = []
    chosen: int | None = None
    for size in candidate_sizes:
        projection = project_stage_b_cost(
            response_count=generating_arm_count * (size + ESTIMATOR_POOL_SIZE),
            seconds_per_response=seconds_per_response,
            gpu_hourly_usd=gpu_hourly_usd,
        )
        affordable = (
            float(projection["projected_usd_with_contingency"]) <= budget_ceiling_usd
        )
        projections.append(
            {"reference_pool_size": size, "affordable": affordable, **projection}
        )
        if affordable and chosen is None:
            chosen = size
    return chosen, projections


def _simulate_constrained_pool(
    success_probability: float, *, size: int, seed: int
) -> tuple[int, ...]:
    """Draw estimator outcomes from the exact constrained policy's value.

    A rollout under the constrained policy succeeds with probability V-pi, so
    simulating Bernoulli draws needs no model inference and keeps the two setups
    subject to the same estimator sampling noise.
    """
    rng = random.Random(seed)
    return tuple(1 if rng.random() < success_probability else 0 for _ in range(size))


def _pool_outcomes(arm: Mapping[str, object], pool: str) -> tuple[int, ...]:
    deterministic = arm.get("deterministic_outcome")
    if deterministic is not None:
        size = (
            ESTIMATOR_POOL_SIZE
            if pool == "estimator"
            else int(arm["reference_pool_size"])
        )
        return (int(deterministic),) * size
    return tuple(
        1 if row["success"] else 0
        for row in arm["responses"]
        if row["pool"] == pool
    )


def _completion_rates(arms: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = [row for arm in arms for row in arm["responses"]]
    if not rows:
        return {"response_count": 0}
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    total = len(rows)
    return {
        "response_count": total,
        "status_counts": status_counts,
        "success_rate": sum(1 for row in rows if row["success"]) / total,
        "answer_tag_rate": sum(
            1 for row in rows if row["status"] != "missing_answer"
        )
        / total,
        "truncation_rate": sum(1 for row in rows if row["is_truncated"]) / total,
        "generated_token_count": sum(int(row["generated_token_count"]) for row in rows),
        "mean_generated_token_count": math.fsum(
            int(row["generated_token_count"]) for row in rows
        )
        / total,
    }


def _subgroup_fidelity(
    pairs: Sequence[PairSamples],
    labels: Sequence[str],
) -> dict[str, dict[str, float | None]]:
    """Point fidelity per method within each label group, without bootstrapping."""
    grouped: dict[str, list[PairSamples]] = {}
    for label, pair in zip(labels, pairs, strict=True):
        grouped.setdefault(label, []).append(pair)
    return {
        label: {
            method: method_fidelity(group, method) for method in METHOD_NAMES
        }
        for label, group in sorted(grouped.items())
        if len(group) >= 2
    }


def _sign_accuracy(
    pairs: Sequence[PairSamples],
    reference_intervals: Sequence[tuple[float, float]],
) -> dict[str, object]:
    """Per-method sign accuracy over pairs whose reference excludes zero."""
    eligible = [
        pair
        for pair, (low, high) in zip(pairs, reference_intervals, strict=True)
        if low > 0.0 or high < 0.0
    ]
    if not eligible:
        return {"eligible_pair_count": 0}
    accuracy: dict[str, float] = {}
    for method in METHOD_NAMES:
        matching = 0
        for pair in eligible:
            credit = method_credit(method, pair.child_outcomes, pair.state_outcomes)
            reference = pair.reference_advantage
            matching += (credit > 0.0 and reference > 0.0) or (
                credit <= 0.0 and reference < 0.0
            )
        accuracy[method] = matching / len(eligible)
    return {"eligible_pair_count": len(eligible), "sign_accuracy": accuracy}


def _reference_interval(
    child: Sequence[int], state: Sequence[int], *, replicates: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = sorted(
        math.fsum(rng.choices(child, k=len(child))) / len(child)
        - math.fsum(rng.choices(state, k=len(state))) / len(state)
        for _ in range(replicates)
    )
    return (
        draws[round((len(draws) - 1) * 0.025)],
        draws[round((len(draws) - 1) * 0.975)],
    )


def _finalize_stage_b(artifact: dict[str, object], args: argparse.Namespace) -> None:
    """Score both setups on the same 24 pairs and apply the three-way gate.

    Steps:
    1. Build per-pair estimator pools and reference advantages for both setups.
    2. Bootstrap method fidelity in each setup.
    3. Measure whether the sampled deliberative reference can rank at all.
    4. Compare method-level rankings across setups and decide.
    """
    arms_by_id = {str(arm["arm_id"]): arm for arm in artifact["arms"]}

    # Step 1: assemble aligned pair vectors for the constrained and rollout setups.
    constrained_pairs: list[PairSamples] = []
    deliberative_pairs: list[PairSamples] = []
    reference_pools: list[tuple[Sequence[int], Sequence[int]]] = []
    reference_intervals: list[tuple[float, float]] = []
    depth_labels: list[str] = []
    stratum_labels: list[str] = []
    for pair in artifact["pairs"]:
        state_arm = arms_by_id[str(pair["state_arm_id"])]
        child_arm = arms_by_id[str(pair["child_arm_id"])]

        child_estimator = _pool_outcomes(child_arm, "estimator")
        state_estimator = _pool_outcomes(state_arm, "estimator")
        child_reference = _pool_outcomes(child_arm, "reference")
        state_reference = _pool_outcomes(state_arm, "reference")
        sampled_reference = math.fsum(child_reference) / len(
            child_reference
        ) - math.fsum(state_reference) / len(state_reference)
        deliberative_pairs.append(
            PairSamples(
                child_outcomes=child_estimator,
                state_outcomes=state_estimator,
                reference_advantage=sampled_reference,
            )
        )
        reference_pools.append((child_reference, state_reference))
        reference_intervals.append(
            _reference_interval(
                child_reference,
                state_reference,
                replicates=args.bootstrap_replicates,
                seed=_completion_seed(
                    args.bootstrap_seed, str(pair["child_arm_id"]), "reference-ci", 0
                ),
            )
        )

        constrained_pairs.append(
            PairSamples(
                child_outcomes=_simulate_constrained_pool(
                    float(pair["constrained_q_pi"]),
                    size=ESTIMATOR_POOL_SIZE,
                    seed=_completion_seed(
                        args.bootstrap_seed,
                        str(pair["child_arm_id"]),
                        "constrained-simulation",
                        0,
                    ),
                ),
                state_outcomes=_simulate_constrained_pool(
                    float(pair["constrained_v_pi"]),
                    size=ESTIMATOR_POOL_SIZE,
                    seed=_completion_seed(
                        args.bootstrap_seed,
                        str(pair["state_arm_id"]),
                        "constrained-simulation",
                        0,
                    ),
                ),
                reference_advantage=float(pair["constrained_advantage"]),
            )
        )
        pair["sampled_deliberative_advantage"] = sampled_reference
        pair["sampled_deliberative_interval"] = list(reference_intervals[-1])
        depth_labels.append(f"depth{int(pair['depth'])}")
        stratum_labels.append(str(pair["stratum"]))

    # Step 2: bootstrap estimator draws in both setups.
    bootstrap_by_setup = {
        "constrained": bootstrap_method_fidelity(
            constrained_pairs,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        ),
        "deliberative": bootstrap_method_fidelity(
            deliberative_pairs,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 1,
        ),
    }
    median_fidelity = {
        setup: {
            method: float(median(values)) if values else None
            for method, values in bootstrap.by_method.items()
        }
        for setup, bootstrap in bootstrap_by_setup.items()
    }
    support_by_setup = {
        setup: directional_support(bootstrap)
        for setup, bootstrap in bootstrap_by_setup.items()
    }

    # Step 3: a reference that cannot rank actions makes everything else moot.
    stability = reference_split_half_stability(
        reference_pools,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 2,
    )

    # Step 4: compare method-level rankings and decide.
    incomplete = [
        setup
        for setup, fidelity in median_fidelity.items()
        if any(value is None for value in fidelity.values())
    ]
    agreement = (
        None
        if incomplete
        else method_rank_agreement(
            median_fidelity["constrained"], median_fidelity["deliberative"]
        )
    )
    gate = ConclusionRobustnessGate()
    decision = gate.evaluate(
        reference_stability=stability,
        constrained_support=support_by_setup["constrained"],
        deliberative_support=support_by_setup["deliberative"],
        method_rank_agreement=agreement,
    )

    artifact["aggregate"] = {
        "pair_count": len(deliberative_pairs),
        "median_fidelity": median_fidelity,
        "bootstrap_fidelity_quantiles": {
            setup: {
                method: _bootstrap_quantiles(values) if values else None
                for method, values in bootstrap.by_method.items()
            }
            for setup, bootstrap in bootstrap_by_setup.items()
        },
        "dropped_bootstrap_replicates": {
            setup: bootstrap.dropped_replicate_count
            for setup, bootstrap in bootstrap_by_setup.items()
        },
        "directional_support": {
            setup: asdict(support) for setup, support in support_by_setup.items()
        },
        "reference_split_half_stability": stability,
        "method_rank_agreement": agreement,
        "fidelity_by_depth": {
            "constrained": _subgroup_fidelity(constrained_pairs, depth_labels),
            "deliberative": _subgroup_fidelity(deliberative_pairs, depth_labels),
        },
        "fidelity_by_stratum": {
            "constrained": _subgroup_fidelity(constrained_pairs, stratum_labels),
            "deliberative": _subgroup_fidelity(deliberative_pairs, stratum_labels),
        },
        "sign_accuracy": {
            "constrained": _sign_accuracy(constrained_pairs, reference_intervals),
            "deliberative": _sign_accuracy(deliberative_pairs, reference_intervals),
        },
        "completion_rates": _completion_rates(artifact["arms"]),
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
        "gate_thresholds": asdict(gate),
        "gate": asdict(decision),
        "credit_interpretation": (
            "Constrained-side reference values are exact A-pi conditional on the "
            "recorded next-token probabilities, with estimator noise simulated "
            "from V-pi. Deliberative-side reference values are sampled from a "
            "held-out rollout pool and carry their own uncertainty, so this "
            "benchmark validates estimator rankings rather than credit equality."
        ),
    }


def stage_b_markdown(artifact: Mapping[str, object]) -> str:
    aggregate = artifact["aggregate"]
    assert isinstance(aggregate, Mapping)
    gate = aggregate["gate"]
    median_fidelity = aggregate["median_fidelity"]
    lines = [
        "# Policy Bridge — Stage B (conclusion robustness)",
        "",
        "Asks whether the ranking of credit estimators survives replacing the "
        "constrained next-token policy with full-solution Qwen completions.",
        "",
        f"- Benchmark state-action pairs: `{aggregate['pair_count']}`",
        f"- Reference split-half stability: `{aggregate['reference_split_half_stability']}`",
        f"- Method-level rank agreement across setups: `{aggregate['method_rank_agreement']}`",
        f"- Decision: `{gate['decision']}`",
        "",
        "## Median fidelity by method",
        "",
        "| Method | Constrained | Deliberative |",
        "| --- | --- | --- |",
    ]
    for method in METHOD_NAMES:
        lines.append(
            f"| `{method}` | `{median_fidelity['constrained'][method]}` "
            f"| `{median_fidelity['deliberative'][method]}` |"
        )
    lines += ["", "## Directional bootstrap support", ""]
    for setup, support in aggregate["directional_support"].items():
        lines.append(
            f"- {setup}: K=16 beats Uniform in `{support['k16_beats_uniform']:.1%}` "
            f"of replicates; fidelity rises with K in "
            f"`{support['positive_log2k_slope']:.1%}`"
        )
    if gate["reasons"]:
        lines += ["", "## Why the decision is not GO", ""]
        lines += [f"- {reason}" for reason in gate["reasons"]]
    lines += [
        "",
        "Constrained-side references are exact A-pi; deliberative-side references "
        "are sampled from a held-out rollout pool and carry uncertainty. An "
        "unstable sampled reference yields `INCONCLUSIVE` rather than `NO-GO`, "
        "because that is evidence about sample size and not about disagreement.",
        "",
    ]
    return "\n".join(lines)


def calibration_markdown(artifact: Mapping[str, object]) -> str:
    selected = artifact["selected_token_cap"]
    lines = [
        "# Policy Bridge — completion calibration",
        "",
        "Measures how many tokens project-rule full-solution completions actually "
        "need, so the paid benchmark can pick the smallest non-truncating cap.",
        "",
        "| Token cap | Responses | Truncated | Answer tag | Solved | Mean tokens |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for measurement in artifact["measurements"]:
        lines.append(
            f"| `{measurement['token_cap']}` | `{measurement['response_count']}` "
            f"| `{measurement['truncation_rate']:.1%}` "
            f"| `{measurement['answer_tag_rate']:.1%}` "
            f"| `{measurement['success_rate']:.1%}` "
            f"| `{measurement['mean_generated_token_count']:.0f}` |"
        )
    lines += [
        "",
        f"- Maximum acceptable truncation rate: `{MAX_TRUNCATION_FRACTION:.0%}`",
        f"- Selected token cap: `{selected}`",
        f"- Local throughput (tokens/second): `{artifact['tokens_per_second']:.2f}`",
        "",
    ]
    if selected is None:
        lines += [
            (
                "No measured cap kept truncation at or below the predeclared "
                "limit, so the completion policy is not measurable within the "
                "predeclared range. Do not launch paid inference."
            ),
            "",
        ]
    return "\n".join(lines)


def _completion_metadata(args: argparse.Namespace) -> dict[str, object]:
    """Everything a resumed run must match exactly before reusing responses."""
    return {
        "artifact_version": ARTIFACT_VERSION,
        "schema_version": STAGE_B_SCHEMA_VERSION,
        "benchmark": "conclusion-robustness",
        "model_repository": args.model,
        "constrained_temperature": args.constrained_temperature,
        "completion_temperature": args.completion_temperature,
        "token_cap": args.token_cap,
        "batch_size": args.batch_size,
        "reference_pool_size": args.reference_pool_size,
        "estimator_pool_size": ESTIMATOR_POOL_SIZE,
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "rules": serialize_rules(DEFAULT_RULES),
        "menu_ordering_ids": list(MENU_ORDERING_IDS),
        "constrained_prompt_version": PROMPT_VERSION,
        "completion_prompt_version": COMPLETION_PROMPT_VERSION,
        "chat_template": {"add_generation_prompt": True, "enable_thinking": False},
        "software_versions": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
        "dataset": str(args.dataset),
        "root_selection_rule": SELECTION_RULE,
        "state_selection_rule": STATE_SELECTION_RULE,
        "k_values": list(K_VALUES),
        "strata": list(STRATA),
    }


def _load_completion_generator(args: argparse.Namespace) -> Any:
    from inference.huggingface_completion_generator import (
        HuggingFaceCompletionGenerator,
    )

    return HuggingFaceCompletionGenerator.load(
        args.model, device=args.device, temperature=args.completion_temperature
    )


def run_calibrate(args: argparse.Namespace) -> None:
    """Measure completion length locally so the paid run can size its token cap."""
    _require_model(args)
    problems = select_bridge_problems(Path(args.dataset), count=12)
    states = [problems[index].state for index in CALIBRATION_ROOT_INDICES]
    generator = _load_completion_generator(args)
    measurements: list[dict[str, object]] = []
    truncation_by_cap: dict[int, float] = {}
    total_tokens = 0
    total_seconds = 0.0
    for token_cap in CALIBRATION_TOKEN_CAPS:
        rows: list[dict[str, object]] = []
        for root_index, state in zip(CALIBRATION_ROOT_INDICES, states, strict=True):
            arm_id = _arm_identifier(root_index, "state", None)
            started = time.monotonic()
            completions = generator.generate_batch(
                build_completion_messages(state),
                count=CALIBRATION_SAMPLES_PER_CAP,
                seed=_completion_seed(args.seed, arm_id, f"calibrate-{token_cap}", 0),
                max_new_tokens=token_cap,
            )
            total_seconds += time.monotonic() - started
            for sample_index, completion in enumerate(completions):
                scored = score_completion(completion.text, state)
                total_tokens += completion.generated_token_count
                rows.append(
                    {
                        "root_index": root_index,
                        "state": serialize_state(state),
                        "sample_index": sample_index,
                        "status": scored.status.value,
                        "success": scored.is_success,
                        "expression": scored.expression,
                        "generated_token_count": completion.generated_token_count,
                        "is_truncated": completion.is_truncated,
                        "raw_response": completion.text,
                    }
                )
        truncation_rate = sum(1 for row in rows if row["is_truncated"]) / len(rows)
        truncation_by_cap[token_cap] = truncation_rate
        measurements.append(
            {
                "token_cap": token_cap,
                "response_count": len(rows),
                "truncation_rate": truncation_rate,
                "answer_tag_rate": sum(
                    1 for row in rows if row["status"] != "missing_answer"
                )
                / len(rows),
                "success_rate": sum(1 for row in rows if row["success"]) / len(rows),
                "mean_generated_token_count": math.fsum(
                    int(row["generated_token_count"]) for row in rows
                )
                / len(rows),
                "responses": rows,
            }
        )

    artifact = {
        "metadata": {
            "artifact_version": ARTIFACT_VERSION,
            "model_repository": args.model,
            "completion_model": _model_metadata(generator.metadata),
            "completion_prompt_version": COMPLETION_PROMPT_VERSION,
            "rules": serialize_rules(DEFAULT_RULES),
            "seed": args.seed,
            "token_caps": list(CALIBRATION_TOKEN_CAPS),
            "samples_per_cap_per_state": CALIBRATION_SAMPLES_PER_CAP,
            "calibration_root_indices": list(CALIBRATION_ROOT_INDICES),
            "maximum_truncation_fraction": MAX_TRUNCATION_FRACTION,
            "software_versions": {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
            },
        },
        "measurements": measurements,
        "selected_token_cap": choose_token_cap(truncation_by_cap),
        "tokens_per_second": total_tokens / total_seconds if total_seconds else 0.0,
    }
    output_dir = Path(args.output)
    path = output_dir / CALIBRATION_FILE
    _write_json(path, artifact)
    (output_dir / CALIBRATION_MARKDOWN).write_text(calibration_markdown(artifact))
    print(f"Wrote {path}")
    if artifact["selected_token_cap"] is None:
        raise ValueError(
            "No token cap kept truncation within the predeclared limit; the "
            "completion policy is not measurable in this range and paid "
            "inference must not be launched"
        )


def run_preflight(args: argparse.Namespace) -> None:
    """Measure paid-GPU throughput at the production cap and batch size.

    Throughput depends strongly on batch size, so the calibration number from a
    batch of four cannot be reused to project the cost of production batches.
    """
    _require_model(args)
    problems = select_bridge_problems(Path(args.dataset), count=12)
    states = [problems[index].state for index in CALIBRATION_ROOT_INDICES]
    generator = _load_completion_generator(args)

    # Sweep batch size because decoding a 4B model is memory-bandwidth bound at
    # small batches, so wall time per response can fall sharply as batch grows.
    measurements: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        batches: list[dict[str, object]] = []
        failure: str | None = None
        for batch_index in range(args.batches):
            state = states[batch_index % len(states)]
            arm_id = _arm_identifier(batch_index, "state", None)
            started = time.monotonic()
            try:
                completions = generator.generate_batch(
                    build_completion_messages(state),
                    count=batch_size,
                    seed=_completion_seed(
                        args.seed, arm_id, f"preflight-{batch_size}", batch_index
                    ),
                    max_new_tokens=args.token_cap,
                )
            except Exception as error:  # noqa: BLE001 - out of memory is expected
                failure = f"{type(error).__name__}: {error}"
                break
            elapsed = time.monotonic() - started
            batches.append(
                {
                    "batch_index": batch_index,
                    "state": serialize_state(state),
                    "seconds": elapsed,
                    "generated_token_count": sum(
                        completion.generated_token_count
                        for completion in completions
                    ),
                    "truncation_count": sum(
                        1 for completion in completions if completion.is_truncated
                    ),
                    "success_count": sum(
                        1
                        for completion in completions
                        if score_completion(completion.text, state).is_success
                    ),
                }
            )
        if failure is not None or not batches:
            measurements.append({"batch_size": batch_size, "failure": failure})
            print(f"batch_size={batch_size} failed: {failure}")
            continue
        seconds = math.fsum(float(batch["seconds"]) for batch in batches)
        responses = len(batches) * batch_size
        tokens = sum(int(batch["generated_token_count"]) for batch in batches)
        measurements.append(
            {
                "batch_size": batch_size,
                "failure": None,
                "batches": batches,
                "response_count": responses,
                "seconds": seconds,
                "seconds_per_response": seconds / responses,
                "tokens_per_second": tokens / seconds if seconds else 0.0,
                "mean_generated_token_count": tokens / responses,
                "truncation_rate": sum(
                    int(batch["truncation_count"]) for batch in batches
                )
                / responses,
                "success_rate": sum(int(batch["success_count"]) for batch in batches)
                / responses,
            }
        )
        print(
            f"batch_size={batch_size}: {seconds / responses:.2f} s/response, "
            f"{tokens / seconds:.1f} tok/s, "
            f"mean_tokens={tokens / responses:.0f}"
        )

    usable = [row for row in measurements if row.get("failure") is None]
    if not usable:
        raise ValueError("Every preflight batch size failed; cannot project cost")
    best = min(usable, key=lambda row: float(row["seconds_per_response"]))
    chosen, projections = choose_reference_pool_size(
        generating_arm_count=args.generating_arm_count,
        seconds_per_response=float(best["seconds_per_response"]),
        gpu_hourly_usd=args.gpu_hourly_usd,
        budget_ceiling_usd=args.budget_ceiling_usd,
    )
    artifact = {
        "metadata": {
            "artifact_version": ARTIFACT_VERSION,
            "model_repository": args.model,
            "completion_model": _model_metadata(generator.metadata),
            "completion_prompt_version": COMPLETION_PROMPT_VERSION,
            "token_cap": args.token_cap,
            "batch_sizes": list(args.batch_sizes),
            "batches_per_batch_size": args.batches,
            "seed": args.seed,
            "gpu_hourly_usd": args.gpu_hourly_usd,
            "budget_ceiling_usd": args.budget_ceiling_usd,
            "software_versions": {
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
            },
        },
        "measurements": measurements,
        "recommended_batch_size": best["batch_size"],
        "seconds_per_response": best["seconds_per_response"],
        "budget_projections": projections,
        "affordable_reference_pool_size": chosen,
    }
    path = Path(args.output) / PREFLIGHT_FILE
    _write_json(path, artifact)
    print(f"\nWrote {path}")
    print(
        f"Best batch size {best['batch_size']} at "
        f"{float(best['seconds_per_response']):.2f} s/response"
    )
    for projection in projections:
        print(
            f"  reference_pool_size={projection['reference_pool_size']} "
            f"responses={projection['response_count']} "
            f"gpu_hours={float(projection['projected_gpu_hours']):.1f} "
            f"usd_with_contingency="
            f"{projection['projected_usd_with_contingency']:.2f} "
            f"affordable={projection['affordable']}"
        )
    if chosen is None:
        raise ValueError(
            f"No reference pool size fits the ${args.budget_ceiling_usd:.2f} "
            "ceiling at the observed throughput; refuse the full run"
        )
    print(
        f"Largest affordable: --reference-pool-size {chosen} "
        f"--batch-size {best['batch_size']}"
    )


def _new_stage_b(
    metadata: Mapping[str, object],
    selections: Sequence[Mapping[str, object]],
    arms: Sequence[Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "metadata": {**metadata, "constrained_model": None, "completion_model": None},
        "selection": {
            "root_rule": SELECTION_RULE,
            "state_rule": STATE_SELECTION_RULE,
            "roots": list(selections),
        },
        "arms": list(arms),
        "pairs": list(pairs),
        "progress": {
            "completed_responses": 0,
            "expected_responses": sum(
                0
                if arm["deterministic_outcome"] is not None
                else int(metadata["reference_pool_size"]) + ESTIMATOR_POOL_SIZE
                for arm in arms
            ),
        },
        "aggregate": None,
    }


def _build_stage_b_selection(
    args: argparse.Namespace,
    problems: Sequence[BridgeProblem],
    *,
    scorer: Any,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Score the constrained policy over all 12 DAGs, then choose the 24 pairs.

    Steps:
    1. Score every nonterminal state under the five menu orderings and average.
    2. Derive occupancy and exact advantages, and record every candidate.
    3. Apply the predeclared depth, occupancy, and stratum rules.
    4. Emit the 36 arms and 24 pairs the rollout stage will fill.
    """
    # Step 1 and 2: exact constrained credit and candidate evidence per root.
    evaluators: list[FloatPolicyEvaluator] = []
    candidates_by_root: list[tuple[StateCandidate, ...]] = []
    selection_rows: list[dict[str, object]] = []
    for problem in problems:
        root = problem.state
        dag = build_reasoning_dag(root)
        constrained_by_state: dict[State, dict[Action, float]] = {}
        for state in dag.topological_order():
            if is_terminal(state):
                continue
            distributions = []
            for ordering_id in MENU_ORDERING_IDS:
                record = scorer.score_states(
                    [state],
                    batch_size=1,
                    menu_ordering_id=ordering_id,
                    rules=DEFAULT_RULES,
                )[0]
                distributions.append(dict(record.probabilities))
            constrained_by_state[state] = average_action_distributions(distributions)
        evaluator, candidates = _root_selection_candidates(
            root, constrained_by_state, dag
        )
        evaluators.append(evaluator)
        candidates_by_root.append(candidates)

    # Step 3: predeclared selection, which never inspects rollout outcomes.
    selections = select_benchmark_pairs(candidates_by_root)

    # Step 4: materialize arms and pairs.
    arms: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for selection, problem, evaluator, candidates in zip(
        selections, problems, evaluators, candidates_by_root, strict=True
    ):
        target, values = selection.canonical_key
        state = State(values=values, target=target)
        action_by_rendering = {
            action.render(): action for action in allowed_actions(state)
        }
        state_arm_id = _arm_identifier(selection.root_index, "state", None)
        arms.append(
            {
                "arm_id": state_arm_id,
                "root_index": selection.root_index,
                "role": "state",
                "action": None,
                "state": serialize_state(state),
                "deterministic_outcome": _deterministic_outcome(state),
                "reference_pool_size": args.reference_pool_size,
                "constrained_value": evaluator.v_pi(state),
                "responses": [],
            }
        )
        for selected_action in selection.actions:
            action = action_by_rendering[selected_action.action_key]
            child = transition(state, action)
            child_arm_id = _arm_identifier(selection.root_index, "child", action)
            arms.append(
                {
                    "arm_id": child_arm_id,
                    "root_index": selection.root_index,
                    "role": "child",
                    "action": serialize_action(action),
                    "state": serialize_state(child),
                    "deterministic_outcome": _deterministic_outcome(child),
                    "reference_pool_size": args.reference_pool_size,
                    "constrained_value": evaluator.v_pi(child),
                    "responses": [],
                }
            )
            pairs.append(
                {
                    "root_index": selection.root_index,
                    "root": serialize_state(problem.state),
                    "state": serialize_state(state),
                    "action": serialize_action(action),
                    "depth": selection.selected_depth,
                    "stratum": selected_action.stratum,
                    "advantage_rank": selected_action.advantage_rank,
                    "state_arm_id": state_arm_id,
                    "child_arm_id": child_arm_id,
                    "constrained_v_pi": evaluator.v_pi(state),
                    "constrained_q_pi": evaluator.q_pi(state, action),
                    "constrained_advantage": evaluator.a_pi(state, action),
                }
            )
        selection_rows.append(
            {
                "root_index": selection.root_index,
                "root": serialize_state(problem.state),
                "source_row_index": problem.source_row_index,
                "assigned_depth": selection.assigned_depth,
                "selected_depth": selection.selected_depth,
                "selected_state": serialize_state(state),
                "selected_state_occupancy": selection.occupancy,
                "omitted_stratum": selection.omitted_stratum,
                "selected_actions": [
                    {
                        "action": serialize_action(
                            action_by_rendering[chosen.action_key]
                        ),
                        "stratum": chosen.stratum,
                        "advantage_rank": chosen.advantage_rank,
                        "constrained_advantage": chosen.constrained_advantage,
                    }
                    for chosen in selection.actions
                ],
                "candidates": [
                    {
                        "state": {
                            "target": candidate.canonical_key[0],
                            "values": list(candidate.canonical_key[1]),
                        },
                        "depth": candidate.depth,
                        "occupancy": candidate.occupancy,
                        "advantages": dict(
                            sorted(candidate.advantage_by_action.items())
                        ),
                    }
                    for candidate in candidates
                ],
            }
        )
    return selection_rows, arms, pairs


def _collect_missing_completions(
    artifact: dict[str, object],
    artifact_path: Path,
    *,
    generator: Any,
    args: argparse.Namespace,
) -> None:
    """Fill both pools of every generating arm, checkpointing after each batch."""
    for arm in artifact["arms"]:
        if arm["deterministic_outcome"] is not None:
            continue
        state_value = arm["state"]
        state = State(
            values=tuple(state_value["values"]), target=int(state_value["target"])
        )
        messages = build_completion_messages(state)
        responses = arm["responses"]
        assert isinstance(responses, list)
        for pool, pool_size in (
            ("reference", int(arm["reference_pool_size"])),
            ("estimator", ESTIMATOR_POOL_SIZE),
        ):
            batch_count = math.ceil(pool_size / args.batch_size)
            saved_batches = sorted(
                int(row["batch_index"]) for row in responses if row["pool"] == pool
            )
            completed = len(set(saved_batches))
            # Checkpoints are written only after a whole batch, so saved batches
            # must form an exact prefix with no partial or duplicated batch.
            expected_prefix = [
                index
                for index in range(completed)
                for _ in range(
                    min(args.batch_size, pool_size - index * args.batch_size)
                )
            ]
            if saved_batches != expected_prefix:
                raise ValueError(
                    f"Arm {arm['arm_id']!r} pool {pool!r} holds a partial or "
                    "duplicated batch; stale responses must not be reused"
                )
            for batch_index in range(completed, batch_count):
                count = min(args.batch_size, pool_size - batch_index * args.batch_size)
                seed = _completion_seed(
                    args.seed, str(arm["arm_id"]), pool, batch_index
                )
                completions = generator.generate_batch(
                    messages,
                    count=count,
                    seed=seed,
                    max_new_tokens=args.token_cap,
                )
                for offset, completion in enumerate(completions):
                    scored = score_completion(completion.text, state)
                    responses.append(
                        {
                            "pool": pool,
                            "batch_index": batch_index,
                            "batch_seed": seed,
                            "sample_index": batch_index * args.batch_size + offset,
                            "status": scored.status.value,
                            "success": scored.is_success,
                            "expression": scored.expression,
                            "value": scored.value,
                            "generated_token_count": completion.generated_token_count,
                            "is_truncated": completion.is_truncated,
                            "raw_response": completion.text,
                        }
                    )
                responses.sort(key=lambda row: (row["pool"], row["sample_index"]))
                artifact["progress"]["completed_responses"] = sum(
                    len(candidate["responses"]) for candidate in artifact["arms"]
                )
                _write_json(artifact_path, artifact)


def run_stage_b(args: argparse.Namespace) -> None:
    """Run or resume the conclusion-robustness benchmark."""
    _require_model(args)
    _validate_stage_contract(args, stage="stage-b")
    output_dir = Path(args.output)
    artifact_path = output_dir / STAGE_B_FILE
    metadata = _completion_metadata(args)

    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text())
        existing_metadata = artifact.get("metadata")
        if not isinstance(existing_metadata, Mapping):
            raise ValueError("Stage B artifact metadata is malformed")
        if existing_metadata.get("schema_version") != STAGE_B_SCHEMA_VERSION:
            raise ValueError(
                "Stage B artifact was written by the superseded complete-DAG "
                "benchmark; move it aside instead of resuming it"
            )
        for key, expected in metadata.items():
            if existing_metadata.get(key) != expected:
                raise ValueError(
                    f"Stage B artifact metadata mismatch for {key!r}; stale "
                    "responses must not be reused"
                )
    else:
        artifact = None

    scorer = _load_constrained_scorer(args)
    constrained_metadata = _model_metadata(scorer.metadata)
    if artifact is None:
        problems = select_bridge_problems(Path(args.dataset), count=args.problem_limit)
        selections, arms, pairs = _build_stage_b_selection(
            args, problems, scorer=scorer
        )
        artifact = _new_stage_b(metadata, selections, arms, pairs)
    existing = artifact["metadata"].get("constrained_model")
    if existing not in (None, constrained_metadata):
        raise ValueError("Stage B constrained model metadata mismatch")
    artifact["metadata"]["constrained_model"] = constrained_metadata
    _write_json(artifact_path, artifact)
    del scorer

    generator = _load_completion_generator(args)
    completion_metadata = _model_metadata(generator.metadata)
    existing = artifact["metadata"].get("completion_model")
    if existing not in (None, completion_metadata):
        raise ValueError("Stage B completion model metadata mismatch")
    artifact["metadata"]["completion_model"] = completion_metadata
    _write_json(artifact_path, artifact)

    started = time.monotonic()
    _collect_missing_completions(
        artifact, artifact_path, generator=generator, args=args
    )
    artifact["progress"]["generation_seconds"] = time.monotonic() - started
    _finalize_stage_b(artifact, args)
    _write_json(artifact_path, artifact)
    (output_dir / STAGE_B_MARKDOWN).write_text(stage_b_markdown(artifact))
    print(f"Wrote {artifact_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "samples_per_ordering", 1) < 1:
        parser.error("--samples-per-ordering must be at least one")
    if getattr(args, "max_new_tokens", 1) < 1:
        parser.error("--max-new-tokens must be at least one")
    if getattr(args, "bootstrap_replicates", 1) < 1:
        parser.error("--bootstrap-replicates must be at least one")
    if getattr(args, "token_cap", 1) < 1:
        parser.error("--token-cap must be at least one")
    if getattr(args, "batch_size", 1) < 1:
        parser.error("--batch-size must be at least one")
    if getattr(args, "batches", 1) < 1:
        parser.error("--batches must be at least one")
    if any(size < 1 for size in getattr(args, "batch_sizes", ()) or ()):
        parser.error("--batch-sizes must all be at least one")
    if getattr(args, "seconds_per_response", 1.0) <= 0.0:
        parser.error("--seconds-per-response must be positive")
    if getattr(args, "gpu_hourly_usd", 1.0) <= 0.0:
        parser.error("--gpu-hourly-usd must be positive")
    if getattr(args, "budget_ceiling_usd", 1.0) <= 0.0:
        parser.error("--budget-ceiling-usd must be positive")
    for name in ("constrained_temperature", "deliberative_temperature", "completion_temperature"):
        if getattr(args, name, 1.0) <= 0.0:
            parser.error("temperatures must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
