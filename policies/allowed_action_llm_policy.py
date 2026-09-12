"""Stationary allowed-action LLM policy contract and persistent cache."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from env.countdown import (
    DEFAULT_RULES,
    Action,
    Operator,
    RuleSet,
    State,
    allowed_actions,
)
from oracle.evaluation import PolicyValidationError
from oracle.float_policy_evaluation import PROBABILITY_TOLERANCE

PROMPT_VERSION = "phase2-allowed-action-v2"
DEFAULT_MENU_ORDERING_ID = "phase2-v1"
MAX_ALLOWED_ACTIONS = 45  # 45 because 6 max digits choose 2 with 3 operators (+, -, *)
POLICY_CACHE_SCHEMA_VERSION = "2.0"
ALLOWED_ACTION_POLICY_CACHE_NAME = "allowed_action_policy_cache.json"


@dataclass(frozen=True, slots=True)
class MenuOption:
    label: str
    action: Action


def serialize_state(state: State) -> dict[str, object]:
    return {"target": state.target, "values": list(state.values)}


def serialize_action(action: Action) -> dict[str, object]:
    return {
        "operator": action.operator.value,
        "left": action.left,
        "right": action.right,
        "result": action.result,
    }


def serialize_rules(rules: RuleSet) -> dict[str, object]:
    return {
        "operators": [operator.value for operator in rules.operators],
        "min_intermediate": rules.min_intermediate,
    }


def _menu_seed(state: State, rules: RuleSet, menu_ordering_id: str) -> int:
    identity = {
        "menu_ordering_id": menu_ordering_id,
        "rules": serialize_rules(rules),
        "state": serialize_state(state),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, "big")


def build_allowed_action_menu(
    state: State,
    actions: tuple[Action, ...],
    labels: Sequence[str],
    *,
    menu_ordering_id: str = DEFAULT_MENU_ORDERING_ID,
    rules: RuleSet = DEFAULT_RULES,
) -> tuple[MenuOption, ...]:
    expected_actions = allowed_actions(state, rules)
    if set(actions) != set(expected_actions):
        raise PolicyValidationError(
            f"Menu actions {set(actions)!r} do not match allowed actions "
            f"{set(expected_actions)!r} at {state!r}"
        )
    if len(labels) < len(expected_actions):
        raise PolicyValidationError(
            f"Need {len(expected_actions)} labels but received {len(labels)}"
        )

    shuffled_actions = list(expected_actions)
    random.Random(_menu_seed(state, rules, menu_ordering_id)).shuffle(shuffled_actions)
    return tuple(
        MenuOption(label=labels[index], action=action)
        for index, action in enumerate(shuffled_actions)
    )


def build_policy_messages(
    state: State,
    menu: tuple[MenuOption, ...],
    rules: RuleSet = DEFAULT_RULES,
) -> tuple[dict[str, str], ...]:
    operators = ", ".join(operator.value for operator in rules.operators)
    menu_text = "\n".join(
        f"{option.label}) {option.action.render()}" for option in menu
    )
    system = (
        "You are choosing the next move in a multi-step Countdown arithmetic "
        "game. Your goal is to eventually make the target number appear. "
        "Return only the option label and no other text."
    )
    user = (
        f"Target: {state.target}\n"
        f"Current numbers: {', '.join(str(value) for value in state.values)}\n"
        f"Allowed operators: {operators}\n"
        "Game rules:\n"
        "- Choose exactly two current numbers and one allowed operation.\n"
        "- The chosen numbers are removed and the operation result is added.\n"
        "- Subtraction must use the larger number minus the smaller number.\n"
        "- Every result must be positive.\n"
        "- Future moves continue with the new numbers until the target appears.\n"
        "- Remaining numbers may be unused once the target appears.\n"
        "Choose the move that gives the best chance of eventually reaching the "
        "target. Consider future moves, not just the immediate result.\n"
        "Allowed actions:\n"
        f"{menu_text}\n"
        "Answer with exactly one option label."
    )
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    )

## Cache classes ##

@dataclass(frozen=True, slots=True)
class PolicyCacheMetadata:
    schema_version: str
    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    prompt_version: str
    enable_thinking: bool
    labels: tuple[str, ...]
    label_token_ids: tuple[int, ...]
    menu_ordering_id: str
    temperature: float
    rules: Mapping[str, object]
    device: str
    dtype: str
    software_versions: Mapping[str, str]
    root_problem: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StateActionProbabilities:
    state: State
    menu: tuple[MenuOption, ...]
    probabilities: Mapping[Action, float]


@dataclass(frozen=True, slots=True)
class AllowedActionPolicyCache:
    metadata: PolicyCacheMetadata
    records: Mapping[State, StateActionProbabilities]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def _metadata_to_dict(metadata: PolicyCacheMetadata) -> dict[str, object]:
    return {
        "schema_version": metadata.schema_version,
        "model_repository": metadata.model_repository,
        "model_revision": metadata.model_revision,
        "tokenizer_repository": metadata.tokenizer_repository,
        "tokenizer_revision": metadata.tokenizer_revision,
        "prompt_version": metadata.prompt_version,
        "enable_thinking": metadata.enable_thinking,
        "labels": list(metadata.labels),
        "label_token_ids": list(metadata.label_token_ids),
        "menu_ordering_id": metadata.menu_ordering_id,
        "temperature": metadata.temperature,
        "rules": dict(metadata.rules),
        "device": metadata.device,
        "dtype": metadata.dtype,
        "software_versions": dict(metadata.software_versions),
        "root_problem": dict(metadata.root_problem),
    }


def _metadata_from_dict(value: object) -> PolicyCacheMetadata:
    if not isinstance(value, dict):
        raise PolicyValidationError("Cache metadata must be a JSON object")
    expected_keys = {
        "schema_version",
        "model_repository",
        "model_revision",
        "tokenizer_repository",
        "tokenizer_revision",
        "prompt_version",
        "enable_thinking",
        "labels",
        "label_token_ids",
        "menu_ordering_id",
        "temperature",
        "rules",
        "device",
        "dtype",
        "software_versions",
        "root_problem",
    }
    if set(value) != expected_keys:
        raise PolicyValidationError(
            f"Cache metadata keys {set(value)!r} do not match {expected_keys!r}"
        )
    try:
        metadata = PolicyCacheMetadata(
            schema_version=str(value["schema_version"]),
            model_repository=str(value["model_repository"]),
            model_revision=str(value["model_revision"]),
            tokenizer_repository=str(value["tokenizer_repository"]),
            tokenizer_revision=str(value["tokenizer_revision"]),
            prompt_version=str(value["prompt_version"]),
            enable_thinking=value["enable_thinking"],
            labels=tuple(value["labels"]),
            label_token_ids=tuple(value["label_token_ids"]),
            menu_ordering_id=str(value["menu_ordering_id"]),
            temperature=float(value["temperature"]),
            rules=dict(value["rules"]),
            device=str(value["device"]),
            dtype=str(value["dtype"]),
            software_versions=dict(value["software_versions"]),
            root_problem=dict(value["root_problem"]),
        )
    except (TypeError, ValueError) as error:
        raise PolicyValidationError(f"Malformed cache metadata: {error}") from error
    _validate_metadata(metadata)
    return metadata


def _validate_metadata(metadata: PolicyCacheMetadata) -> None:
    if metadata.schema_version != POLICY_CACHE_SCHEMA_VERSION:
        raise PolicyValidationError(
            f"Unsupported cache schema version {metadata.schema_version!r}"
        )
    if metadata.enable_thinking is not False:
        raise PolicyValidationError("Allowed-action policy requires thinking disabled")
    if len(metadata.labels) < MAX_ALLOWED_ACTIONS:
        raise PolicyValidationError(
            f"Cache has {len(metadata.labels)} labels; need {MAX_ALLOWED_ACTIONS}"
        )
    if len(metadata.labels) != len(metadata.label_token_ids):
        raise PolicyValidationError("Labels and label token IDs have different lengths")
    if len(set(metadata.labels)) != len(metadata.labels):
        raise PolicyValidationError("Cache labels are not unique")
    if len(set(metadata.label_token_ids)) != len(metadata.label_token_ids):
        raise PolicyValidationError("Cache label token IDs are not unique")
    if not math.isfinite(metadata.temperature) or metadata.temperature <= 0.0:
        raise PolicyValidationError("Cache temperature must be finite and positive")
    if set(metadata.software_versions) != {"torch", "transformers"}:
        raise PolicyValidationError(
            "Cache software_versions must identify torch and transformers"
        )

    _rules_from_dict(metadata.rules)
    try:
        State(
            values=tuple(metadata.root_problem["values"]),
            target=int(metadata.root_problem["target"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PolicyValidationError(
            f"Malformed root problem in cache metadata: {error}"
        ) from error


def _rules_from_dict(value: Mapping[str, object]) -> RuleSet:
    try:
        operators = tuple(Operator(symbol) for symbol in value["operators"])
        min_intermediate = int(value["min_intermediate"])
    except (KeyError, TypeError, ValueError) as error:
        raise PolicyValidationError(f"Malformed RuleSet in cache: {error}") from error
    return RuleSet(operators=operators, min_intermediate=min_intermediate)


def _action_from_dict(value: object) -> Action:
    if not isinstance(value, dict):
        raise PolicyValidationError("Cached action must be a JSON object")
    if set(value) != {"operator", "left", "right", "result"}:
        raise PolicyValidationError(f"Malformed cached action keys: {set(value)!r}")
    try:
        action = Action(
            operator=Operator(value["operator"]),
            left=int(value["left"]),
            right=int(value["right"]),
        )
        recorded_result = int(value["result"])
    except (TypeError, ValueError) as error:
        raise PolicyValidationError(f"Malformed cached action: {error}") from error
    if recorded_result != action.result:
        raise PolicyValidationError(
            f"Cached action result {recorded_result} does not match {action.result}"
        )
    return action


def _validate_record(
    record: StateActionProbabilities,
    metadata: PolicyCacheMetadata,
) -> None:
    rules = _rules_from_dict(metadata.rules)
    expected_actions = allowed_actions(record.state, rules)
    if not expected_actions:
        raise PolicyValidationError(
            f"Cache must not contain terminal state {record.state!r}"
        )

    menu_actions = tuple(option.action for option in record.menu)
    if len(set(menu_actions)) != len(menu_actions):
        raise PolicyValidationError(f"Cached menu repeats an action at {record.state!r}")
    if set(menu_actions) != set(expected_actions):
        raise PolicyValidationError(
            f"Cached menu actions do not match allowed actions at {record.state!r}"
        )
    expected_labels = metadata.labels[: len(record.menu)]
    actual_labels = tuple(option.label for option in record.menu)
    if actual_labels != expected_labels:
        raise PolicyValidationError(
            f"Cached menu labels {actual_labels!r} do not match {expected_labels!r}"
        )
    if set(record.probabilities) != set(expected_actions):
        raise PolicyValidationError(
            f"Cached probabilities do not cover allowed actions at {record.state!r}"
        )
    if any(
        not math.isfinite(probability) or probability < 0.0
        for probability in record.probabilities.values()
    ):
        raise PolicyValidationError(
            f"Cached probabilities are non-finite or negative at {record.state!r}"
        )
    total = math.fsum(record.probabilities.values())
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise PolicyValidationError(
            f"Cached probabilities at {record.state!r} sum to {total}, not 1"
        )


def _record_to_dict(
    record: StateActionProbabilities,
    metadata: PolicyCacheMetadata,
) -> dict[str, object]:
    _validate_record(record, metadata)
    return {
        "state": serialize_state(record.state),
        "menu": [
            {
                "label": option.label,
                "action": serialize_action(option.action),
                "probability": record.probabilities[option.action],
            }
            for option in record.menu
        ],
    }


def _record_from_dict(
    value: object,
    metadata: PolicyCacheMetadata,
) -> StateActionProbabilities:
    if not isinstance(value, dict) or set(value) != {"state", "menu"}:
        raise PolicyValidationError("Cached state record must contain state and menu")
    state_value = value["state"]
    if not isinstance(state_value, dict) or set(state_value) != {"target", "values"}:
        raise PolicyValidationError("Malformed cached state")
    try:
        state = State(
            values=tuple(state_value["values"]),
            target=int(state_value["target"]),
        )
    except (TypeError, ValueError) as error:
        raise PolicyValidationError(f"Malformed cached state: {error}") from error

    menu_value = value["menu"]
    if not isinstance(menu_value, list):
        raise PolicyValidationError("Cached menu must be a JSON array")
    menu: list[MenuOption] = []
    probabilities: dict[Action, float] = {}
    for item in menu_value:
        if not isinstance(item, dict) or set(item) != {
            "label",
            "action",
            "probability",
        }:
            raise PolicyValidationError("Malformed cached menu entry")
        action = _action_from_dict(item["action"])
        if action in probabilities:
            raise PolicyValidationError(
                f"Cached menu repeats action {action!r} at {state!r}"
            )
        try:
            probability = float(item["probability"])
        except (TypeError, ValueError) as error:
            raise PolicyValidationError(
                f"Malformed cached probability: {error}"
            ) from error
        menu.append(MenuOption(label=str(item["label"]), action=action))
        probabilities[action] = probability

    record = StateActionProbabilities(
        state=state,
        menu=tuple(menu),
        probabilities=probabilities,
    )
    _validate_record(record, metadata)
    return record


def save_cache_snapshot(path: Path, snapshot: AllowedActionPolicyCache) -> None:
    _validate_metadata(snapshot.metadata)
    records: list[dict[str, object]] = []
    for state in sorted(
        snapshot.records,
        key=lambda item: (item.target, len(item.values), item.values),
    ):
        record = snapshot.records[state]
        if record.state != state:
            raise PolicyValidationError(
                f"Cache key {state!r} does not match record state {record.state!r}"
            )
        records.append(_record_to_dict(record, snapshot.metadata))

    payload = {
        "metadata": _metadata_to_dict(snapshot.metadata),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(_canonical_json(payload))
    temporary_path.replace(path)


def load_cache_snapshot(
    path: Path,
    expected_metadata: PolicyCacheMetadata | None = None,
) -> AllowedActionPolicyCache:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PolicyValidationError(f"Could not read policy cache: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"metadata", "records"}:
        raise PolicyValidationError(
            "Policy cache must contain metadata and records"
        )

    metadata = _metadata_from_dict(payload["metadata"])
    if expected_metadata is not None:
        _validate_metadata(expected_metadata)
        if metadata != expected_metadata:
            raise PolicyValidationError(
                "Policy cache metadata does not match the requested evaluation"
            )

    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise PolicyValidationError("Policy cache records must be a JSON array")
    records: dict[State, StateActionProbabilities] = {}
    for raw_record in raw_records:
        record = _record_from_dict(raw_record, metadata)
        if record.state in records:
            raise PolicyValidationError(
                f"Policy cache repeats state {record.state!r}"
            )
        records[record.state] = record
    return AllowedActionPolicyCache(metadata=metadata, records=records)


def validate_cache_coverage(
    snapshot: AllowedActionPolicyCache,
    expected_states: set[State],
) -> None:
    cached_states = set(snapshot.records)
    if cached_states != expected_states:
        missing = expected_states - cached_states
        extra = cached_states - expected_states
        raise PolicyValidationError(
            f"Policy cache coverage mismatch: missing={missing!r}, extra={extra!r}"
        )


class CachedAllowedActionLLMPolicy:
    def __init__(
        self,
        records: Mapping[State, StateActionProbabilities],
    ) -> None:
        self._records = dict(records)

    def state_action_probabilities(
        self,
        state: State,
        actions: tuple[Action, ...],
    ) -> StateActionProbabilities:
        try:
            record = self._records[state]
        except KeyError as error:
            raise PolicyValidationError(
                f"No cached probabilities for state {state!r}"
            ) from error
        if set(record.probabilities) != set(actions):
            raise PolicyValidationError(
                f"Cached actions at {state!r} do not match evaluator actions"
            )
        return record

    def action_probs(
        self,
        state: State,
        actions: tuple[Action, ...],
    ) -> Mapping[Action, float]:
        return dict(self.state_action_probabilities(state, actions).probabilities)
