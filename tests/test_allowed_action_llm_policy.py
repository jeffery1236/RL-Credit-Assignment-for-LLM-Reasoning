"""Tests for the stationary allowed-action LLM policy contract and cache."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from env.countdown import DEFAULT_RULES, State, allowed_actions
from oracle.evaluation import PolicyValidationError
from policies.allowed_action_llm_policy import (
    ALLOWED_ACTION_POLICY_CACHE_NAME,
    MAX_ALLOWED_ACTIONS,
    POLICY_CACHE_SCHEMA_VERSION,
    PROMPT_VERSION,
    AllowedActionPolicyCache,
    CachedAllowedActionLLMPolicy,
    PolicyCacheMetadata,
    StateActionProbabilities,
    build_allowed_action_menu,
    build_policy_messages,
    load_cache_snapshot,
    save_cache_snapshot,
    serialize_rules,
    serialize_state,
    validate_cache_coverage,
)

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def make_state(values, target):
    return State(values=tuple(sorted(values)), target=target)


def test_six_distinct_values_require_45_labels():
    state = make_state([1, 2, 3, 4, 5, 6], 999)
    actions = allowed_actions(state)

    assert len(actions) == MAX_ALLOWED_ACTIONS == 45
    menu = build_allowed_action_menu(state, actions, LABELS)
    assert len(menu) == 45
    assert {option.action for option in menu} == set(actions)


def test_menu_is_independent_of_input_action_order():
    state = make_state([3, 7, 10, 25], 245)
    actions = allowed_actions(state)

    forward = build_allowed_action_menu(state, actions, LABELS)
    reverse = build_allowed_action_menu(state, tuple(reversed(actions)), LABELS)

    assert forward == reverse


def test_menu_ordering_id_changes_order_but_not_action_set():
    state = make_state([3, 7, 10, 25], 245)
    actions = allowed_actions(state)

    first = build_allowed_action_menu(
        state, actions, LABELS, menu_ordering_id="first"
    )
    second = build_allowed_action_menu(
        state, actions, LABELS, menu_ordering_id="second"
    )

    assert first != second
    assert {option.action for option in first} == {
        option.action for option in second
    }


def test_wrong_action_set_is_rejected():
    state = make_state([3, 7, 10, 25], 245)
    actions = allowed_actions(state)

    with pytest.raises(PolicyValidationError):
        build_allowed_action_menu(state, actions[:-1], LABELS)


def test_too_few_labels_is_rejected():
    state = make_state([3, 7, 10, 25], 245)
    actions = allowed_actions(state)

    with pytest.raises(PolicyValidationError):
        build_allowed_action_menu(state, actions, LABELS[: len(actions) - 1])


def test_prompt_contains_current_state_and_no_history_fields():
    state = make_state([3, 7, 10, 25], 245)
    menu = build_allowed_action_menu(state, allowed_actions(state), LABELS)
    messages = build_policy_messages(state, menu)
    rendered = json.dumps(messages)

    assert "245" in rendered
    assert "3, 7, 10, 25" in rendered
    assert "previous action" not in rendered.lower()
    assert "trajectory" not in rendered.lower()
    assert "chain-of-thought" not in rendered.lower()
    assert "eventually make the target" in rendered.lower()
    assert "removed" in rendered.lower()
    assert "result is added" in rendered.lower()
    assert "future moves" in rendered.lower()
    assert "not just the immediate result" in rendered.lower()


def test_state_and_rules_serialization_are_explicit_and_canonical():
    state = make_state([25, 3, 10, 7], 245)

    assert serialize_state(state) == {
        "target": 245,
        "values": [3, 7, 10, 25],
    }
    assert serialize_rules(DEFAULT_RULES) == {
        "operators": ["+", "-", "*"],
        "min_intermediate": 1,
    }


def test_menu_is_byte_identical_across_processes():
    script = """
import json
from env.countdown import State, allowed_actions
from policies.allowed_action_llm_policy import build_allowed_action_menu

labels = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
state = State(values=(3, 7, 10, 25), target=245)
menu = build_allowed_action_menu(state, allowed_actions(state), labels)
print(json.dumps([[option.label, option.action.render()] for option in menu]))
"""

    first = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    second = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert first == second


def make_metadata():
    return PolicyCacheMetadata(
        schema_version=POLICY_CACHE_SCHEMA_VERSION,
        model_repository="Qwen/Qwen3-0.6B",
        model_revision="model-sha",
        tokenizer_repository="Qwen/Qwen3-0.6B",
        tokenizer_revision="tokenizer-sha",
        prompt_version=PROMPT_VERSION,
        enable_thinking=False,
        labels=LABELS,
        label_token_ids=tuple(range(len(LABELS))),
        menu_ordering_id="test-ordering",
        temperature=1.0,
        rules=serialize_rules(DEFAULT_RULES),
        device="cpu",
        dtype="float32",
        software_versions={"torch": "test", "transformers": "test"},
        root_problem={"values": [1, 2], "target": 3},
    )


def make_record(state):
    actions = allowed_actions(state)
    menu = build_allowed_action_menu(
        state, actions, LABELS, menu_ordering_id="test-ordering"
    )
    probability = 1.0 / len(actions)
    return StateActionProbabilities(
        state=state,
        menu=menu,
        probabilities={action: probability for action in actions},
    )


def test_cache_round_trip_preserves_action_mapping(tmp_path):
    root = State(values=(1, 2), target=3)
    record = make_record(root)
    path = tmp_path / ALLOWED_ACTION_POLICY_CACHE_NAME
    snapshot = AllowedActionPolicyCache(
        metadata=make_metadata(),
        records={root: record},
    )

    save_cache_snapshot(path, snapshot)
    loaded = load_cache_snapshot(path, expected_metadata=make_metadata())

    assert loaded == snapshot


def test_snapshot_rewrite_preserves_previous_batches(tmp_path):
    root = State(values=(1, 2), target=3)
    second = State(values=(1, 3), target=4)
    path = tmp_path / ALLOWED_ACTION_POLICY_CACHE_NAME
    first_snapshot = AllowedActionPolicyCache(
        metadata=make_metadata(),
        records={root: make_record(root)},
    )
    save_cache_snapshot(path, first_snapshot)

    resumed = load_cache_snapshot(path, expected_metadata=make_metadata())
    updated_records = dict(resumed.records)
    updated_records[second] = make_record(second)
    save_cache_snapshot(
        path,
        AllowedActionPolicyCache(
            metadata=resumed.metadata,
            records=updated_records,
        ),
    )

    assert set(load_cache_snapshot(path).records) == {root, second}


def test_cache_serialization_is_canonical_regardless_of_record_insertion_order(
    tmp_path,
):
    first = State(values=(1, 2), target=3)
    second = State(values=(1, 3), target=4)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    save_cache_snapshot(
        first_path,
        AllowedActionPolicyCache(
            metadata=make_metadata(),
            records={first: make_record(first), second: make_record(second)},
        ),
    )
    save_cache_snapshot(
        second_path,
        AllowedActionPolicyCache(
            metadata=make_metadata(),
            records={second: make_record(second), first: make_record(first)},
        ),
    )

    assert first_path.read_bytes() == second_path.read_bytes()


METADATA_MUTATIONS = [
    ("schema_version", "3.0"),
    ("model_repository", "different/model"),
    ("model_revision", "different-model-sha"),
    ("tokenizer_repository", "different/tokenizer"),
    ("tokenizer_revision", "different-tokenizer-sha"),
    ("prompt_version", "different-prompt"),
    ("enable_thinking", True),
    ("labels", tuple(reversed(LABELS))),
    ("label_token_ids", tuple(reversed(range(len(LABELS))))),
    ("menu_ordering_id", "different-ordering"),
    ("temperature", 0.5),
    ("rules", {"operators": ["+"], "min_intermediate": 1}),
    ("device", "mps"),
    ("dtype", "float16"),
    ("software_versions", {"torch": "different", "transformers": "test"}),
    ("root_problem", {"values": [1, 3], "target": 4}),
]


@pytest.mark.parametrize(("field_name", "different_value"), METADATA_MUTATIONS)
def test_every_metadata_mismatch_is_a_hard_error(
    tmp_path, field_name, different_value
):
    path = tmp_path / ALLOWED_ACTION_POLICY_CACHE_NAME
    snapshot = AllowedActionPolicyCache(metadata=make_metadata(), records={})
    save_cache_snapshot(path, snapshot)

    with pytest.raises(PolicyValidationError):
        load_cache_snapshot(
            path,
            expected_metadata=replace(
                make_metadata(), **{field_name: different_value}
            ),
        )


def test_coverage_validation_rejects_missing_and_extra_states():
    root = State(values=(1, 2), target=3)
    missing = State(values=(1, 3), target=4)
    snapshot = AllowedActionPolicyCache(
        metadata=make_metadata(),
        records={root: make_record(root)},
    )

    with pytest.raises(PolicyValidationError):
        validate_cache_coverage(snapshot, expected_states={root, missing})
    with pytest.raises(PolicyValidationError):
        validate_cache_coverage(snapshot, expected_states=set())


def test_duplicate_state_records_are_rejected_while_loading(tmp_path):
    root = State(values=(1, 2), target=3)
    path = tmp_path / ALLOWED_ACTION_POLICY_CACHE_NAME
    save_cache_snapshot(
        path,
        AllowedActionPolicyCache(
            metadata=make_metadata(),
            records={root: make_record(root)},
        ),
    )
    payload = json.loads(path.read_text())
    payload["records"].append(payload["records"][0])
    path.write_text(json.dumps(payload))

    with pytest.raises(PolicyValidationError):
        load_cache_snapshot(path)


def test_cache_rejects_probability_action_set_mismatch(tmp_path):
    root = State(values=(1, 2), target=3)
    record = make_record(root)
    action_to_remove = next(iter(record.probabilities))
    malformed = StateActionProbabilities(
        state=root,
        menu=record.menu,
        probabilities={
            action: probability
            for action, probability in record.probabilities.items()
            if action != action_to_remove
        },
    )
    path = tmp_path / ALLOWED_ACTION_POLICY_CACHE_NAME

    with pytest.raises(PolicyValidationError):
        save_cache_snapshot(
            path,
            AllowedActionPolicyCache(
                metadata=make_metadata(),
                records={root: malformed},
            ),
        )


@pytest.mark.parametrize("bad_probability", [-0.1, float("nan"), float("inf")])
def test_cache_rejects_negative_and_nonfinite_probabilities(
    tmp_path, bad_probability
):
    root = State(values=(1, 2), target=3)
    record = make_record(root)
    probabilities = dict(record.probabilities)
    probabilities[next(iter(probabilities))] = bad_probability
    malformed = StateActionProbabilities(
        state=root,
        menu=record.menu,
        probabilities=probabilities,
    )

    with pytest.raises(PolicyValidationError):
        save_cache_snapshot(
            tmp_path / ALLOWED_ACTION_POLICY_CACHE_NAME,
            AllowedActionPolicyCache(
                metadata=make_metadata(),
                records={root: malformed},
            ),
        )


def test_cached_policy_returns_complete_validated_state_action_probabilities():
    root = State(values=(1, 2), target=3)
    record = make_record(root)
    policy = CachedAllowedActionLLMPolicy({root: record})
    actions = allowed_actions(root)

    assert policy.state_action_probabilities(root, actions) == record
    assert policy.action_probs(root, actions) == record.probabilities


def test_cached_policy_rejects_different_allowed_action_set():
    root = State(values=(1, 2), target=3)
    policy = CachedAllowedActionLLMPolicy({root: make_record(root)})

    with pytest.raises(PolicyValidationError):
        policy.state_action_probabilities(root, allowed_actions(root)[:-1])


def test_cached_policy_rejects_state_missing_from_cache():
    root = State(values=(1, 2), target=3)
    policy = CachedAllowedActionLLMPolicy({})

    with pytest.raises(PolicyValidationError):
        policy.state_action_probabilities(root, allowed_actions(root))
