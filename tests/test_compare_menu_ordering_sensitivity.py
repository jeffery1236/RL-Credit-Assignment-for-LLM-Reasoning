"""Tests for complete-DAG menu-order sensitivity comparisons."""

import json

import pytest

from env.countdown import DEFAULT_RULES, Action, Operator, State, allowed_actions
from experiments.compare_menu_ordering_sensitivity import (
    compare_menu_ordering_caches,
    comparison_to_markdown,
    main,
)
from oracle.evaluation import PolicyValidationError
from policies.allowed_action_llm_policy import (
    POLICY_CACHE_SCHEMA_VERSION,
    PROMPT_VERSION,
    AllowedActionPolicyCache,
    PolicyCacheMetadata,
    StateActionProbabilities,
    build_allowed_action_menu,
    save_cache_snapshot,
    serialize_rules,
    serialize_state,
)

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def write_single_state_cache(path, *, root, ordering_id, probabilities):
    actions = allowed_actions(root)
    menu = build_allowed_action_menu(
        root,
        actions,
        LABELS,
        menu_ordering_id=ordering_id,
    )
    metadata = PolicyCacheMetadata(
        schema_version=POLICY_CACHE_SCHEMA_VERSION,
        model_repository="fake/model",
        model_revision="fake-model-sha",
        tokenizer_repository="fake/model",
        tokenizer_revision="fake-tokenizer-sha",
        prompt_version=PROMPT_VERSION,
        enable_thinking=False,
        labels=LABELS,
        label_token_ids=tuple(range(len(LABELS))),
        menu_ordering_id=ordering_id,
        temperature=1.0,
        rules=serialize_rules(DEFAULT_RULES),
        device="cpu",
        dtype="float32",
        software_versions={"torch": "test", "transformers": "test"},
        root_problem=serialize_state(root),
    )
    save_cache_snapshot(
        path,
        AllowedActionPolicyCache(
            metadata=metadata,
            records={
                root: StateActionProbabilities(
                    state=root,
                    menu=menu,
                    probabilities=probabilities,
                )
            },
        ),
    )


def test_comparison_matches_actions_across_different_menu_orders(tmp_path):
    root = State(values=(1, 2), target=3)
    add = Action(Operator.ADD, 1, 2)
    multiply = Action(Operator.MUL, 1, 2)
    subtract = Action(Operator.SUB, 2, 1)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_single_state_cache(
        first_path,
        root=root,
        ordering_id="ordering-a",
        probabilities={add: 0.5, multiply: 0.25, subtract: 0.25},
    )
    write_single_state_cache(
        second_path,
        root=root,
        ordering_id="ordering-b",
        probabilities={add: 0.25, multiply: 0.5, subtract: 0.25},
    )

    comparison = compare_menu_ordering_caches(
        {"ordering-a": first_path, "ordering-b": second_path}
    )

    assert comparison["menu_orderings"]["ordering-a"]["root_v_pi"] == pytest.approx(0.5)
    assert comparison["menu_orderings"]["ordering-b"]["root_v_pi"] == pytest.approx(0.25)
    assert comparison["pairwise_total_variation"][0]["distance"] == pytest.approx(
        0.25
    )
    assert comparison["winning_root_action"] == {
        "left": 1,
        "operator": "+",
        "result": 3,
        "right": 2,
    }


def test_comparison_rejects_different_roots_or_state_coverage(tmp_path):
    first_root = State(values=(1, 2), target=3)
    second_root = State(values=(1, 3), target=4)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_single_state_cache(
        first_path,
        root=first_root,
        ordering_id="ordering-a",
        probabilities={
            action: 1.0 / len(allowed_actions(first_root))
            for action in allowed_actions(first_root)
        },
    )
    write_single_state_cache(
        second_path,
        root=second_root,
        ordering_id="ordering-b",
        probabilities={
            action: 1.0 / len(allowed_actions(second_root))
            for action in allowed_actions(second_root)
        },
    )

    with pytest.raises(PolicyValidationError):
        compare_menu_ordering_caches(
            {"ordering-a": first_path, "ordering-b": second_path}
        )

    incompatible = json.loads(first_path.read_text())
    incompatible["metadata"]["menu_ordering_id"] = "ordering-b"
    incompatible["records"] = []
    second_path.write_text(json.dumps(incompatible))
    with pytest.raises(PolicyValidationError):
        compare_menu_ordering_caches(
            {"ordering-a": first_path, "ordering-b": second_path}
        )


def test_markdown_reports_winning_action_and_label_bias(tmp_path):
    root = State(values=(1, 2), target=3)
    actions = allowed_actions(root)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    for path, ordering_id in [
        (first_path, "ordering-a"),
        (second_path, "ordering-b"),
    ]:
        write_single_state_cache(
            path,
            root=root,
            ordering_id=ordering_id,
            probabilities={action: 1.0 / len(actions) for action in actions},
        )
    comparison = compare_menu_ordering_caches(
        {"ordering-a": first_path, "ordering-b": second_path}
    )

    markdown = comparison_to_markdown(comparison)

    assert "Winning root action" in markdown
    assert "Average probability by option label" in markdown


def test_cli_writes_canonical_json_and_markdown(tmp_path):
    root = State(values=(1, 2), target=3)
    actions = allowed_actions(root)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    output = tmp_path / "output"
    for path, ordering_id in [
        (first_path, "ordering-a"),
        (second_path, "ordering-b"),
    ]:
        write_single_state_cache(
            path,
            root=root,
            ordering_id=ordering_id,
            probabilities={action: 1.0 / len(actions) for action in actions},
        )

    main(
        [
            "--cache",
            f"ordering-a={first_path}",
            "--cache",
            f"ordering-b={second_path}",
            "--output",
            str(output),
        ]
    )

    parsed = json.loads((output / "menu_ordering_sensitivity.json").read_text())
    assert sorted(parsed["menu_orderings"]) == ["ordering-a", "ordering-b"]
    assert (output / "menu_ordering_sensitivity.md").exists()
