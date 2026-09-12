"""Tests for Hugging Face scoring of the stationary allowed-action policy."""

import math
import os
from types import SimpleNamespace

import pytest
import torch

from env.countdown import State, allowed_actions
from inference.huggingface_allowed_action_scorer import (
    HuggingFaceAllowedActionScorer,
    ScorerMetadata,
    ValidatedLabelInventory,
)
from policies.allowed_action_llm_policy import (
    MenuOption,
    build_allowed_action_menu,
)


class FakeTokenizer:
    def __init__(self, token_by_label):
        self.token_by_label = token_by_label
        self.decoded_by_token = {
            token: label
            for label, token in token_by_label.items()
            if isinstance(token, int)
        }
        self.chat_template_calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        self.chat_template_calls.append(messages)
        return "PROMPT:"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "PROMPT:":
            return [100]
        label = text.removeprefix("PROMPT:")
        token = self.token_by_label[label]
        return [100, token] if isinstance(token, int) else [100, *token]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return self.decoded_by_token[token_ids[0]]

    def __call__(
        self,
        prompts,
        *,
        return_tensors,
        padding,
        add_special_tokens,
    ):
        assert return_tensors == "pt"
        assert padding is True
        assert add_special_tokens is False
        batch_size = len(prompts)
        return {
            "input_ids": torch.ones((batch_size, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
        }


class FakeModel:
    def __call__(self, **encoded):
        batch_size = encoded["input_ids"].shape[0]
        logits = torch.tensor(
            [0.0, 1.0, 2.0, 100.0],
            device=encoded["input_ids"].device,
        ).repeat(batch_size, 1)
        return SimpleNamespace(logits=logits.unsqueeze(1))


def make_fake_scorer(*, temperature=1.0, device="cpu"):
    inventory = ValidatedLabelInventory(
        labels=("A", "B", "C"),
        token_ids=(0, 1, 2),
    )
    tokenizer = FakeTokenizer({"A": 0, "B": 1, "C": 2})
    scorer = HuggingFaceAllowedActionScorer(
        model=FakeModel(),
        tokenizer=tokenizer,
        metadata=ScorerMetadata(
            model_repository="fake/model",
            model_revision="fake-sha",
            tokenizer_repository="fake/model",
            tokenizer_revision="fake-sha",
            device=device,
            dtype="float16" if device == "mps" else "float32",
            temperature=temperature,
            label_inventory=inventory,
        ),
    )
    return scorer, tokenizer


def test_label_inventory_has_at_least_45_unique_single_tokens():
    candidates = ValidatedLabelInventory.CANDIDATE_LABELS
    mapping = {label: index + 1 for index, label in enumerate(candidates)}
    inventory = ValidatedLabelInventory.from_tokenizer(
        FakeTokenizer(mapping),
        messages=(),
    )

    assert len(inventory.labels) >= 45
    assert len(set(inventory.token_ids)) == len(inventory.token_ids)


def test_invalid_labels_are_excluded_from_inventory():
    candidates = ValidatedLabelInventory.CANDIDATE_LABELS
    mapping = {label: index + 1 for index, label in enumerate(candidates)}
    mapping[candidates[0]] = [1, 2]
    mapping[candidates[1]] = mapping[candidates[2]]
    tokenizer = FakeTokenizer(mapping)

    inventory = ValidatedLabelInventory.from_tokenizer(tokenizer, messages=())

    assert candidates[0] not in inventory.labels
    assert candidates[1] not in inventory.labels
    assert len(inventory.labels) >= 45


def test_label_is_excluded_when_token_decodes_to_different_text():
    candidates = ValidatedLabelInventory.CANDIDATE_LABELS
    mapping = {label: index + 1 for index, label in enumerate(candidates)}
    tokenizer = FakeTokenizer(mapping)
    tokenizer.decoded_by_token[mapping[candidates[0]]] = " " + candidates[0]

    inventory = ValidatedLabelInventory.from_tokenizer(tokenizer, messages=())

    assert candidates[0] not in inventory.labels


def test_label_validation_fails_when_fewer_than_45_survive():
    candidates = ValidatedLabelInventory.CANDIDATE_LABELS
    mapping = {
        label: (index + 1 if index < 44 else [index + 1, index + 100])
        for index, label in enumerate(candidates)
    }

    with pytest.raises(ValueError, match="need 45"):
        ValidatedLabelInventory.from_tokenizer(
            FakeTokenizer(mapping),
            messages=(),
        )


def test_menu_label_order_matches_token_id_order():
    state = State(values=(1, 2), target=3)
    inventory = ValidatedLabelInventory(
        labels=("A", "B", "C"),
        token_ids=(11, 12, 13),
    )
    menu = build_allowed_action_menu(
        state, allowed_actions(state), inventory.labels
    )

    assert inventory.token_ids_for(menu) == (11, 12, 13)

    mismatched = (
        MenuOption(label="B", action=menu[0].action),
        *menu[1:],
    )
    with pytest.raises(ValueError):
        inventory.token_ids_for(mismatched)


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.nan, math.inf])
def test_scorer_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        make_fake_scorer(temperature=temperature)


def test_score_states_maps_label_probabilities_back_to_menu_actions():
    scorer, tokenizer = make_fake_scorer()
    state = State(values=(1, 2), target=3)

    record = scorer.score_states(
        [state],
        batch_size=1,
        menu_ordering_id="unit-test",
    )[0]

    expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0).tolist()
    assert set(record.probabilities) == set(allowed_actions(state))
    assert [record.probabilities[option.action] for option in record.menu] == (
        pytest.approx(expected)
    )
    assert math.fsum(record.probabilities.values()) == pytest.approx(1.0)
    assert tokenizer.chat_template_calls


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS regression requires Apple Silicon",
)
def test_score_states_moves_selected_mps_logits_to_cpu_for_float64_softmax():
    scorer, _ = make_fake_scorer(device="mps")
    state = State(values=(1, 2), target=3)

    record = scorer.score_states(
        [state],
        batch_size=1,
        menu_ordering_id="mps-regression",
    )[0]

    assert math.fsum(record.probabilities.values()) == pytest.approx(1.0)


def test_score_states_rejects_terminal_states():
    scorer, _ = make_fake_scorer()
    terminal = State(values=(3,), target=3)

    with pytest.raises(ValueError, match="nonterminal"):
        scorer.score_states(
            [terminal],
            batch_size=1,
            menu_ordering_id="unit-test",
        )


def test_benchmark_uses_only_eight_representative_states():
    scorer, _ = make_fake_scorer()
    states = tuple(State(values=(1, value), target=999) for value in range(2, 22))

    benchmark = scorer.benchmark(
        states,
        batch_sizes=(1, 4, 8),
        menu_ordering_id="benchmark-test",
    )

    assert len(benchmark.states) == 8
    assert benchmark.states[0] == states[0]
    assert benchmark.states[-1] == states[-1]
    assert [measurement.batch_size for measurement in benchmark.measurements] == [
        1,
        4,
        8,
    ]
    assert {measurement.state_count for measurement in benchmark.measurements} == {8}
    assert all(
        measurement.states_per_second > 0
        for measurement in benchmark.measurements
    )


def test_resolve_device_rejects_unknown_device():
    with pytest.raises(ValueError, match="Unsupported device"):
        HuggingFaceAllowedActionScorer.load("fake/model", device="tpu")


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 to download and load Qwen",
)
def test_qwen_returns_complete_normalized_probabilities():
    scorer = HuggingFaceAllowedActionScorer.load("Qwen/Qwen3-0.6B")
    state = State(values=(1, 2), target=3)

    record = scorer.score_states(
        [state],
        batch_size=1,
        menu_ordering_id="integration-test",
    )[0]

    assert set(record.probabilities) == set(allowed_actions(state))
    assert math.fsum(record.probabilities.values()) == pytest.approx(1.0)
