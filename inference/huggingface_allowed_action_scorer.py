"""Batched Hugging Face scoring for the stationary allowed-action LLM policy."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from string import ascii_letters, digits
from typing import Any, ClassVar, Sequence

import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from env.countdown import DEFAULT_RULES, Action, RuleSet, State, allowed_actions
from oracle.float_policy_evaluation import PROBABILITY_TOLERANCE
from policies.allowed_action_llm_policy import (
    MAX_ALLOWED_ACTIONS,
    MenuOption,
    StateActionProbabilities,
    build_allowed_action_menu,
    build_policy_messages,
)

@dataclass(frozen=True, slots=True)
class ValidatedLabelInventory:
    """Model-specific option labels and their exact next-token IDs."""

    CANDIDATE_LABELS: ClassVar[tuple[str, ...]] = tuple(ascii_letters + digits)

    labels: tuple[str, ...]
    token_ids: tuple[int, ...]

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        messages: Sequence[dict[str, str]],
    ) -> "ValidatedLabelInventory":
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        labels: list[str] = []
        token_ids: list[int] = []
        for label in cls.CANDIDATE_LABELS:
            extended_ids = tokenizer.encode(
                prompt + label,
                add_special_tokens=False,
            )
            if (
                len(extended_ids) != len(prompt_ids) + 1
                or extended_ids[:-1] != prompt_ids
            ):
                continue

            token_id = extended_ids[-1]
            if token_id in token_ids:
                continue
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if decoded != label:
                continue

            labels.append(label)
            token_ids.append(token_id)

        if len(labels) < MAX_ALLOWED_ACTIONS:
            raise ValueError(
                f"Tokenizer exposes {len(labels)} valid labels; "
                f"need {MAX_ALLOWED_ACTIONS}"
            )
        return cls(tuple(labels), tuple(token_ids))

    def token_ids_for(self, menu: Sequence[MenuOption]) -> tuple[int, ...]:
        expected_labels = self.labels[: len(menu)]
        actual_labels = tuple(option.label for option in menu)
        if actual_labels != expected_labels:
            raise ValueError(
                f"Menu labels {actual_labels!r} do not match token order "
                f"{expected_labels!r}"
            )
        return self.token_ids[: len(menu)]


@dataclass(frozen=True, slots=True)
class ScorerMetadata:
    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    device: str
    dtype: str
    temperature: float
    label_inventory: ValidatedLabelInventory


@dataclass(frozen=True, slots=True)
class ThroughputMeasurement:
    batch_size: int
    state_count: int
    elapsed_seconds: float
    states_per_second: float


@dataclass(frozen=True, slots=True)
class ThroughputBenchmark:
    states: tuple[State, ...]
    measurements: tuple[ThroughputMeasurement, ...]


class HuggingFaceAllowedActionScorer:
    """Score complete allowed-action menus with next-token model logits."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        metadata: ScorerMetadata,
    ) -> None:
        if (
            not math.isfinite(metadata.temperature)
            or metadata.temperature <= 0.0
        ):
            raise ValueError("temperature must be finite and greater than zero")
        self._model = model
        self._tokenizer = tokenizer
        self.metadata = metadata

    @classmethod
    def load(
        cls,
        model_repository: str,
        *,
        device: str = "auto",
        temperature: float = 1.0,
    ) -> "HuggingFaceAllowedActionScorer":
        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError(f"Unsupported device: {device}")
        if device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is not available")
            resolved_device = "cuda"
        elif device == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("MPS was requested but is not available")
            resolved_device = "mps"
        elif device == "cpu":
            resolved_device = "cpu"
        elif torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"

        dtype = (
            torch.float16 if resolved_device in {"mps", "cuda"} else torch.float32
        )
        resolved_revision = model_info(model_repository).sha
        if not resolved_revision:
            raise ValueError(
                f"Could not resolve an immutable revision for {model_repository}"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_repository,
            revision=resolved_revision,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither a pad token nor an EOS token")
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_repository,
            revision=resolved_revision,
            dtype=dtype,
        ).to(resolved_device)
        model.eval()

        probe_state = State(values=(1, 2), target=3)
        probe_menu = build_allowed_action_menu(
            probe_state,
            allowed_actions(probe_state),
            ValidatedLabelInventory.CANDIDATE_LABELS,
            menu_ordering_id="label-validation",
        )
        probe_messages = build_policy_messages(probe_state, probe_menu)
        label_inventory = ValidatedLabelInventory.from_tokenizer(
            tokenizer,
            probe_messages,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            metadata=ScorerMetadata(
                model_repository=model_repository,
                model_revision=resolved_revision,
                tokenizer_repository=model_repository,
                tokenizer_revision=resolved_revision,
                device=resolved_device,
                dtype=str(dtype).removeprefix("torch."),
                temperature=temperature,
                label_inventory=label_inventory,
            ),
        )

    def score_states(
        self,
        states: Sequence[State],
        *,
        batch_size: int,
        menu_ordering_id: str,
        rules: RuleSet = DEFAULT_RULES,
    ) -> tuple[StateActionProbabilities, ...]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        actions_by_state: list[tuple[Action, ...]] = []
        for state in states:
            actions = allowed_actions(state, rules)
            if not actions:
                raise ValueError(
                    f"Scoring requires nonterminal states; received {state!r}"
                )
            actions_by_state.append(actions)

        records: list[StateActionProbabilities] = []
        for start in range(0, len(states), batch_size):
            state_batch = states[start : start + batch_size]
            action_batch = actions_by_state[start : start + batch_size]
            menus = [
                build_allowed_action_menu(
                    state,
                    actions,
                    self.metadata.label_inventory.labels,
                    menu_ordering_id=menu_ordering_id,
                    rules=rules,
                )
                for state, actions in zip(
                    state_batch,
                    action_batch,
                    strict=True,
                )
            ]
            prompts = [
                self._tokenizer.apply_chat_template(
                    build_policy_messages(state, menu, rules),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for state, menu in zip(state_batch, menus, strict=True)
            ]
            encoded = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            device_inputs = {
                name: tensor.to(self.metadata.device)
                for name, tensor in encoded.items()
            }
            with torch.inference_mode():
                logits = self._model(**device_inputs).logits[:, -1, :]

            for state, menu, state_logits in zip(
                state_batch,
                menus,
                logits,
                strict=True,
            ):
                token_ids = self.metadata.label_inventory.token_ids_for(menu)
                selected_logits = (
                    state_logits[list(token_ids)].float().cpu().double()
                    / self.metadata.temperature
                )
                values = torch.softmax(selected_logits, dim=0).cpu().tolist()
                if any(
                    not math.isfinite(probability) or probability < 0.0
                    for probability in values
                ):
                    raise ValueError("Model produced malformed label probabilities")
                probabilities = {
                    option.action: probability
                    for option, probability in zip(menu, values, strict=True)
                }
                total = math.fsum(probabilities.values())
                if abs(total - 1.0) > PROBABILITY_TOLERANCE:
                    raise ValueError(
                        f"Probabilities at {state!r} sum to {total}, not 1"
                    )
                records.append(
                    StateActionProbabilities(
                        state=state,
                        menu=menu,
                        probabilities=probabilities,
                    )
                )
        return tuple(records)

    def benchmark(
        self,
        states: Sequence[State],
        *,
        batch_sizes: Sequence[int] = (1, 4, 8),
        menu_ordering_id: str,
        rules: RuleSet = DEFAULT_RULES,
    ) -> ThroughputBenchmark:
        if not states:
            raise ValueError("Benchmark requires at least one state")
        if len(states) <= 8:
            benchmark_states = tuple(states)
        else:
            indices = [
                round(index * (len(states) - 1) / 7)
                for index in range(8)
            ]
            benchmark_states = tuple(states[index] for index in indices)

        def synchronize() -> None:
            if self.metadata.device == "cuda":
                torch.cuda.synchronize()
            elif self.metadata.device == "mps":
                torch.mps.synchronize()

        measurements: list[ThroughputMeasurement] = []
        for batch_size in batch_sizes:
            if batch_size < 1:
                raise ValueError("batch sizes must be at least 1")
            try:
                warmup_states = benchmark_states[
                    : min(batch_size, len(benchmark_states))
                ]
                self.score_states(
                    warmup_states,
                    batch_size=batch_size,
                    menu_ordering_id=menu_ordering_id,
                    rules=rules,
                )
                synchronize()
                started = time.perf_counter()
                self.score_states(
                    benchmark_states,
                    batch_size=batch_size,
                    menu_ordering_id=menu_ordering_id,
                    rules=rules,
                )
                synchronize()
                elapsed = time.perf_counter() - started
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                break
            measurements.append(
                ThroughputMeasurement(
                    batch_size=batch_size,
                    state_count=len(benchmark_states),
                    elapsed_seconds=elapsed,
                    states_per_second=len(benchmark_states) / elapsed,
                )
            )
        return ThroughputBenchmark(
            states=benchmark_states,
            measurements=tuple(measurements),
        )
