"""Hugging Face autoregressive generation for the state-only bridge policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class DeliberativeGeneratorMetadata:
    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    device: str
    dtype: str
    temperature: float


class HuggingFaceDeliberativeGenerator:
    """Load Qwen once and generate one independently seeded response per state."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        metadata: DeliberativeGeneratorMetadata,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.metadata = metadata

    @classmethod
    def load(
        cls,
        model_repository: str,
        *,
        device: str = "auto",
        temperature: float = 0.6,
    ) -> "HuggingFaceDeliberativeGenerator":
        if model_repository != "Qwen/Qwen3.5-4B":
            raise ValueError("The policy bridge is restricted to Qwen/Qwen3.5-4B")
        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError(f"Unsupported device: {device}")
        if temperature <= 0.0:
            raise ValueError("temperature must be greater than zero")

        # Keep heavy imports local so normal imports and --help are lightweight.
        import torch
        from huggingface_hub import model_info
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
            model_repository, revision=resolved_revision
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_repository, revision=resolved_revision, dtype=dtype
        ).to(resolved_device)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            metadata=DeliberativeGeneratorMetadata(
                model_repository=model_repository,
                model_revision=resolved_revision,
                tokenizer_repository=model_repository,
                tokenizer_revision=resolved_revision,
                device=resolved_device,
                dtype=str(dtype).removeprefix("torch."),
                temperature=temperature,
            ),
        )

    def generate(
        self,
        messages_batch: Sequence[Sequence[dict[str, str]]],
        *,
        seeds: Sequence[int],
        max_new_tokens: int,
    ) -> tuple[str, ...]:
        """Generate separately so each saved response has an independent seed."""
        if len(messages_batch) != len(seeds):
            raise ValueError("messages_batch and seeds must have the same length")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least one")
        import torch

        responses = []
        for messages, seed in zip(messages_batch, seeds, strict=True):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = self._tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            inputs = {
                name: tensor.to(self.metadata.device) for name, tensor in encoded.items()
            }
            input_length = inputs["input_ids"].shape[1]
            # generate() samples from the global RNG and takes no generator kwarg.
            torch.manual_seed(seed)
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=self.metadata.temperature,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            responses.append(
                self._tokenizer.decode(
                    generated[0][input_length:], skip_special_tokens=True
                )
            )
        return tuple(responses)
