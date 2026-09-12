"""Hugging Face batched generation for full-solution Countdown rollouts.

Key Terms:
- Arm: one state whose completions estimate a single value, either V(s) or the
  Q(s,a) of an already-applied action.
- Batch seed: the global RNG seed used for one ``generate`` call; every response
  in that call is reproducible only as a whole batch.

Generation is batched rather than one response at a time because ``generate``
samples from the global RNG and accepts no per-sequence generator, so seeding
each response independently would force batch size one and cost roughly eight
times more GPU time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class GeneratedCompletion:
    """One decoded completion plus the length evidence calibration needs."""

    text: str
    generated_token_count: int
    # True when the response used its whole allowance, so reasoning was cut off
    # rather than finished. Truncated responses are scored as failed rollouts.
    is_truncated: bool


@dataclass(frozen=True, slots=True)
class CompletionGeneratorMetadata:
    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    device: str
    dtype: str
    temperature: float


class HuggingFaceCompletionGenerator:
    """Load Qwen once and draw reproducible batches of full-solution rollouts."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        metadata: CompletionGeneratorMetadata,
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
    ) -> "HuggingFaceCompletionGenerator":
        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError(f"Unsupported device: {device}")
        if temperature <= 0.0:
            raise ValueError("temperature must be greater than zero")

        # Keep heavy imports local so module import and --help stay lightweight.
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
            metadata=CompletionGeneratorMetadata(
                model_repository=model_repository,
                model_revision=resolved_revision,
                tokenizer_repository=model_repository,
                tokenizer_revision=resolved_revision,
                device=resolved_device,
                dtype=str(dtype).removeprefix("torch."),
                temperature=temperature,
            ),
        )

    def generate_batch(
        self,
        messages: Sequence[dict[str, str]],
        *,
        count: int,
        seed: int,
        max_new_tokens: int,
    ) -> tuple[GeneratedCompletion, ...]:
        """Draw ``count`` completions for one prompt under a single batch seed."""
        if count < 1:
            raise ValueError("count must be at least one")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least one")
        import torch

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
                num_return_sequences=count,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        completions = []
        for sequence in generated:
            continuation = sequence[input_length:]
            # Padding only appears after the model stopped, so stripping it
            # keeps the token count equal to what generation actually produced.
            token_ids = [
                int(token)
                for token in continuation.tolist()
                if int(token) != self._tokenizer.pad_token_id
            ]
            completions.append(
                GeneratedCompletion(
                    text=self._tokenizer.decode(
                        continuation, skip_special_tokens=True
                    ),
                    generated_token_count=len(token_ids),
                    is_truncated=len(token_ids) >= max_new_tokens,
                )
            )
        return tuple(completions)
