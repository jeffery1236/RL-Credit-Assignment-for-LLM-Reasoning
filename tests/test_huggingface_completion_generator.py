"""Tests for batched full-solution generation without model downloads."""

import torch

from env.countdown import State
from inference.huggingface_completion_generator import (
    CompletionGeneratorMetadata,
    HuggingFaceCompletionGenerator,
)
from policies.countdown_completion_policy import build_completion_messages

STATE = State(values=(2, 3, 4, 5), target=14)
PROMPT_TOKENS = (4, 5)

# transformers.generate() rejects unknown model_kwargs, so the fakes must too.
SUPPORTED_GENERATE_KWARGS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "do_sample",
        "temperature",
        "max_new_tokens",
        "num_return_sequences",
        "pad_token_id",
        "eos_token_id",
    }
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "prompt"

    def __call__(self, prompts, **kwargs):
        assert kwargs == {
            "return_tensors": "pt",
            "padding": True,
            "add_special_tokens": False,
        }
        return {"input_ids": torch.tensor([list(PROMPT_TOKENS) for _ in prompts])}

    def decode(self, token_ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return "|".join(str(int(value)) for value in token_ids)


class BatchedFakeModel:
    """Returns one distinct global-RNG draw per requested sequence."""

    def __init__(self, *, padded_tail=0):
        self.padded_tail = padded_tail
        self.calls = []

    def generate(self, **kwargs):
        unsupported = sorted(set(kwargs) - SUPPORTED_GENERATE_KWARGS)
        assert not unsupported, f"unsupported model_kwargs: {unsupported}"
        assert kwargs["do_sample"] is True
        self.calls.append(kwargs)
        count = kwargs["num_return_sequences"]
        prompt = kwargs["input_ids"][0]
        rows = []
        for _ in range(count):
            drawn = torch.randint(low=100, high=1_000_000, size=(2,))
            padding = torch.zeros(self.padded_tail, dtype=drawn.dtype)
            rows.append(torch.cat([prompt, drawn, padding]))
        return torch.stack(rows)


def _generator(model, tokenizer):
    return HuggingFaceCompletionGenerator(
        model=model,
        tokenizer=tokenizer,
        metadata=CompletionGeneratorMetadata(
            model_repository="Qwen/Qwen3.5-4B",
            model_revision="fake-sha",
            tokenizer_repository="Qwen/Qwen3.5-4B",
            tokenizer_revision="fake-sha",
            device="cpu",
            dtype="float16",
            temperature=0.6,
        ),
    )


def test_one_generate_call_serves_the_whole_batch():
    model = BatchedFakeModel()
    tokenizer = FakeTokenizer()

    completions = _generator(model, tokenizer).generate_batch(
        build_completion_messages(STATE), count=8, seed=42, max_new_tokens=1024
    )

    assert len(completions) == 8
    assert len(model.calls) == 1, "batching must not fall back to one call per sample"
    assert model.calls[0]["num_return_sequences"] == 8
    assert model.calls[0]["max_new_tokens"] == 1024
    assert model.calls[0]["temperature"] == 0.6
    assert tokenizer.template_calls


def test_batch_seed_reproduces_and_distinct_seeds_differ():
    def run(seed):
        return tuple(
            completion.text
            for completion in _generator(
                BatchedFakeModel(), FakeTokenizer()
            ).generate_batch(
                build_completion_messages(STATE),
                count=4,
                seed=seed,
                max_new_tokens=1024,
            )
        )

    assert run(7) == run(7)
    assert run(7) != run(8)
    assert len(set(run(7))) == 4, "each sequence in a batch must differ"


def test_only_the_continuation_is_returned_and_padding_is_not_counted():
    model = BatchedFakeModel(padded_tail=5)

    completions = _generator(model, FakeTokenizer()).generate_batch(
        build_completion_messages(STATE), count=2, seed=3, max_new_tokens=1024
    )

    for completion in completions:
        assert str(PROMPT_TOKENS[0]) not in completion.text.split("|")[:2]
        # Two sampled tokens plus five pad tokens must count as two.
        assert completion.generated_token_count == 2
        assert completion.is_truncated is False


def test_using_the_whole_allowance_is_reported_as_truncated():
    completions = _generator(BatchedFakeModel(), FakeTokenizer()).generate_batch(
        build_completion_messages(STATE), count=1, seed=3, max_new_tokens=2
    )

    assert completions[0].generated_token_count == 2
    assert completions[0].is_truncated is True
