"""Tests for autoregressive deliberative generation without model downloads."""

from types import SimpleNamespace

import torch

from env.countdown import State, allowed_actions
from inference.huggingface_deliberative_generator import (
    DeliberativeGeneratorMetadata,
    HuggingFaceDeliberativeGenerator,
)
from policies.allowed_action_llm_policy import build_allowed_action_menu
from policies.deliberative_action_policy import build_deliberative_messages


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
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
        return {"input_ids": torch.tensor([[4, 5] for _ in prompts])}

    def decode(self, token_ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return "<answer>1 + 2 = 3</answer>"


# transformers.generate() rejects unknown model_kwargs, so the fake must too.
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


class FakeModel:
    def generate(self, **kwargs):
        unsupported = sorted(set(kwargs) - SUPPORTED_GENERATE_KWARGS)
        assert not unsupported, f"unsupported model_kwargs: {unsupported}"
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.6
        assert kwargs["max_new_tokens"] == 64
        assert kwargs["num_return_sequences"] == 1
        return torch.tensor([[4, 5, 6]])


class GlobalRngFakeModel:
    """Samples from the global RNG, the way transformers' sampler does."""

    def generate(self, **kwargs):
        unsupported = sorted(set(kwargs) - SUPPORTED_GENERATE_KWARGS)
        assert not unsupported, f"unsupported model_kwargs: {unsupported}"
        drawn = torch.randint(low=100, high=1_000_000, size=(1,))
        return torch.cat([kwargs["input_ids"][0], drawn]).unsqueeze(0)


class EchoTokenizer(FakeTokenizer):
    def decode(self, token_ids, **kwargs):
        return "|".join(str(int(value)) for value in token_ids)


def test_generator_uses_chat_template_without_native_thinking():
    state = State(values=(1, 2), target=3)
    menu = build_allowed_action_menu(
        state, allowed_actions(state), tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )
    tokenizer = FakeTokenizer()
    generator = HuggingFaceDeliberativeGenerator(
        model=FakeModel(),
        tokenizer=tokenizer,
        metadata=DeliberativeGeneratorMetadata(
            model_repository="Qwen/Qwen3.5-4B",
            model_revision="fake-sha",
            tokenizer_repository="Qwen/Qwen3.5-4B",
            tokenizer_revision="fake-sha",
            device="cpu",
            dtype="float32",
            temperature=0.6,
        ),
    )

    responses = generator.generate(
        [build_deliberative_messages(state, menu)],
        seeds=[42],
        max_new_tokens=64,
    )

    assert responses == ("<answer>1 + 2 = 3</answer>",)
    assert tokenizer.calls


def _seeded_generator(model, tokenizer):
    return HuggingFaceDeliberativeGenerator(
        model=model,
        tokenizer=tokenizer,
        metadata=DeliberativeGeneratorMetadata(
            model_repository="Qwen/Qwen3.5-4B",
            model_revision="fake-sha",
            tokenizer_repository="Qwen/Qwen3.5-4B",
            tokenizer_revision="fake-sha",
            device="cpu",
            dtype="float32",
            temperature=0.6,
        ),
    )


def test_each_seed_reproducibly_drives_sampling():
    state = State(values=(1, 2), target=3)
    menu = build_allowed_action_menu(
        state, allowed_actions(state), tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )
    messages = build_deliberative_messages(state, menu)

    def run(seeds):
        generator = _seeded_generator(GlobalRngFakeModel(), EchoTokenizer())
        return generator.generate(
            [messages for _ in seeds], seeds=list(seeds), max_new_tokens=64
        )

    # The same seed must reproduce, and distinct seeds must actually differ.
    assert run([7]) == run([7])
    assert run([7]) != run([8])
    distinct = run([11, 12, 13])
    assert len(set(distinct)) == 3, distinct
