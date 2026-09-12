# Phase 2 Allowed-Action LLM Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate Qwen3-0.6B as a stationary policy over all allowed actions in one complete Countdown reasoning DAG, validate the numerical values against Monte Carlo rollouts, and produce a reproducible mixed-credit trajectory artifact.

**Architecture:** Use a two-stage pipeline. First, enumerate every nonterminal state, render a state-only action menu, batch-score the allowed option labels with Qwen, and persist a validated probability cache. Second, load that cache without model weights, compute floating-point `Vπ/Qπ/Aπ` over the existing DAG, validate the result, and generate JSON and Markdown artifacts.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face Transformers, Qwen3-0.6B, pytest, existing Countdown environment and oracle.

## Global Constraints

- The authoritative design is `docs/superpowers/specs/2026-08-15-phase-2-allowed-action-llm-policy-design.md`.
- The policy prompt depends only on the canonical current state, active `RuleSet`, and complete allowed-action set.
- Apply Qwen's chat template with `add_generation_prompt=True` and `enable_thinking=False`.
- Never include trajectory history, prior model output, chain-of-thought, or oracle information in the prompt.
- Menu ordering uses `hashlib.blake2b`; Python's built-in `hash()` is forbidden.
- The scorer extracts label logits and normalizes only across labels assigned to allowed actions.
- Six-number states require capacity for 45 allowed actions.
- Floating-point policy validation uses probability tolerance `1e-6`; Bellman validation uses numerical tolerance `1e-9`.
- Malformed probabilities, stale caches, incomplete state coverage, and tokenizer-label failures are hard errors. Never substitute uniform probabilities or truncate actions.
- Keep `oracle/evaluation.py` and its exact `Fraction` behavior unchanged.
- Start on Apple Silicon MPS. CUDA must work through the same scorer interface.
- Phase 2 targets only `(3, 7, 10, 25) → 245`; no full validation-set evaluation.
- Cloud spending requires a prior cost estimate and must remain below $100.
- Do not run `git add` or `git commit`. The user will review and commit all
  changes at the end.

## File Map

- Create `oracle/float_policy_evaluation.py`: validated floating-point dynamic programming over `ReasoningDag`.
- Create `tests/test_float_policy_evaluation.py`: hand-computed values, validation failures, and Bellman invariants.
- Create `policies/allowed_action_llm_policy.py`: canonical serialization, menu construction, state-only messages, cache schema, persistence, and cache-backed `Policy[float]`.
- Create `tests/test_allowed_action_llm_policy.py`: prompt/menu determinism and cache behavior.
- Create `inference/huggingface_allowed_action_scorer.py`: model loading, label-token validation, batching, scoring, and throughput measurement.
- Create `tests/test_huggingface_allowed_action_scorer.py`: fake-tokenizer and synthetic-logit tests plus opt-in real-model smoke test.
- Create `experiments/evaluate_allowed_action_llm_policy.py`: `score` and `evaluate` CLI commands, Monte Carlo validation, trajectory selection, and artifact writers.
- Create `tests/test_evaluate_allowed_action_llm_policy.py`: rollout, selection, artifact, and CLI tests.
- Modify `pyproject.toml`: add runtime dependencies and a model-test marker.
- Modify `tests/test_bootstrap.py`: include the new inference namespace and experiment CLI.
- Modify `README.md`: document Phase 2 status, commands, and interpretation limits.
- Modify `research/experiment_log.md`: record the throughput and worked-example experiment after it runs.
- Create `artifacts/phase2/qwen3-0.6b-worked-example/`: committed policy-cache snapshot, benchmark, evaluation summary, and trajectory artifacts.

---

### Task 1: Floating-Point Policy Evaluation

**Files:**
- Create: `oracle/float_policy_evaluation.py`
- Create: `tests/test_float_policy_evaluation.py`

**Interfaces:**
- Consumes: `Policy[float]`, `ReasoningDag`, `RuleSet`, `allowed_actions`, `is_success`, `is_terminal`, `transition`, and `PolicyValidationError`.
- Produces: `FloatPolicyEvaluator`, `PROBABILITY_TOLERANCE`, and `NUMERICAL_TOLERANCE`.

- [ ] **Step 1: Write hand-computed value and validation tests**

Create `tests/test_float_policy_evaluation.py` with these core cases:

```python
import math

import pytest

from env.countdown import Action, Operator, State
from oracle.evaluation import PolicyValidationError, build_reasoning_dag
from oracle.float_policy_evaluation import (
    NUMERICAL_TOLERANCE,
    FloatPolicyEvaluator,
)


def make_state(values, target):
    return State(values=tuple(sorted(values)), target=target)


class FloatUniformPolicy:
    def action_probs(self, state, actions):
        probability = 1.0 / len(actions)
        return {action: probability for action in actions}


def test_hand_computed_uniform_policy_value_and_advantage():
    root = make_state([1, 2], 3)
    evaluator = FloatPolicyEvaluator(FloatUniformPolicy(), build_reasoning_dag(root))

    add = Action(Operator.ADD, 1, 2)
    multiply = Action(Operator.MUL, 1, 2)
    subtract = Action(Operator.SUB, 2, 1)

    assert evaluator.v_pi(root) == pytest.approx(1.0 / 3.0)
    assert evaluator.q_pi(root, add) == pytest.approx(1.0)
    assert evaluator.a_pi(root, add) == pytest.approx(2.0 / 3.0)
    assert evaluator.a_pi(root, multiply) == pytest.approx(-1.0 / 3.0)
    assert evaluator.a_pi(root, subtract) == pytest.approx(-1.0 / 3.0)


def test_advantages_have_zero_policy_weighted_mean():
    root = make_state([1, 2, 3], 10)
    dag = build_reasoning_dag(root)
    evaluator = FloatPolicyEvaluator(FloatUniformPolicy(), dag)

    for state in dag.nodes:
        probabilities = evaluator.action_probs(state)
        if not probabilities:
            continue
        weighted_mean = math.fsum(
            probabilities[action] * evaluator.a_pi(state, action)
            for action in probabilities
        )
        assert abs(weighted_mean) <= NUMERICAL_TOLERANCE


@pytest.mark.parametrize(
    "probability_factory",
    [
        lambda actions: {actions[0]: 1.0},
        lambda actions: {action: 0.1 for action in actions},
        lambda actions: {
            action: (-0.1 if index == 0 else 1.1 / (len(actions) - 1))
            for index, action in enumerate(actions)
        },
        lambda actions: {
            action: (math.nan if index == 0 else 0.0)
            for index, action in enumerate(actions)
        },
        lambda actions: {
            action: (math.inf if index == 0 else 0.0)
            for index, action in enumerate(actions)
        },
    ],
)
def test_invalid_float_policy_output_raises(probability_factory):
    class InvalidPolicy:
        def action_probs(self, state, actions):
            return probability_factory(actions)

    root = make_state([1, 2, 3], 10)
    with pytest.raises(PolicyValidationError):
        FloatPolicyEvaluator(InvalidPolicy(), build_reasoning_dag(root))
```

- [ ] **Step 2: Implement the minimal validated evaluator**

Create `oracle/float_policy_evaluation.py` with this public shape:

```python
"""Floating-point policy values over an existing complete reasoning DAG."""

from __future__ import annotations

import math
from typing import Mapping

from env.countdown import (
    DEFAULT_RULES,
    Action,
    RuleSet,
    State,
    allowed_actions,
    is_success,
    is_terminal,
    transition,
)
from oracle.evaluation import Policy, PolicyValidationError, ReasoningDag

PROBABILITY_TOLERANCE = 1e-6
NUMERICAL_TOLERANCE = 1e-9


class FloatPolicyEvaluator:
    def __init__(
        self,
        policy: Policy[float],
        dag: ReasoningDag,
        rules: RuleSet = DEFAULT_RULES,
    ) -> None:
        self._policy = policy
        self._dag = dag
        self._rules = rules
        self._v_pi_cache: dict[State, float] = {}
        self._action_probs_cache: dict[State, Mapping[Action, float]] = {}
        self._compute_all()

    def _get_action_probs(self, state: State) -> Mapping[Action, float]:
        cached = self._action_probs_cache.get(state)
        if cached is not None:
            return cached

        actions = allowed_actions(state, self._rules)
        if not actions:
            return {}
        returned = dict(self._policy.action_probs(state, actions))
        if set(returned) != set(actions):
            raise PolicyValidationError(
                f"Policy returned actions {set(returned)!r}; expected {set(actions)!r}"
            )
        if any(not math.isfinite(value) for value in returned.values()):
            raise PolicyValidationError("Policy returned a non-finite probability")
        if any(value < 0.0 for value in returned.values()):
            raise PolicyValidationError("Policy returned a negative probability")
        total = math.fsum(returned.values())
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            raise PolicyValidationError(
                f"Policy probabilities at {state!r} sum to {total}, not 1"
            )
        self._action_probs_cache[state] = returned
        return returned

    def _compute_all(self) -> None:
        for state in reversed(self._dag.topological_order()):
            if is_success(state):
                self._v_pi_cache[state] = 1.0
            elif is_terminal(state, self._rules):
                self._v_pi_cache[state] = 0.0
            else:
                probabilities = self._get_action_probs(state)
                value = math.fsum(
                    probabilities[edge.action] * self._v_pi_cache[edge.target_state]
                    for edge in self._dag.out_edges(state)
                )
                if not -NUMERICAL_TOLERANCE <= value <= 1.0 + NUMERICAL_TOLERANCE:
                    raise PolicyValidationError(
                        f"Computed Vpi={value} outside [0, 1] at {state!r}"
                    )
                self._v_pi_cache[state] = value

    def v_pi(self, state: State) -> float:
        return self._v_pi_cache[state]

    def q_pi(self, state: State, action: Action) -> float:
        return self._v_pi_cache[transition(state, action, self._rules)]

    def a_pi(self, state: State, action: Action) -> float:
        return self.q_pi(state, action) - self.v_pi(state)

    def advantages(self, state: State) -> dict[Action, float]:
        return {
            action: self.a_pi(state, action)
            for action in allowed_actions(state, self._rules)
        }

    def action_probs(self, state: State) -> Mapping[Action, float]:
        if is_terminal(state, self._rules):
            return {}
        return self._get_action_probs(state)
```

Do not modify `ExactPolicyEvaluator`.

- [ ] **Step 3: Add Bellman, bounds, and exact-evaluator comparison tests**

Add tests that compare `FloatPolicyEvaluator(FloatUniformPolicy())` with
`ExactPolicyEvaluator(UniformPolicy())` for every state in roots
`(1,2)→3`, `(1,2,3)→10`, and `(3,7,10,25)→245`. Assert:

```python
assert float(exact.v_pi(state)) == pytest.approx(numerical.v_pi(state), abs=1e-12)
assert abs(
    numerical.v_pi(state)
    - math.fsum(
        numerical.action_probs(state)[edge.action]
        * numerical.v_pi(edge.target_state)
        for edge in dag.out_edges(state)
    )
) <= NUMERICAL_TOLERANCE
```

- [ ] **Step 4: Run the focused and existing oracle suites**

Run:

```bash
uv run pytest tests/test_float_policy_evaluation.py tests/test_oracle.py -q
```

Expected: all tests pass.

---

### Task 2: Deterministic Allowed-Action Menus and State-Only Prompts

**Files:**
- Create: `policies/allowed_action_llm_policy.py`
- Create: `tests/test_allowed_action_llm_policy.py`

**Interfaces:**
- Produces: `PROMPT_VERSION`, `DEFAULT_MENU_ORDERING_ID`, `MAX_ALLOWED_ACTIONS`, `MenuOption`, `serialize_state`, `serialize_action`, `serialize_rules`, `build_allowed_action_menu`, and `build_policy_messages`.
- Consumed later by: the Hugging Face scorer, cache layer, and experiment CLI.

- [ ] **Step 1: Write menu-capacity, determinism, and prompt tests**

Create `tests/test_allowed_action_llm_policy.py` with:

```python
import json
import subprocess
import sys

import pytest

from env.countdown import DEFAULT_RULES, State, allowed_actions
from oracle.evaluation import PolicyValidationError
from policies.allowed_action_llm_policy import (
    MAX_ALLOWED_ACTIONS,
    build_allowed_action_menu,
    build_policy_messages,
    serialize_rules,
    serialize_state,
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
    first = build_allowed_action_menu(state, actions, LABELS, menu_ordering_id="first")
    second = build_allowed_action_menu(state, actions, LABELS, menu_ordering_id="second")
    assert first != second
    assert {option.action for option in first} == {option.action for option in second}


def test_wrong_action_set_is_rejected():
    state = make_state([3, 7, 10, 25], 245)
    actions = allowed_actions(state)
    with pytest.raises(PolicyValidationError):
        build_allowed_action_menu(state, actions[:-1], LABELS)


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
```

Add a subprocess test that prints the serialized menu twice from separate
Python processes and asserts byte equality. This specifically catches use of
Python's randomized `hash()`.

- [ ] **Step 2: Implement canonical serialization and deterministic menus**

Start `policies/allowed_action_llm_policy.py` with:

```python
"""Stationary allowed-action LLM policy contract and persistent cache."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from env.countdown import (
    DEFAULT_RULES,
    Action,
    RuleSet,
    State,
    allowed_actions,
)
from oracle.evaluation import PolicyValidationError

PROMPT_VERSION = "phase2-allowed-action-v1"
DEFAULT_MENU_ORDERING_ID = "phase2-v1"
MAX_ALLOWED_ACTIONS = 45


@dataclass(frozen=True, slots=True)
class MenuOption:
    label: str
    action: Action


def serialize_state(state: State) -> dict:
    return {"target": state.target, "values": list(state.values)}


def serialize_action(action: Action) -> dict:
    return {
        "operator": action.operator.value,
        "left": action.left,
        "right": action.right,
        "result": action.result,
    }


def serialize_rules(rules: RuleSet) -> dict:
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
    expected = allowed_actions(state, rules)
    if set(actions) != set(expected):
        raise PolicyValidationError(
            f"Menu actions {set(actions)!r} do not match allowed actions {set(expected)!r}"
        )
    if len(labels) < len(expected):
        raise PolicyValidationError(
            f"Need {len(expected)} labels but received {len(labels)}"
        )
    shuffled = list(expected)
    random.Random(_menu_seed(state, rules, menu_ordering_id)).shuffle(shuffled)
    return tuple(
        MenuOption(label=labels[index], action=action)
        for index, action in enumerate(shuffled)
    )
```

- [ ] **Step 3: Implement the exact state-only message template**

Add:

```python
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
        "You select one allowed action in a Countdown arithmetic game. "
        "Return only the option label and no other text."
    )
    user = (
        f"Target: {state.target}\n"
        f"Current numbers: {', '.join(str(value) for value in state.values)}\n"
        f"Allowed operators: {operators}\n"
        "Choose exactly two current numbers. Subtraction must use the larger "
        "number minus the smaller number, and every result must be positive.\n"
        "Allowed actions:\n"
        f"{menu_text}\n"
        "Answer with exactly one option label."
    )
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    )
```

Do not add a parameter for history or previous reasoning.

- [ ] **Step 4: Run menu and existing policy tests**

Run:

```bash
uv run pytest tests/test_allowed_action_llm_policy.py tests/test_policies.py -q
```

Expected: all tests pass.

---

### Task 3: Resumable Probability Cache and Cache-Backed Policy

**Files:**
- Modify: `policies/allowed_action_llm_policy.py`
- Modify: `tests/test_allowed_action_llm_policy.py`

**Interfaces:**
- Produces: `PolicyCacheMetadata`, `StateActionProbabilities`, `AllowedActionPolicyCache`, `CachedAllowedActionLLMPolicy`, `load_cache_snapshot`, `save_cache_snapshot`, and `validate_cache_coverage`.
- Consumed later by: scorer and experiment CLI.

- [ ] **Step 1: Write cache round-trip and invalidation tests**

Add tests covering:

```python
from dataclasses import replace

from env.countdown import DEFAULT_RULES, State, allowed_actions
from oracle.evaluation import PolicyValidationError
from policies.allowed_action_llm_policy import (
    ALLOWED_ACTION_POLICY_CACHE_NAME,
    AllowedActionPolicyCache,
    CachedAllowedActionLLMPolicy,
    PolicyCacheMetadata,
    StateActionProbabilities,
    build_allowed_action_menu,
    load_cache_snapshot,
    save_cache_snapshot,
    serialize_rules,
    validate_cache_coverage,
)

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def make_metadata():
    return PolicyCacheMetadata(
        schema_version="1.0",
        model_repository="Qwen/Qwen3-0.6B",
        model_revision="model-sha",
        tokenizer_repository="Qwen/Qwen3-0.6B",
        tokenizer_revision="tokenizer-sha",
        prompt_version="phase2-allowed-action-v1",
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


METADATA_MUTATIONS = [
    ("schema_version", "2.0"),
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


def test_coverage_validation_rejects_missing_states():
    root = State(values=(1, 2), target=3)
    missing = State(values=(1, 3), target=4)
    snapshot = AllowedActionPolicyCache(
        metadata=make_metadata(),
        records={root: make_record(root)},
    )
    with pytest.raises(PolicyValidationError):
        validate_cache_coverage(snapshot, expected_states={root, missing})


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


def test_cached_policy_rejects_different_allowed_action_set():
    root = State(values=(1, 2), target=3)
    record = make_record(root)
    policy = CachedAllowedActionLLMPolicy({root: record})
    with pytest.raises(PolicyValidationError):
        policy.action_probs(root, allowed_actions(root)[:-1])
```

- [ ] **Step 2: Add explicit cache dataclasses and file name**

Add these constants and dataclasses:

```python
from pathlib import Path

POLICY_CACHE_SCHEMA_VERSION = "1.0"
ALLOWED_ACTION_POLICY_CACHE_NAME = "allowed_action_policy_cache.json"


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
```

Metadata equality is the invalidation rule. Resume is allowed only when the
loaded metadata equals the expected metadata exactly. This intentionally makes
device, dtype, PyTorch version, and Transformers version invalidating because
each may alter the model's floating-point probabilities. Sampling seeds are
excluded because they affect only downstream Monte Carlo and trajectory
selection.

- [ ] **Step 3: Implement strict snapshot serialization and resumability**

Use canonical JSON:

```python
def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
```

The snapshot must have this top-level schema:

```json
{
  "metadata": {
    "schema_version": "1.0",
    "model_repository": "Qwen/Qwen3-0.6B",
    "model_revision": "resolved-commit-sha"
  },
  "records": [
    {
      "state": {"target": 245, "values": [3, 7, 10, 25]},
      "menu": [
        {
          "label": "A",
          "action": {"operator": "+", "left": 3, "right": 7, "result": 10},
          "probability": 0.05
        }
      ]
    }
  ]
}
```

Implement deserialization through `Operator(symbol)` and the existing
`Action` constructor so invalid operators or operand order fail through domain
validation. Validate finite/nonnegative probabilities and the complete action
set on every read. Reject duplicate states while decoding the `records` array.

`load_cache_snapshot(path, expected_metadata=None)` loads the complete file and
optionally enforces exact metadata equality. `save_cache_snapshot(path,
snapshot)` sorts records by `(target, len(values), values)`, writes one
canonical temporary JSON file, and atomically replaces `path` with
`Path.replace`. The scorer updates an in-memory records dictionary and calls
`save_cache_snapshot` after each batch. `validate_cache_coverage` compares the
record-state keys with the exact nonterminal DAG state set before evaluation.

- [ ] **Step 4: Implement the cache-backed policy**

Add:

```python
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
```

`state_action_probabilities` is the substantive cache operation: it performs
lookup and action-set validation while returning the complete menu and
probabilities. `action_probs` remains only as the adapter required by the
shared `Policy` protocol.

- [ ] **Step 5: Run cache tests and the full fast suite**

Run:

```bash
uv run pytest tests/test_allowed_action_llm_policy.py -q
uv run pytest -q
```

Expected: all tests pass.

---

### Task 4: Hugging Face Allowed-Action Scorer and Throughput Probe

**Files:**
- Create: `inference/huggingface_allowed_action_scorer.py`
- Create: `tests/test_huggingface_allowed_action_scorer.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: menu/message builders and `StateActionProbabilities`.
- Produces: `HuggingFaceAllowedActionScorer`, `ScorerMetadata`,
  `ValidatedLabelInventory`, `ThroughputBenchmark`, and
  `ThroughputMeasurement`.

**Post-review implementation decision:** The helper-oriented sketches below
were consolidated during review. `ValidatedLabelInventory` owns tokenizer
validation and menu-token lookup. The scorer owns device selection and masked
softmax. `benchmark()` owns representative-state selection and synchronization
and returns `ThroughputBenchmark(states, measurements)`. The final module-level
interface contains no standalone pipeline-step helpers.

On MPS, copy only the selected option logits to CPU before converting to
float64 for softmax; MPS itself does not support float64 tensors.

- [ ] **Step 1: Add current runtime dependencies through uv**

Run:

```bash
uv add torch transformers huggingface-hub
```

Expected: `pyproject.toml` lists all three directly imported packages and
`uv.lock` updates.

Add a pytest marker:

```toml
markers = [
    "slow: marks tests as slow",
    "model: requires downloading and loading an external model checkpoint",
]
```

Real-model tests must carry both `@pytest.mark.slow` and
`@pytest.mark.model`, and skip unless `RUN_MODEL_TESTS=1`.

- [ ] **Step 2: Write tokenizer-label and masked-softmax tests**

Create `tests/test_huggingface_allowed_action_scorer.py` with fakes that do not
download a model:

```python
import math
import os

import pytest
import torch

from inference.huggingface_allowed_action_scorer import (
    CANDIDATE_LABELS,
    ValidatedLabelInventory,
    label_token_ids_for_menu,
    masked_label_probabilities,
    select_representative_benchmark_states,
    validate_single_token_labels,
)
from env.countdown import State, allowed_actions
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


def test_label_inventory_has_at_least_45_unique_single_tokens():
    mapping = {label: index + 1 for index, label in enumerate(CANDIDATE_LABELS)}
    inventory = validate_single_token_labels(FakeTokenizer(mapping), messages=())
    assert len(inventory.labels) >= 45
    assert len(set(inventory.token_ids)) == len(inventory.token_ids)


def test_invalid_labels_are_excluded_from_inventory():
    mapping = {label: index + 1 for index, label in enumerate(CANDIDATE_LABELS)}
    mapping[CANDIDATE_LABELS[0]] = [1, 2]
    mapping[CANDIDATE_LABELS[1]] = mapping[CANDIDATE_LABELS[2]]
    tokenizer = FakeTokenizer(mapping)
    inventory = validate_single_token_labels(tokenizer, messages=())
    assert CANDIDATE_LABELS[0] not in inventory.labels
    assert CANDIDATE_LABELS[1] not in inventory.labels
    assert len(inventory.labels) >= 45


def test_label_is_excluded_when_token_decodes_to_different_text():
    mapping = {label: index + 1 for index, label in enumerate(CANDIDATE_LABELS)}
    tokenizer = FakeTokenizer(mapping)
    tokenizer.decoded_by_token[mapping[CANDIDATE_LABELS[0]]] = (
        " " + CANDIDATE_LABELS[0]
    )
    inventory = validate_single_token_labels(tokenizer, messages=())
    assert CANDIDATE_LABELS[0] not in inventory.labels


def test_label_validation_fails_when_fewer_than_45_survive():
    mapping = {
        label: (index + 1 if index < 44 else [index + 1, index + 100])
        for index, label in enumerate(CANDIDATE_LABELS)
    }
    with pytest.raises(ValueError, match="need 45"):
        validate_single_token_labels(FakeTokenizer(mapping), messages=())


def test_menu_label_order_matches_token_id_order():
    state = State(values=(1, 2), target=3)
    inventory = ValidatedLabelInventory(
        labels=("A", "B", "C"),
        token_ids=(11, 12, 13),
    )
    menu = build_allowed_action_menu(
        state, allowed_actions(state), inventory.labels
    )
    assert label_token_ids_for_menu(menu, inventory) == (11, 12, 13)

    mismatched = (
        MenuOption(label="B", action=menu[0].action),
        *menu[1:],
    )
    with pytest.raises(ValueError):
        label_token_ids_for_menu(mismatched, inventory)


def test_benchmark_subset_is_fixed_small_and_spans_input():
    states = tuple(State(values=(1, value), target=999) for value in range(2, 22))
    selected = select_representative_benchmark_states(states)
    assert len(selected) == 8
    assert selected[0] == states[0]
    assert selected[-1] == states[-1]


def test_masked_softmax_uses_only_allowed_label_logits():
    logits = torch.tensor([0.0, 1.0, 2.0, 100.0])
    probabilities = masked_label_probabilities(
        logits, label_token_ids=(0, 1, 2), temperature=1.0
    )
    expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)
    assert probabilities == pytest.approx(expected.tolist())
    assert math.fsum(probabilities) == pytest.approx(1.0)
```

- [ ] **Step 3: Implement label validation and pure scoring helpers**

Create `inference/huggingface_allowed_action_scorer.py` beginning with:

```python
"""Batched Hugging Face scoring for the stationary allowed-action LLM policy."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from string import ascii_letters, digits
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from env.countdown import DEFAULT_RULES, RuleSet, State, allowed_actions
from oracle.float_policy_evaluation import PROBABILITY_TOLERANCE
from policies.allowed_action_llm_policy import (
    MAX_ALLOWED_ACTIONS,
    MenuOption,
    StateActionProbabilities,
    build_allowed_action_menu,
    build_policy_messages,
)

CANDIDATE_LABELS = tuple(ascii_letters + digits)


@dataclass(frozen=True, slots=True)
class ValidatedLabelInventory:
    labels: tuple[str, ...]
    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ThroughputMeasurement:
    batch_size: int
    state_count: int
    elapsed_seconds: float
    states_per_second: float


def validate_single_token_labels(tokenizer, messages) -> ValidatedLabelInventory:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    labels: list[str] = []
    token_ids: list[int] = []
    for label in CANDIDATE_LABELS:
        extended = tokenizer.encode(prompt + label, add_special_tokens=False)
        if extended[:-1] != prompt_ids or len(extended) != len(prompt_ids) + 1:
            continue
        token_id = extended[-1]
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
            f"Tokenizer exposes {len(labels)} valid labels; need {MAX_ALLOWED_ACTIONS}"
        )
    return ValidatedLabelInventory(tuple(labels), tuple(token_ids))


def label_token_ids_for_menu(
    menu: Sequence[MenuOption],
    inventory: ValidatedLabelInventory,
) -> tuple[int, ...]:
    expected_labels = inventory.labels[: len(menu)]
    actual_labels = tuple(option.label for option in menu)
    if actual_labels != expected_labels:
        raise ValueError(
            f"Menu labels {actual_labels!r} do not match token order "
            f"{expected_labels!r}"
        )
    return inventory.token_ids[: len(menu)]


def select_representative_benchmark_states(
    states: Sequence[State],
    count: int = 8,
) -> tuple[State, ...]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if len(states) <= count:
        return tuple(states)
    if count == 1:
        return (states[0],)
    indices = [
        round(index * (len(states) - 1) / (count - 1))
        for index in range(count)
    ]
    return tuple(states[index] for index in indices)


def masked_label_probabilities(
    logits: torch.Tensor,
    label_token_ids: Sequence[int],
    temperature: float,
) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and greater than zero")
    selected = logits[list(label_token_ids)].float() / temperature
    probabilities = torch.softmax(selected, dim=0).cpu().tolist()
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("Model produced malformed label probabilities")
    return probabilities
```

The integration test may reveal that Qwen's answer boundary requires a leading
space before labels. If so, change `CANDIDATE_LABELS` once, document it in the
cache metadata, and keep the contextual tokenizer test as the authority. Do
not add a silent fallback path.

- [ ] **Step 4: Implement model loading and batched scoring**

Add `from huggingface_hub import model_info` and implement:

```python
def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"Unsupported device: {requested}")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class HuggingFaceAllowedActionScorer:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        model_repository: str,
        resolved_revision: str,
        device: str,
        dtype: torch.dtype,
        temperature: float,
        label_inventory: ValidatedLabelInventory,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model_repository = model_repository
        self._resolved_revision = resolved_revision
        self._device = device
        self._dtype = dtype
        self._temperature = temperature
        self._label_inventory = label_inventory

    @classmethod
    def load(
        cls,
        model_repository: str,
        *,
        device: str = "auto",
        temperature: float = 1.0,
    ) -> "HuggingFaceAllowedActionScorer":
        resolved_device = resolve_device(device)
        dtype = torch.float16 if resolved_device in {"mps", "cuda"} else torch.float32
        resolved_revision = model_info(model_repository).sha
        tokenizer = AutoTokenizer.from_pretrained(
            model_repository, revision=resolved_revision
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_repository,
            revision=resolved_revision,
            torch_dtype=dtype,
        ).to(resolved_device)
        model.eval()

        probe_state = State(values=(1, 2), target=3)
        probe_menu = build_allowed_action_menu(
            probe_state,
            allowed_actions(probe_state),
            CANDIDATE_LABELS,
            menu_ordering_id="label-validation",
        )
        probe_messages = build_policy_messages(probe_state, probe_menu)
        inventory = validate_single_token_labels(tokenizer, probe_messages)
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_repository=model_repository,
            resolved_revision=resolved_revision,
            device=resolved_device,
            dtype=dtype,
            temperature=temperature,
            label_inventory=inventory,
        )

    @property
    def model_revision(self) -> str:
        return self._resolved_revision

    @property
    def tokenizer_revision(self) -> str:
        return self._resolved_revision

    @property
    def label_inventory(self) -> ValidatedLabelInventory:
        return self._label_inventory

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype_name(self) -> str:
        return str(self._dtype).removeprefix("torch.")

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
        records: list[StateActionProbabilities] = []
        for start in range(0, len(states), batch_size):
            state_batch = states[start : start + batch_size]
            menus = [
                build_allowed_action_menu(
                    state,
                    allowed_actions(state, rules),
                    self._label_inventory.labels,
                    menu_ordering_id=menu_ordering_id,
                    rules=rules,
                )
                for state in state_batch
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
            encoded = {
                name: tensor.to(self._device) for name, tensor in encoded.items()
            }
            with torch.inference_mode():
                logits = self._model(**encoded).logits[:, -1, :]

            for state, menu, state_logits in zip(
                state_batch, menus, logits, strict=True
            ):
                token_ids = label_token_ids_for_menu(
                    menu, self._label_inventory
                )
                values = masked_label_probabilities(
                    state_logits, token_ids, self._temperature
                )
                probabilities = {
                    option.action: probability
                    for option, probability in zip(menu, values, strict=True)
                }
                if (
                    abs(math.fsum(probabilities.values()) - 1.0)
                    > PROBABILITY_TOLERANCE
                ):
                    raise ValueError(f"Probabilities do not sum to one at {state!r}")
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
    ) -> tuple[ThroughputMeasurement, ...]:
        benchmark_states = select_representative_benchmark_states(states)
        measurements: list[ThroughputMeasurement] = []
        for batch_size in batch_sizes:
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
                _synchronize(self._device)
                started = time.perf_counter()
                self.score_states(
                    benchmark_states,
                    batch_size=batch_size,
                    menu_ordering_id=menu_ordering_id,
                    rules=rules,
                )
                _synchronize(self._device)
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
        return tuple(measurements)
```

Implementation requirements:

1. Resolve `model_repository` to an immutable Hugging Face commit SHA before
   loading; load model and tokenizer from that SHA.
2. Device `auto` selects CUDA, then MPS, then CPU.
3. Use float16 on MPS/CUDA and float32 on CPU.
4. Set tokenizer padding to the left and use the EOS token as padding when no
   pad token exists.
5. Render each message tuple with `enable_thinking=False`.
6. Run under `torch.inference_mode()`.
7. Read `outputs.logits[:, -1, :]`.
8. Assign the first `len(menu)` validated labels to that state's deterministic
   menu and return one probability per action.
9. Verify every result sums to one within `1e-6`.
10. Benchmark with warm-up batches and synchronize CUDA before timing.
11. If a batch size raises an out-of-memory exception, record no result for
    that and larger sizes; do not crash after smaller measurements succeeded.

- [ ] **Step 5: Add the opt-in real-model smoke test**

```python
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
        [state], batch_size=1, menu_ordering_id="integration-test"
    )[0]
    assert set(record.probabilities) == set(allowed_actions(state))
    assert math.fsum(record.probabilities.values()) == pytest.approx(1.0)
```

- [ ] **Step 6: Register package and scorer tests**

Add `"inference"` to `PACKAGES` in `tests/test_bootstrap.py`.

Run:

```bash
uv run pytest tests/test_huggingface_allowed_action_scorer.py tests/test_bootstrap.py -q
uv run pytest -q
```

Expected: fake tests and all fast tests pass; the real-model test is skipped.

---

### Task 5: Evaluation CLI, Monte Carlo Check, and Mixed-Credit Artifacts

**Files:**
- Create: `experiments/evaluate_allowed_action_llm_policy.py`
- Create: `tests/test_evaluate_allowed_action_llm_policy.py`
- Modify: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: scorer, cache functions, `FloatPolicyEvaluator`, and existing DAG/environment functions.
- Produces: `score` and `evaluate` CLI commands plus `throughput_benchmark.json`, `policy_evaluation_summary.json`, `mixed_credit_trajectory.json`, and `mixed_credit_trajectory.md`.

- [ ] **Step 1: Write rollout and selection tests with an in-memory policy**

Create `tests/test_evaluate_allowed_action_llm_policy.py` with:

```python
import pytest

from env.countdown import Action, Operator, State
from experiments import evaluate_allowed_action_llm_policy as experiment
from experiments.evaluate_allowed_action_llm_policy import (
    EvaluatedTrajectory,
    TrajectoryStep,
    monte_carlo_success_rate,
    select_mixed_credit_trajectory,
    trajectory_to_markdown,
)
from oracle.evaluation import build_reasoning_dag
from oracle.float_policy_evaluation import FloatPolicyEvaluator


class FloatUniformPolicy:
    def action_probs(self, state, actions):
        return {action: 1.0 / len(actions) for action in actions}


def make_fake_trajectory(advantages, *, reward, sample_index):
    state = State(values=(1, 2), target=3)
    next_state = State(values=(3,), target=3)
    action = Action(Operator.ADD, 1, 2)
    steps = tuple(
        TrajectoryStep(
            state=state,
            action=action,
            action_probability=1.0,
            next_state=next_state,
            v_pi=0.5,
            q_pi=0.5 + advantage,
            advantage=advantage,
        )
        for advantage in advantages
    )
    return EvaluatedTrajectory(
        steps=steps,
        final_state=next_state,
        final_reward=reward,
        sample_index=sample_index,
    )


def test_monte_carlo_is_seed_deterministic():
    root = State(values=(1, 2), target=3)
    first = monte_carlo_success_rate(
        root, FloatUniformPolicy(), rollout_count=1000, seed=123
    )
    second = monte_carlo_success_rate(
        root, FloatUniformPolicy(), rollout_count=1000, seed=123
    )
    assert first == second


def test_selection_returns_first_qualifying_trajectory(monkeypatch):
    root = State(values=(1, 2, 3), target=10)
    policy = FloatUniformPolicy()
    evaluator = FloatPolicyEvaluator(policy, build_reasoning_dag(root))
    candidates = iter(
        [
            make_fake_trajectory([0.1], reward=0, sample_index=0),
            make_fake_trajectory([0.2, -0.3], reward=0, sample_index=1),
        ]
    )
    monkeypatch.setattr(
        experiment,
        "sample_evaluated_trajectory",
        lambda *args, **kwargs: next(candidates),
    )
    selected = select_mixed_credit_trajectory(
        root,
        policy,
        evaluator,
        seed=20260815,
        max_attempts=10_000,
    )
    assert selected.sample_index == 1


def test_markdown_labels_outcome_credit_as_display_transformation():
    trajectory = make_fake_trajectory([0.2, -0.3], reward=0, sample_index=1)
    markdown = trajectory_to_markdown(trajectory)
    assert "Outcome-only display credit" in markdown
    assert "environment reward remains 0/1" in markdown
```

- [ ] **Step 2: Implement trajectory and Monte Carlo primitives**

Create `experiments/evaluate_allowed_action_llm_policy.py` with these data
types and signatures:

```python
@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    state: State
    action: Action
    action_probability: float
    next_state: State
    v_pi: float
    q_pi: float
    advantage: float


@dataclass(frozen=True, slots=True)
class EvaluatedTrajectory:
    steps: tuple[TrajectoryStep, ...]
    final_state: State
    final_reward: int
    sample_index: int


def sample_evaluated_trajectory(
    root: State,
    policy: Policy[float],
    evaluator: FloatPolicyEvaluator,
    rng: random.Random,
    sample_index: int,
) -> EvaluatedTrajectory:
    state = root
    steps: list[TrajectoryStep] = []
    while not is_terminal(state):
        actions = allowed_actions(state)
        probabilities = policy.action_probs(state, actions)
        action = rng.choices(
            actions,
            weights=[probabilities[candidate] for candidate in actions],
            k=1,
        )[0]
        next_state = transition(state, action)
        steps.append(
            TrajectoryStep(
                state=state,
                action=action,
                action_probability=probabilities[action],
                next_state=next_state,
                v_pi=evaluator.v_pi(state),
                q_pi=evaluator.q_pi(state, action),
                advantage=evaluator.a_pi(state, action),
            )
        )
        state = next_state
    return EvaluatedTrajectory(
        steps=tuple(steps),
        final_state=state,
        final_reward=reward(state),
        sample_index=sample_index,
    )


def monte_carlo_success_rate(
    root: State,
    policy: Policy[float],
    *,
    rollout_count: int,
    seed: int,
) -> float:
    if rollout_count < 1:
        raise ValueError("rollout_count must be at least 1")
    rng = random.Random(seed)
    successes = 0
    for _ in range(rollout_count):
        state = root
        while not is_terminal(state):
            actions = allowed_actions(state)
            probabilities = policy.action_probs(state, actions)
            action = rng.choices(
                actions,
                weights=[probabilities[candidate] for candidate in actions],
                k=1,
            )[0]
            state = transition(state, action)
        successes += reward(state)
    return successes / rollout_count


def select_mixed_credit_trajectory(
    root: State,
    policy: Policy[float],
    evaluator: FloatPolicyEvaluator,
    *,
    seed: int,
    max_attempts: int,
) -> EvaluatedTrajectory:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    rng = random.Random(seed)
    for sample_index in range(max_attempts):
        trajectory = sample_evaluated_trajectory(
            root, policy, evaluator, rng, sample_index
        )
        advantages = [step.advantage for step in trajectory.steps]
        if (
            trajectory.final_reward == 0
            and any(value > 1e-9 for value in advantages)
            and any(value < -1e-9 for value in advantages)
        ):
            return trajectory
    raise RuntimeError(
        f"No failed mixed-credit trajectory found in {max_attempts} attempts"
    )
```

Use `rng.choices(actions, weights=probabilities, k=1)[0]`, transition until
terminal, and record evaluator values before each transition. Selection returns
the first failed trajectory with at least one advantage above `1e-9` and one
below `-1e-9`; otherwise raise a descriptive `RuntimeError` after exactly
`max_attempts`.

- [ ] **Step 3: Implement canonical JSON and Markdown artifact builders**

Use these file names:

```python
THROUGHPUT_FILE = "throughput_benchmark.json"
EVALUATION_FILE = "policy_evaluation_summary.json"
TRAJECTORY_JSON_FILE = "mixed_credit_trajectory.json"
TRAJECTORY_MARKDOWN_FILE = "mixed_credit_trajectory.md"
```

The trajectory JSON records the complete menu at every state, selected action,
selected probability, resulting state, `Vπ/Qπ/Aπ`, final reward, and uniform
outcome-only display credit (`+1` on success, `-1` on failure). The Markdown
must state:

```text
The environment reward remains 0/1. “Outcome-only display credit” maps the
final outcome to +1/-1 only for comparison with per-action Aπ. This trajectory
was selected by the predeclared seed-and-first-match rule and does not estimate
population frequency.
```

Serialize JSON with sorted keys, two-space indentation, and a trailing newline.
`policy_evaluation_summary.json` records the CLI seed, Monte Carlo rollout
count, artifact-attempt limit, numerical root value, observed success rate,
standard error, and acceptance margin. The seed belongs here rather than in
`PolicyCacheMetadata` because it affects only downstream sampling.

- [ ] **Step 4: Implement `score` and `evaluate` subcommands**

The parser must expose:

```bash
uv run python -m experiments.evaluate_allowed_action_llm_policy score --help
uv run python -m experiments.evaluate_allowed_action_llm_policy evaluate --help
```

`score` arguments:

```text
--numbers 3 7 10 25
--target 245
--output artifacts/phase2/qwen3-0.6b-worked-example
--model Qwen/Qwen3-0.6B
--device auto
--temperature 1.0
--menu-ordering-id phase2-v1
--batch-size 8
```

`score` must:

1. build the DAG and collect its canonical nonterminal states;
2. load the scorer and construct the expected cache metadata;
3. validate or create the cache snapshot;
4. select eight states evenly across canonical nonterminal-state order and run
   the throughput probe for batch sizes `1,4,8`;
5. write `throughput_benchmark.json`;
6. estimate full scoring time as
   `103 / max(measurement.states_per_second)` and refuse to continue if it
   exceeds 30 minutes;
7. add each completed batch to the in-memory record dictionary and atomically
   rewrite the snapshot;
8. require exact nonterminal-state coverage before returning success.

`evaluate` takes only `--output`, `--mc-rollouts` (default `20000`),
`--artifact-attempts` (default `10000`), and `--seed` (default `20260815`).
It must not import or load Transformers model classes. It loads the cache
metadata, probability records, and root problem; validates all 103 expected nonterminal
states; computes `FloatPolicyEvaluator`; checks Monte Carlo agreement using:

```python
standard_error = (
    math.sqrt(root_value * (1.0 - root_value) / rollout_count)
    if 0.0 < root_value < 1.0
    else 1e-9
)
margin = 5.0 * standard_error + 1e-6
if abs(success_rate - root_value) > margin:
    raise RuntimeError("Monte Carlo success rate disagrees with numerical Vpi")
```

Then it writes the summary and selected trajectory artifacts.

- [ ] **Step 5: Register and test the CLI**

Add `"experiments.evaluate_allowed_action_llm_policy"` to
`EXPERIMENT_MODULES` in `tests/test_bootstrap.py`.

Add subprocess tests asserting top-level, `score`, and `evaluate` help return
zero without loading model weights.

Run:

```bash
uv run pytest tests/test_evaluate_allowed_action_llm_policy.py tests/test_bootstrap.py -q
uv run pytest -q
```

Expected: all tests pass.

---

### Task 6: Real Qwen Integration and Worked-Example Artifacts

**Files:**
- Create: `artifacts/phase2/qwen3-0.6b-worked-example/allowed_action_policy_cache.json`
- Create: `artifacts/phase2/qwen3-0.6b-worked-example/throughput_benchmark.json`
- Create: `artifacts/phase2/qwen3-0.6b-worked-example/policy_evaluation_summary.json`
- Create: `artifacts/phase2/qwen3-0.6b-worked-example/mixed_credit_trajectory.json`
- Create: `artifacts/phase2/qwen3-0.6b-worked-example/mixed_credit_trajectory.md`

**Interfaces:**
- Consumes: completed `score` and `evaluate` commands.
- Produces: the empirical Phase 2 deliverable.

- [ ] **Step 1: Run the real-model smoke test on Apple Silicon**

Run:

```bash
RUN_MODEL_TESTS=1 uv run pytest tests/test_huggingface_allowed_action_scorer.py -m model -q
```

Expected: Qwen loads on MPS (or the explicitly selected available device), the
tiny state returns all allowed actions, and probabilities sum to one.

If the tokenizer-label contextual test fails, make the one explicit
`CANDIDATE_LABELS` correction described in Task 4, add a regression test for
the actual boundary behavior, and rerun the complete fast suite before
continuing.

- [ ] **Step 2: Score the complete worked DAG**

Run:

```bash
uv run python -m experiments.evaluate_allowed_action_llm_policy score \
  --numbers 3 7 10 25 \
  --target 245 \
  --output artifacts/phase2/qwen3-0.6b-worked-example \
  --model Qwen/Qwen3-0.6B \
  --device mps \
  --temperature 1.0 \
  --menu-ordering-id phase2-v1 \
  --batch-size 8
```

Expected:

- the cache metadata records immutable resolved model/tokenizer SHAs;
- the benchmark records at least one successful batch size;
- the estimate is at most 30 minutes;
- exactly 103 nonterminal state records are present.

If MPS cannot meet the gate, stop. Record measured states/second, estimate a
single CUDA EC2 run from the same 103-state count, and obtain budget approval
before launching cloud compute.

- [ ] **Step 3: Evaluate from cache without loading Qwen**

Run:

```bash
uv run python -m experiments.evaluate_allowed_action_llm_policy evaluate \
  --output artifacts/phase2/qwen3-0.6b-worked-example \
  --mc-rollouts 20000 \
  --artifact-attempts 10000 \
  --seed 20260815
```

Expected:

- numerical invariants pass;
- Monte Carlo success agrees with `Vπ(root)` inside the five-standard-error
  margin;
- a failed mixed-sign trajectory is found under the predeclared rule;
- JSON and Markdown artifacts are written.

If no qualifying trajectory is found, stop and report that Phase 2's approved
artifact criterion was not met. Do not select another problem or seed without
a design revision.

- [ ] **Step 4: Verify cache-only reproducibility**

Disable Hugging Face network access and rerun only `evaluate`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python -m experiments.evaluate_allowed_action_llm_policy evaluate \
  --output artifacts/phase2/qwen3-0.6b-worked-example \
  --mc-rollouts 20000 \
  --artifact-attempts 10000 \
  --seed 20260815
```

The command must succeed without constructing
`HuggingFaceAllowedActionScorer`. Compare hashes:

```bash
shasum -a 256 \
  artifacts/phase2/qwen3-0.6b-worked-example/policy_evaluation_summary.json \
  artifacts/phase2/qwen3-0.6b-worked-example/mixed_credit_trajectory.json \
  artifacts/phase2/qwen3-0.6b-worked-example/mixed_credit_trajectory.md
```

Run `evaluate` again and repeat the hash command. Expected: all three hashes
are unchanged.

- [ ] **Step 5: Run complete verification**

Run:

```bash
uv run pytest -q
uv run pytest -q -m slow
git diff --check
```

Expected: the fast suite and existing slow Monte Carlo tests pass; no whitespace
errors.

---

### Task 6.5: Five-Ordering Menu Sensitivity

**Files:**
- Create: `experiments/compare_menu_ordering_sensitivity.py`
- Create: `tests/test_compare_menu_ordering_sensitivity.py`
- Modify: `tests/test_bootstrap.py`
- Create: `artifacts/phase2/menu-ordering-sensitivity/phase2-sensitivity-{1,2,3,4}/allowed_action_policy_cache.json`
- Create: `artifacts/phase2/menu-ordering-sensitivity/menu_ordering_sensitivity.json`
- Create: `artifacts/phase2/menu-ordering-sensitivity/menu_ordering_sensitivity.md`

**Interfaces:**
- Consumes: named `AllowedActionPolicyCache` paths and the existing
  `FloatPolicyEvaluator`.
- Produces: `compare_menu_ordering_caches(named_cache_paths) -> dict`,
  canonical JSON, and blog-ready Markdown.

- [ ] **Step 1: Write comparison tests**

Create synthetic caches for the same `(1,2)→3` DAG with different menu orderings
and semantic action probabilities. Tests must assert:

```python
def test_comparison_matches_actions_across_different_menu_orders(tmp_path):
    comparison = compare_menu_ordering_caches(
        {
            "ordering-a": cache_a_path,
            "ordering-b": cache_b_path,
        }
    )
    assert comparison["menu_orderings"]["ordering-a"]["root_v_pi"] == pytest.approx(0.5)
    assert comparison["menu_orderings"]["ordering-b"]["root_v_pi"] == pytest.approx(0.25)
    assert comparison["pairwise_total_variation"][0]["distance"] == pytest.approx(
        0.25
    )


def test_comparison_rejects_different_roots_or_state_coverage(tmp_path):
    with pytest.raises(PolicyValidationError):
        compare_menu_ordering_caches(
            {"ordering-a": complete_cache_path, "ordering-b": incompatible_cache_path}
        )


def test_markdown_reports_winning_action_and_label_bias():
    markdown = comparison_to_markdown(comparison_fixture)
    assert "Winning root action" in markdown
    assert "Average probability by option label" in markdown
```

Do not run a separate missing-module failure command.

- [ ] **Step 2: Implement the focused comparison module**

Create `experiments/compare_menu_ordering_sensitivity.py` with these exact public
interfaces:

- `compare_menu_ordering_caches(named_cache_paths: Mapping[str, Path]) -> dict[str, object]`
- `comparison_to_markdown(comparison: Mapping[str, object]) -> str`
- `main(argv: Sequence[str] | None = None) -> None`

Implementation requirements:

1. Require at least two uniquely named caches.
2. Require identical root problem, model/tokenizer revisions, prompt version,
   temperature, rules, device, dtype, software versions, and state coverage.
   Only `menu_ordering_id`, menu ordering, and probabilities may differ.
3. Build the DAG once and compute the unique solution-preserving root action
   from exact `V*`/`Q*`; fail if it is not unique.
4. Evaluate each cache with `FloatPolicyEvaluator` and
   `validate_numerical_policy_invariants`.
5. Match root probability vectors by canonical `Action.sort_key()`, never by
   option label.
6. Report pairwise total-variation distance:
   `0.5 * sum(abs(p_i - q_i))`.
7. For each label, report appearance count and mean probability across all
   cached nonterminal-state menus.
8. Write sorted, indented JSON with a trailing newline and deterministic
   Markdown.
9. Expose repeated CLI arguments in the form
   `--cache NAME=PATH`, plus `--output`.

- [ ] **Step 3: Register and verify the comparison CLI**

Add `"experiments.compare_menu_ordering_sensitivity"` to
`EXPERIMENT_MODULES` in `tests/test_bootstrap.py`.

Run:

```bash
uv run pytest tests/test_compare_menu_ordering_sensitivity.py tests/test_bootstrap.py -q
```

Expected: all focused tests pass.

- [ ] **Step 4: Score four additional complete-DAG policies**

Run the existing scorer once per new menu ordering:

```bash
for index in 1 2 3 4; do
  uv run python -m experiments.evaluate_allowed_action_llm_policy score \
    --numbers 3 7 10 25 \
    --target 245 \
    --output "artifacts/phase2/menu-ordering-sensitivity/phase2-sensitivity-${index}" \
    --model Qwen/Qwen3-0.6B \
    --device mps \
    --temperature 1.0 \
    --menu-ordering-id "phase2-sensitivity-${index}" \
    --batch-size 8
done
```

Expected: every cache contains exactly 103 nonterminal states and records the
requested menu ordering ID.

- [ ] **Step 5: Generate the canonical sensitivity comparison**

Run:

```bash
uv run python -m experiments.compare_menu_ordering_sensitivity \
  --cache phase2-v1=artifacts/phase2/qwen3-0.6b-worked-example/allowed_action_policy_cache.json \
  --cache phase2-sensitivity-1=artifacts/phase2/menu-ordering-sensitivity/phase2-sensitivity-1/allowed_action_policy_cache.json \
  --cache phase2-sensitivity-2=artifacts/phase2/menu-ordering-sensitivity/phase2-sensitivity-2/allowed_action_policy_cache.json \
  --cache phase2-sensitivity-3=artifacts/phase2/menu-ordering-sensitivity/phase2-sensitivity-3/allowed_action_policy_cache.json \
  --cache phase2-sensitivity-4=artifacts/phase2/menu-ordering-sensitivity/phase2-sensitivity-4/allowed_action_policy_cache.json \
  --output artifacts/phase2/menu-ordering-sensitivity
```

Expected: JSON and Markdown report all five menu orderings, root values, winning-action
probabilities, top root choices, pairwise total-variation distances, and label
probability summaries.

- [ ] **Step 6: Verify reproducibility and regressions**

Hash both summary files, rerun the comparison command, and confirm identical
hashes. Then run:

```bash
uv run pytest -q
uv run pytest -q -m slow
```

Do not run `git add` or `git commit`; the user owns repository staging.

---

### Task 6.6: Literature-Style Countdown Capability Check

**Files:**
- Create: `data/countdown-capability/tinyzero-test-12.jsonl`
- Create: `experiments/evaluate_countdown_capability.py`
- Create: `tests/test_evaluate_countdown_capability.py`
- Modify: `policies/allowed_action_llm_policy.py`
- Modify: `tests/test_allowed_action_llm_policy.py`
- Modify: `tests/test_bootstrap.py`

**Interfaces:**
- Exact mode keeps returning one allowed-action label with thinking disabled.
- Capability mode generates complete expressions and reports accuracy,
  Pass@32, All-correct@32, format rate, and valid-equation rate.

- [ ] Freeze TinyZero source rows `327680` through `327691` with source indices.
- [ ] Expand the exact-credit prompt with the multi-step objective, transition
  rules, and explicit future-move instruction; bump `PROMPT_VERSION`.
- [ ] Implement a restricted AST verifier for `+`, `-`, `*`, `/`, parentheses,
  exact number usage, and exact rational evaluation.
- [ ] Implement resumable model generation with 32 samples per problem,
  temperature `0.6`, and `max_new_tokens=1024`.
- [ ] Register `experiments.evaluate_countdown_capability` in bootstrap tests.
- [ ] Run a one-problem/two-sample smoke before the complete evaluation.

Full 4B command:

```bash
uv run python -m experiments.evaluate_countdown_capability \
  --model Qwen/Qwen3.5-4B \
  --output artifacts/phase2/capability/qwen3.5-4b \
  --samples-per-problem 32 \
  --temperature 0.6 \
  --max-new-tokens 1024 \
  --generation-batch-size 1 \
  --device mps
```

Repeat with `Qwen/Qwen3.5-9B` and a separate output directory only after the 4B
run completes. Do not use capability outputs as exact-credit caches.

---

### Task 7: Research Log and Phase 2 Documentation

**Files:**
- Modify: `README.md`
- Modify: `research/experiment_log.md`

**Interfaces:**
- Consumes: committed Phase 2 cache snapshot, benchmark, summary, and trajectory.
- Produces: reproducible user-facing instructions and an E003 experiment record.

- [ ] **Step 1: Update README status and commands**

Replace the Phase 1-only status framing with:

```markdown
## Status

- Phase 1: complete — deterministic Countdown environment, exact oracle,
  exploratory profile, and frozen dataset.
- Phase 2: complete for the worked example — Qwen3-0.6B evaluated as a
  stationary allowed-action policy over the full `(3, 7, 10, 25) → 245` DAG.

The Phase 2 policy is constrained: Qwen receives only the canonical current
state and a deterministically shuffled menu, thinking is disabled, and its
next-token logits are normalized over the allowed option labels. This is not
free-form chain-of-thought behavior.
```

Add the exact `score` and `evaluate` commands from Task 6 and link:

- the design specification;
- the implementation plan;
- `policy_evaluation_summary.json`;
- `mixed_credit_trajectory.md`.

- [ ] **Step 2: Add E003 to the experiment log**

Append `E003 — Qwen3-0.6B allowed-action policy on one complete DAG` using the
existing Question / Hypothesis / Independent variable / Dependent variable /
Controls / Expected result / Actual result / Interpretation / Next experiment
format.

Populate `Actual result` only from committed files:

- device, dtype, and resolved revisions from the `metadata` object in
  `allowed_action_policy_cache.json`;
- successful batch sizes and states/second from `throughput_benchmark.json`;
- numerical root value, Monte Carlo rate, and margin from
  `policy_evaluation_summary.json`;
- selected sample index and per-step advantages from
  `mixed_credit_trajectory.json`.

Do not characterize the single selected trajectory as a frequency estimate.
State explicitly that it is an illustrative trajectory chosen by the
predeclared first-match rule.

- [ ] **Step 3: Verify every documented command and link**

Run:

```bash
uv run python -m experiments.evaluate_allowed_action_llm_policy --help
uv run python -m experiments.evaluate_allowed_action_llm_policy score --help
uv run python -m experiments.evaluate_allowed_action_llm_policy evaluate --help
uv run pytest -q
git diff --check
```

Expected: commands return zero, tests pass, and links point to existing files.

## Final Verification

- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run pytest -q -m slow`.
- [ ] Run the cache-only `evaluate` command twice and confirm artifact hashes are identical.
- [ ] Confirm `git status --short` contains no model weights or unrelated files.
- [ ] Confirm the implementation matches every success criterion in the approved design.
