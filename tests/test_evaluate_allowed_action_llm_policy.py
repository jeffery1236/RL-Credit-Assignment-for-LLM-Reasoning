"""Tests for allowed-action policy scoring, evaluation, and artifacts."""

import json
import random
import subprocess
import sys

import pytest

from env.countdown import DEFAULT_RULES, Action, Operator, State, allowed_actions
from experiments import evaluate_allowed_action_llm_policy as experiment
from experiments.evaluate_allowed_action_llm_policy import (
    EVALUATION_FILE,
    TRAJECTORY_JSON_FILE,
    TRAJECTORY_MARKDOWN_FILE,
    EvaluatedTrajectory,
    TrajectoryStep,
    main,
    monte_carlo_success_rate,
    sample_evaluated_trajectory,
    select_mixed_credit_trajectory,
    trajectory_to_dict,
    trajectory_to_markdown,
    validate_numerical_policy_invariants,
)
from inference import huggingface_allowed_action_scorer as scorer_module
from inference.huggingface_allowed_action_scorer import (
    ScorerMetadata,
    ThroughputBenchmark,
    ThroughputMeasurement,
    ValidatedLabelInventory,
)
from oracle.evaluation import build_reasoning_dag
from oracle.float_policy_evaluation import FloatPolicyEvaluator
from policies.allowed_action_llm_policy import (
    ALLOWED_ACTION_POLICY_CACHE_NAME,
    POLICY_CACHE_SCHEMA_VERSION,
    PROMPT_VERSION,
    AllowedActionPolicyCache,
    CachedAllowedActionLLMPolicy,
    MenuOption,
    PolicyCacheMetadata,
    StateActionProbabilities,
    build_allowed_action_menu,
    load_cache_snapshot,
    save_cache_snapshot,
    serialize_rules,
    serialize_state,
)

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


class FloatUniformPolicy:
    def action_probs(self, state, actions):
        return {action: 1.0 / len(actions) for action in actions}


def make_cached_uniform_policy(root):
    dag = build_reasoning_dag(root)
    records = {}
    for state in dag.topological_order():
        actions = allowed_actions(state)
        if not actions:
            continue
        menu = build_allowed_action_menu(
            state,
            actions,
            LABELS,
            menu_ordering_id="test-ordering",
        )
        probability = 1.0 / len(actions)
        records[state] = StateActionProbabilities(
            state=state,
            menu=menu,
            probabilities={action: probability for action in actions},
        )
    return CachedAllowedActionLLMPolicy(records), records


def make_fake_trajectory(advantages, *, final_reward, sample_index):
    state = State(values=(1, 2), target=3)
    next_state = State(values=(3,), target=3)
    action = Action(Operator.ADD, 1, 2)
    menu = (MenuOption(label="A", action=action),)
    steps = tuple(
        TrajectoryStep(
            state=state,
            menu=menu,
            action_probabilities={action: 1.0},
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
        final_reward=final_reward,
        sample_index=sample_index,
    )


def test_monte_carlo_is_seed_deterministic():
    root = State(values=(1, 2), target=3)

    first = monte_carlo_success_rate(
        root,
        FloatUniformPolicy(),
        rollout_count=1000,
        seed=123,
    )
    second = monte_carlo_success_rate(
        root,
        FloatUniformPolicy(),
        rollout_count=1000,
        seed=123,
    )

    assert first == second


def test_uniform_policy_satisfies_numerical_invariants():
    root = State(values=(1, 2, 3), target=10)
    dag = build_reasoning_dag(root)
    evaluator = FloatPolicyEvaluator(FloatUniformPolicy(), dag)

    validate_numerical_policy_invariants(dag, evaluator)


def test_sampled_trajectory_records_complete_cached_menu():
    root = State(values=(1, 2), target=3)
    policy, records = make_cached_uniform_policy(root)
    evaluator = FloatPolicyEvaluator(policy, build_reasoning_dag(root))

    trajectory = sample_evaluated_trajectory(
        root,
        policy,
        evaluator,
        random.Random(5),
        sample_index=0,
    )

    assert len(trajectory.steps) == 1
    step = trajectory.steps[0]
    assert step.menu == records[root].menu
    assert step.action_probabilities == records[root].probabilities


def test_selection_returns_first_qualifying_trajectory(monkeypatch):
    root = State(values=(1, 2, 3), target=10)
    policy = FloatUniformPolicy()
    evaluator = FloatPolicyEvaluator(policy, build_reasoning_dag(root))
    candidates = iter(
        [
            make_fake_trajectory([0.1], final_reward=0, sample_index=0),
            make_fake_trajectory([0.2, -0.3], final_reward=0, sample_index=1),
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


def test_selection_raises_after_declared_attempt_limit(monkeypatch):
    root = State(values=(1, 2, 3), target=10)
    policy = FloatUniformPolicy()
    evaluator = FloatPolicyEvaluator(policy, build_reasoning_dag(root))
    calls = 0

    def sample_without_mixed_credit(*args, **kwargs):
        nonlocal calls
        trajectory = make_fake_trajectory(
            [0.1],
            final_reward=0,
            sample_index=calls,
        )
        calls += 1
        return trajectory

    monkeypatch.setattr(
        experiment,
        "sample_evaluated_trajectory",
        sample_without_mixed_credit,
    )

    with pytest.raises(RuntimeError, match="3 attempts"):
        select_mixed_credit_trajectory(
            root,
            policy,
            evaluator,
            seed=20260815,
            max_attempts=3,
        )
    assert calls == 3


def test_trajectory_artifacts_include_menu_and_credit_disclaimer():
    trajectory = make_fake_trajectory(
        [0.2, -0.3],
        final_reward=0,
        sample_index=1,
    )

    artifact = trajectory_to_dict(trajectory, seed=20260815)
    markdown = trajectory_to_markdown(trajectory)

    assert artifact["steps"][0]["menu"]
    assert artifact["steps"][0]["outcome_only_display_credit"] == -1
    assert "Outcome-only display credit" in markdown
    assert "environment reward remains 0/1" in markdown


class FakeScorer:
    metadata = ScorerMetadata(
        model_repository="fake/model",
        model_revision="fake-model-sha",
        tokenizer_repository="fake/model",
        tokenizer_revision="fake-tokenizer-sha",
        device="cpu",
        dtype="float32",
        temperature=1.0,
        label_inventory=ValidatedLabelInventory(
            labels=LABELS,
            token_ids=tuple(range(len(LABELS))),
        ),
    )

    def benchmark(self, states, *, menu_ordering_id, rules):
        benchmark_states = tuple(states[:8])
        return ThroughputBenchmark(
            states=benchmark_states,
            measurements=(
                ThroughputMeasurement(
                    batch_size=1,
                    state_count=len(benchmark_states),
                    elapsed_seconds=0.1,
                    states_per_second=100.0,
                ),
            ),
        )

    def score_states(self, states, *, batch_size, menu_ordering_id, rules):
        records = []
        for state in states:
            actions = allowed_actions(state, rules)
            menu = build_allowed_action_menu(
                state,
                actions,
                self.metadata.label_inventory.labels,
                menu_ordering_id=menu_ordering_id,
                rules=rules,
            )
            probability = 1.0 / len(actions)
            records.append(
                StateActionProbabilities(
                    state=state,
                    menu=menu,
                    probabilities={action: probability for action in actions},
                )
            )
        return tuple(records)


def test_score_command_writes_complete_cache_and_throughput(
    tmp_path,
    monkeypatch,
):
    fake_scorer = FakeScorer()
    monkeypatch.setattr(
        scorer_module.HuggingFaceAllowedActionScorer,
        "load",
        lambda *args, **kwargs: fake_scorer,
    )

    main(
        [
            "score",
            "--numbers",
            "1",
            "2",
            "--target",
            "3",
            "--output",
            str(tmp_path),
            "--model",
            "fake/model",
            "--device",
            "cpu",
            "--menu-ordering-id",
            "test-ordering",
            "--batch-size",
            "1",
        ]
    )

    snapshot = load_cache_snapshot(tmp_path / ALLOWED_ACTION_POLICY_CACHE_NAME)
    assert len(snapshot.records) == 1
    assert (tmp_path / "throughput_benchmark.json").exists()


def test_evaluate_command_writes_summary_and_mixed_credit_artifacts(tmp_path):
    root = State(values=(2, 3, 5), target=30)
    _, records = make_cached_uniform_policy(root)
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
        menu_ordering_id="test-ordering",
        temperature=1.0,
        rules=serialize_rules(DEFAULT_RULES),
        device="cpu",
        dtype="float32",
        software_versions={"torch": "test", "transformers": "test"},
        root_problem=serialize_state(root),
    )
    save_cache_snapshot(
        tmp_path / ALLOWED_ACTION_POLICY_CACHE_NAME,
        AllowedActionPolicyCache(metadata=metadata, records=records),
    )

    main(
        [
            "evaluate",
            "--output",
            str(tmp_path),
            "--mc-rollouts",
            "5000",
            "--artifact-attempts",
            "100",
            "--seed",
            "20260815",
        ]
    )

    summary = json.loads((tmp_path / EVALUATION_FILE).read_text())
    trajectory = json.loads((tmp_path / TRAJECTORY_JSON_FILE).read_text())
    assert summary["monte_carlo_check_passed"] is True
    assert summary["numerical_invariants_passed"] is True
    assert summary["artifact_found"] is True
    assert any(step["advantage"] > 0 for step in trajectory["steps"])
    assert any(step["advantage"] < 0 for step in trajectory["steps"])
    assert (tmp_path / TRAJECTORY_MARKDOWN_FILE).exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["score", "--help"],
        ["evaluate", "--help"],
    ],
)
def test_cli_help_does_not_load_model(arguments):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.evaluate_allowed_action_llm_policy",
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_importing_experiment_does_not_import_transformers():
    script = """
import sys
import experiments.evaluate_allowed_action_llm_policy
print("transformers" in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
