"""Tests for bridge selection, budget refusal, checkpointing, and light CLI imports."""

import json
import random
import subprocess
import sys
from argparse import Namespace
from itertools import combinations

import pytest

from env.countdown import State, allowed_actions
from experiments import evaluate_policy_bridge as bridge
from experiments.evaluate_policy_bridge import (
    STAGE_A_FILE,
    STAGE_B_FILE,
    load_stage_a_results,
    run_stage_a,
    run_stage_b,
    select_bridge_problems,
)
from policies.allowed_action_llm_policy import (
    StateActionProbabilities,
    build_allowed_action_menu,
)

REFERENCE_POOL_SIZE = 4
MENU_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
FAKE_MODEL_METADATA = {
    "model_repository": "Qwen/Qwen3.5-4B",
    "tokenizer_repository": "Qwen/Qwen3.5-4B",
    "device": "cpu",
    "dtype": "float32",
}


def _write_dataset(tmp_path):
    """Two four-number roots, enough to exercise depth cycling and both strata."""
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"values": [2, 3, 4, 5], "target": 14, "n_values": 4},
                {"values": [1, 2, 3, 7], "target": 15, "n_values": 4},
            )
        )
        + "\n"
    )
    return dataset


def _solve(terms, target):
    """Brute-force a project-rule expression reaching target, or None.

    The fake generator emits real solutions so scoring is exercised for real
    rather than against a hand-written answer string.
    """
    for expression, value in terms:
        if value == target:
            return expression
    for left, right in combinations(range(len(terms)), 2):
        rest = [term for index, term in enumerate(terms) if index not in (left, right)]
        (left_expression, left_value), (right_expression, right_value) = (
            terms[left],
            terms[right],
        )
        combined = [
            (f"({left_expression} + {right_expression})", left_value + right_value),
            (f"({left_expression} * {right_expression})", left_value * right_value),
        ]
        if left_value > right_value:
            combined.append(
                (f"({left_expression} - {right_expression})", left_value - right_value)
            )
        elif right_value > left_value:
            combined.append(
                (f"({right_expression} - {left_expression})", right_value - left_value)
            )
        for candidate in combined:
            found = _solve(rest + [candidate], target)
            if found is not None:
                return found
    return None


class _FakeCompletion:
    def __init__(self, text, generated_token_count, is_truncated):
        self.text = text
        self.generated_token_count = generated_token_count
        self.is_truncated = is_truncated


def _install_fakes(
    monkeypatch,
    *,
    model_revision="model-sha",
    refuse_generation=False,
    truncated_caps=(),
):
    class FakeScorer:
        metadata = Namespace(
            **FAKE_MODEL_METADATA,
            model_revision=model_revision,
            tokenizer_revision=model_revision,
            temperature=1.0,
        )

        def score_states(self, states, *, batch_size, menu_ordering_id, rules):
            records = []
            for state in states:
                actions = allowed_actions(state, rules)
                # Weight by action result so advantages vary and strata differ.
                weights = [1.0 + action.result % 5 for action in actions]
                total = sum(weights)
                records.append(
                    StateActionProbabilities(
                        state=state,
                        menu=build_allowed_action_menu(
                            state,
                            actions,
                            MENU_LABELS,
                            menu_ordering_id=menu_ordering_id,
                        ),
                        probabilities={
                            action: weight / total
                            for action, weight in zip(actions, weights, strict=True)
                        },
                    )
                )
            return tuple(records)

    class FakeCompletionGenerator:
        metadata = Namespace(
            **FAKE_MODEL_METADATA,
            model_revision=model_revision,
            tokenizer_revision=model_revision,
            temperature=0.6,
        )

        def generate_batch(self, messages, *, count, seed, max_new_tokens):
            if refuse_generation:
                raise AssertionError("Resumption regenerated a saved batch")
            prompt = messages[-1]["content"]
            target = int(
                next(
                    line for line in prompt.splitlines() if line.startswith("Target:")
                ).removeprefix("Target:")
            )
            numbers = [
                int(value)
                for value in next(
                    line
                    for line in prompt.splitlines()
                    if line.startswith("Current numbers:")
                )
                .removeprefix("Current numbers:")
                .split(",")
            ]
            solution = _solve([(str(value), value) for value in numbers], target)
            completions = []
            for offset in range(count):
                rng = random.Random(seed + offset)
                succeeds = solution is not None and rng.random() < 0.5
                text = (
                    f"Reasoning.\n<answer>{solution}</answer>"
                    if succeeds
                    else "Reasoning that never finished"
                )
                is_truncated = max_new_tokens in truncated_caps
                completions.append(
                    _FakeCompletion(
                        text,
                        max_new_tokens
                        if is_truncated
                        else min(len(text), max_new_tokens),
                        is_truncated,
                    )
                )
            return tuple(completions)

    monkeypatch.setattr(bridge, "_load_constrained_scorer", lambda args: FakeScorer())
    monkeypatch.setattr(
        bridge, "_load_completion_generator", lambda args: FakeCompletionGenerator()
    )


def _stage_b_args(tmp_path, dataset):
    return Namespace(
        output=str(tmp_path),
        dataset=str(dataset),
        model="Qwen/Qwen3.5-4B",
        device="cpu",
        constrained_temperature=1.0,
        completion_temperature=0.6,
        problem_limit=2,
        token_cap=1024,
        batch_size=2,
        reference_pool_size=REFERENCE_POOL_SIZE,
        bootstrap_replicates=25,
        bootstrap_seed=2,
        seed=1,
        smoke=True,
    )


def test_problem_selection_filters_four_numbers_and_uses_canonical_state_order(tmp_path):
    dataset = tmp_path / "test.jsonl"
    rows = [
        {"values": [4, 3, 2, 1], "target": 9, "n_values": 4},
        {"values": [1, 2, 3], "target": 1, "n_values": 3},
        {"values": [2, 1, 4, 3], "target": 7, "n_values": 4},
        {"values": [3, 2, 1, 4], "target": 7, "n_values": 4},
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    selected = select_bridge_problems(dataset, count=3)

    assert [problem.state for problem in selected] == [
        State(values=(1, 2, 3, 4), target=7),
        State(values=(1, 2, 3, 4), target=7),
        State(values=(1, 2, 3, 4), target=9),
    ]


def test_stage_a_loader_fails_closed_for_metadata_mismatch(tmp_path):
    output = tmp_path / STAGE_A_FILE
    output.write_text(json.dumps({"metadata": {"model": "wrong"}, "roots": []}))

    with pytest.raises(ValueError, match="metadata"):
        load_stage_a_results(output, expected_metadata={"model": "expected"})


def test_stage_b_refuses_to_resume_a_superseded_complete_dag_artifact(tmp_path):
    (tmp_path / STAGE_B_FILE).write_text(
        json.dumps({"metadata": {"artifact_version": "1.0"}, "problems": []})
    )

    with pytest.raises(ValueError, match="superseded complete-DAG"):
        run_stage_b(_stage_b_args(tmp_path, tmp_path / "unused.jsonl"))


def test_stage_a_checkpoints_raw_responses_and_resumes_with_matching_metadata(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        json.dumps({"values": [1, 2, 8, 9], "target": 3, "n_values": 4}) + "\n"
    )

    class FakeScorer:
        metadata = Namespace(
            model_repository="Qwen/Qwen3.5-4B",
            model_revision="model-sha",
            tokenizer_repository="Qwen/Qwen3.5-4B",
            tokenizer_revision="tokenizer-sha",
            device="cpu",
            dtype="float32",
            temperature=1.0,
        )

        def score_states(self, states, *, batch_size, menu_ordering_id, rules):
            records = []
            for state in states:
                actions = allowed_actions(state, rules)
                menu = build_allowed_action_menu(
                    state, actions, tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), menu_ordering_id=menu_ordering_id
                )
                records.append(
                    StateActionProbabilities(
                        state=state,
                        menu=menu,
                        probabilities={action: 1 / len(actions) for action in actions},
                    )
                )
            return tuple(records)

    class FakeGenerator:
        metadata = Namespace(
            model_repository="Qwen/Qwen3.5-4B",
            model_revision="model-sha",
            tokenizer_repository="Qwen/Qwen3.5-4B",
            tokenizer_revision="tokenizer-sha",
            device="cpu",
            dtype="float32",
            temperature=0.6,
        )

        def generate(self, messages_batch, *, seeds, max_new_tokens):
            return ("<answer>1 + 2 = 3</answer>",)

    monkeypatch.setattr(bridge, "_load_constrained_scorer", lambda args: FakeScorer())
    monkeypatch.setattr(bridge, "_load_deliberative_generator", lambda args: FakeGenerator())
    args = Namespace(
        output=str(tmp_path),
        dataset=str(dataset),
        model="Qwen/Qwen3.5-4B",
        device="cpu",
        constrained_temperature=1.0,
        deliberative_temperature=0.6,
        samples_per_ordering=1,
        max_new_tokens=64,
        seed=1,
        problem_limit=1,
        smoke=True,
    )

    run_stage_a(args)
    run_stage_a(args)

    artifact = json.loads((tmp_path / STAGE_A_FILE).read_text())
    assert len(artifact["roots"][0]["deliberative_responses"]) == 5
    assert artifact["aggregate"]["valid_legal_response_rate"] == 1.0


def test_stage_b_selects_two_actions_per_root_and_scores_both_setups(
    tmp_path, monkeypatch
):
    dataset = _write_dataset(tmp_path)
    _install_fakes(monkeypatch)
    args = _stage_b_args(tmp_path, dataset)

    run_stage_b(args)

    artifact = json.loads((tmp_path / STAGE_B_FILE).read_text())
    assert len(artifact["selection"]["roots"]) == 2
    assert len(artifact["pairs"]) == 4
    assert len(artifact["arms"]) == 6
    # Selection must be justified by exact constrained credit, never by rollouts.
    assert all(
        root["candidates"] and root["selected_state_occupancy"] > 0.0
        for root in artifact["selection"]["roots"]
    )
    assert {pair["stratum"] for pair in artifact["pairs"]} <= {
        "top",
        "middle",
        "bottom",
    }
    aggregate = artifact["aggregate"]
    assert set(aggregate["median_fidelity"]) == {"constrained", "deliberative"}
    assert aggregate["gate"]["decision"] in {"GO", "NO-GO", "INCONCLUSIVE"}
    assert (tmp_path / "policy_bridge_stage_b.md").exists()


def test_stage_b_keeps_reference_and_estimator_pools_disjoint(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path)
    _install_fakes(monkeypatch)

    run_stage_b(_stage_b_args(tmp_path, dataset))

    artifact = json.loads((tmp_path / STAGE_B_FILE).read_text())
    for arm in artifact["arms"]:
        if arm["deterministic_outcome"] is not None:
            assert arm["responses"] == []
            continue
        reference = [row for row in arm["responses"] if row["pool"] == "reference"]
        estimator = [row for row in arm["responses"] if row["pool"] == "estimator"]
        assert len(reference) == REFERENCE_POOL_SIZE
        assert len(estimator) == 16
        assert not {row["batch_seed"] for row in reference} & {
            row["batch_seed"] for row in estimator
        }


def test_stage_b_resumes_saved_batches_without_regenerating_them(
    tmp_path, monkeypatch
):
    dataset = _write_dataset(tmp_path)
    _install_fakes(monkeypatch)
    args = _stage_b_args(tmp_path, dataset)

    run_stage_b(args)
    first = json.loads((tmp_path / STAGE_B_FILE).read_text())
    _install_fakes(monkeypatch, refuse_generation=True)
    run_stage_b(args)
    second = json.loads((tmp_path / STAGE_B_FILE).read_text())

    assert first["arms"] == second["arms"]


def test_stage_b_refuses_to_reuse_responses_after_a_metadata_change(
    tmp_path, monkeypatch
):
    dataset = _write_dataset(tmp_path)
    _install_fakes(monkeypatch)
    run_stage_b(_stage_b_args(tmp_path, dataset))

    drifted = _stage_b_args(tmp_path, dataset)
    drifted.token_cap = 2048

    with pytest.raises(ValueError, match="metadata mismatch for 'token_cap'"):
        run_stage_b(drifted)


def test_full_stage_a_rejects_reduced_problem_or_sample_count(tmp_path):
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        json.dumps({"values": [1, 2, 8, 9], "target": 3, "n_values": 4}) + "\n"
    )

    with pytest.raises(ValueError, match="exactly 32"):
        run_stage_a(
            Namespace(
                output=str(tmp_path),
                dataset=str(dataset),
                model="Qwen/Qwen3.5-4B",
                constrained_temperature=1.0,
                deliberative_temperature=0.6,
                samples_per_ordering=1,
                max_new_tokens=64,
                seed=1,
                problem_limit=1,
                smoke=False,
            )
        )


def test_stage_b_rejects_model_revision_drift_between_resumed_runs(
    tmp_path, monkeypatch
):
    dataset = _write_dataset(tmp_path)
    _install_fakes(monkeypatch)
    args = _stage_b_args(tmp_path, dataset)
    run_stage_b(args)

    _install_fakes(monkeypatch, model_revision="a-different-sha")

    with pytest.raises(ValueError, match="model metadata mismatch"):
        run_stage_b(args)


def test_full_stage_b_rejects_reduced_root_or_bootstrap_counts(tmp_path):
    args = _stage_b_args(tmp_path, tmp_path / "unused.jsonl")
    args.smoke = False

    with pytest.raises(ValueError, match="exactly 12 roots"):
        run_stage_b(args)

    args.problem_limit = 12
    with pytest.raises(ValueError, match="exactly 1,000 bootstrap"):
        run_stage_b(args)


def test_cost_projection_is_measured_wall_time_not_a_token_cap_bound():
    """The 4090 preflight measured 21.64 seconds per response at $0.34/hour."""
    projection = bridge.project_stage_b_cost(
        response_count=36 * (128 + 16),
        seconds_per_response=21.64,
        gpu_hourly_usd=0.34,
    )

    assert projection["response_count"] == 5184
    assert projection["projected_gpu_hours"] == pytest.approx(31.16, abs=0.02)
    assert projection["projected_usd"] == pytest.approx(10.59, abs=0.02)
    assert projection["projected_usd_with_contingency"] == pytest.approx(
        12.71, abs=0.02
    )


def test_budget_projection_picks_the_largest_affordable_reference_pool():
    chosen, projections = bridge.choose_reference_pool_size(
        generating_arm_count=36,
        seconds_per_response=21.64,
        gpu_hourly_usd=0.34,
        budget_ceiling_usd=10.0,
    )

    # The larger pools overrun at this rate, so planning falls back to the
    # biggest one that fits rather than refusing outright.
    assert chosen == 64
    assert [projection["reference_pool_size"] for projection in projections] == [
        256,
        128,
        64,
        32,
    ]
    assert [projection["affordable"] for projection in projections] == [
        False,
        False,
        True,
        True,
    ]
    assert projections[1]["response_count"] == 36 * (128 + 16)


def test_budget_ceiling_is_explicit_so_an_override_is_visible():
    arguments = {
        "generating_arm_count": 36,
        "seconds_per_response": 21.64,
        "gpu_hourly_usd": 0.34,
    }

    # A 128-sample reference needs about $12.71, so it is reachable only by
    # explicitly raising the predeclared $10 ceiling.
    assert bridge.choose_reference_pool_size(**arguments, budget_ceiling_usd=10.0)[0] == 64
    assert bridge.choose_reference_pool_size(**arguments, budget_ceiling_usd=13.0)[0] == 128
    assert bridge.choose_reference_pool_size(**arguments, budget_ceiling_usd=25.0)[0] == 256


def test_budget_refuses_the_run_when_even_the_small_pool_is_too_expensive():
    chosen, projections = bridge.choose_reference_pool_size(
        generating_arm_count=36,
        seconds_per_response=200.0,
        gpu_hourly_usd=0.34,
    )

    assert chosen is None
    assert all(not projection["affordable"] for projection in projections)


def _write_twelve_root_dataset(tmp_path):
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"values": [2, 3, 4, offset], "target": 14, "n_values": 4})
            for offset in range(5, 17)
        )
        + "\n"
    )
    return dataset


def _calibrate_args(tmp_path, dataset):
    return Namespace(
        output=str(tmp_path),
        dataset=str(dataset),
        model="Qwen/Qwen3.5-4B",
        device="cpu",
        completion_temperature=0.6,
        seed=1,
        smoke=True,
    )


def test_calibration_selects_the_smallest_cap_that_avoids_truncation(
    tmp_path, monkeypatch
):
    dataset = _write_twelve_root_dataset(tmp_path)
    _install_fakes(monkeypatch, truncated_caps=(1024,))

    bridge.run_calibrate(_calibrate_args(tmp_path, dataset))

    artifact = json.loads((tmp_path / "policy_bridge_calibration.json").read_text())
    assert [row["token_cap"] for row in artifact["measurements"]] == [1024, 2048, 3072]
    assert artifact["measurements"][0]["truncation_rate"] == 1.0
    assert artifact["measurements"][1]["truncation_rate"] == 0.0
    assert artifact["selected_token_cap"] == 2048
    assert (tmp_path / "policy_bridge_calibration.md").exists()


def test_calibration_refuses_paid_inference_when_every_cap_truncates(
    tmp_path, monkeypatch
):
    dataset = _write_twelve_root_dataset(tmp_path)
    _install_fakes(monkeypatch, truncated_caps=(1024, 2048, 3072))

    with pytest.raises(ValueError, match="not measurable"):
        bridge.run_calibrate(_calibrate_args(tmp_path, dataset))

    artifact = json.loads((tmp_path / "policy_bridge_calibration.json").read_text())
    assert artifact["selected_token_cap"] is None


def test_preflight_measures_throughput_and_reports_affordable_pool_size(
    tmp_path, monkeypatch
):
    dataset = _write_twelve_root_dataset(tmp_path)
    _install_fakes(monkeypatch)

    bridge.run_preflight(
        Namespace(
            output=str(tmp_path),
            dataset=str(dataset),
            model="Qwen/Qwen3.5-4B",
            device="cuda",
            completion_temperature=0.6,
            token_cap=2048,
            batch_sizes=[4, 8],
            batches=2,
            generating_arm_count=36,
            gpu_hourly_usd=0.44,
            budget_ceiling_usd=25.0,
            seed=1,
            smoke=True,
        )
    )

    artifact = json.loads((tmp_path / "policy_bridge_preflight.json").read_text())
    assert [row["batch_size"] for row in artifact["measurements"]] == [4, 8]
    assert all(len(row["batches"]) == 2 for row in artifact["measurements"])
    assert artifact["seconds_per_response"] > 0.0
    assert artifact["recommended_batch_size"] in (4, 8)
    assert artifact["affordable_reference_pool_size"] in bridge.REFERENCE_POOL_SIZES


def test_preflight_keeps_going_when_a_large_batch_size_runs_out_of_memory(
    tmp_path, monkeypatch
):
    dataset = _write_twelve_root_dataset(tmp_path)
    _install_fakes(monkeypatch)
    load_fake_generator = bridge._load_completion_generator

    class OutOfMemoryAboveEight:
        def __init__(self, inner):
            self._inner = inner
            self.metadata = inner.metadata

        def generate_batch(self, messages, *, count, seed, max_new_tokens):
            if count > 8:
                raise RuntimeError("CUDA out of memory: tried to allocate 2.00 GiB")
            return self._inner.generate_batch(
                messages, count=count, seed=seed, max_new_tokens=max_new_tokens
            )

    monkeypatch.setattr(
        bridge,
        "_load_completion_generator",
        lambda args: OutOfMemoryAboveEight(load_fake_generator(args)),
    )

    bridge.run_preflight(
        Namespace(
            output=str(tmp_path),
            dataset=str(dataset),
            model="Qwen/Qwen3.5-4B",
            device="cuda",
            completion_temperature=0.6,
            token_cap=2048,
            batch_sizes=[8, 16],
            batches=1,
            generating_arm_count=36,
            gpu_hourly_usd=0.34,
            budget_ceiling_usd=25.0,
            seed=1,
            smoke=True,
        )
    )

    artifact = json.loads((tmp_path / "policy_bridge_preflight.json").read_text())
    assert artifact["recommended_batch_size"] == 8
    failed = [row for row in artifact["measurements"] if row["failure"] is not None]
    assert len(failed) == 1
    assert "out of memory" in failed[0]["failure"].lower()


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--batches", "0"), ("--batch-sizes", "0"), ("--gpu-hourly-usd", "0")],
)
def test_preflight_rejects_nonsense_arguments_before_loading_a_model(flag, value):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.evaluate_policy_bridge",
            "preflight",
            "--output",
            "/tmp/unused",
            "--token-cap",
            "2048",
            "--gpu-hourly-usd",
            "0.44",
            flag,
            value,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode != 0
    assert "at least one" in result.stderr or "must be positive" in result.stderr
    assert "traceback" not in result.stderr.lower()


def test_token_cap_selection_prefers_the_smallest_non_truncating_cap():
    assert bridge.choose_token_cap({1024: 0.6, 2048: 0.2, 3072: 0.1}) == 2048
    assert bridge.choose_token_cap({1024: 0.1, 2048: 0.05}) == 1024
    # Every cap truncating means the completion policy was never measured.
    assert bridge.choose_token_cap({1024: 0.9, 2048: 0.8, 3072: 0.7}) is None


def test_deterministic_arms_need_no_generation():
    solved = State(values=(4, 9), target=9)
    dead_end = State(values=(9,), target=4)
    open_state = State(values=(2, 3), target=5)

    assert bridge._deterministic_outcome(solved) == 1
    assert bridge._deterministic_outcome(dead_end) == 0
    assert bridge._deterministic_outcome(open_state) is None


@pytest.mark.parametrize("arguments", [["--help"], ["stage-a", "--help"], ["stage-b", "--help"]])
def test_policy_bridge_cli_help_and_import_do_not_load_transformers(arguments):
    result = subprocess.run(
        [sys.executable, "-m", "experiments.evaluate_policy_bridge", *arguments],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "transformers" not in result.stderr.lower()


def test_importing_policy_bridge_does_not_import_transformers():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import experiments.evaluate_policy_bridge; "
                "print('transformers' in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
