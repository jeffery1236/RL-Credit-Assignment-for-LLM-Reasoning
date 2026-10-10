"""Tests for deterministic Phase 3 pilot selection and estimator evaluation."""

import functools
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from analysis.conclusion_robustness_metrics import METHOD_NAMES, PairSamples
from analysis.credit_estimator_fidelity import (
    PHASE3_ESTIMATOR_ORDER,
    ROLLOUT_COST_BY_ESTIMATOR,
    ProblemClusteredFidelity,
    derive_estimator_pools,
)
from env.countdown import Action, State, allowed_actions, transition
from experiments.benchmark_credit_estimator_fidelity import (
    Phase3PilotProblem,
    _benchmark_b_fidelity_bootstrap_interval,
    _write_stratified_plots,
    assess_phase3_pilot_gate,
    build_arg_parser,
    build_state_candidates,
    compute_credit_estimator_fidelity,
    compute_policy_visit_probabilities,
    evaluate_stratified_gate,
    main,
    phase3_estimator_plot_label,
    run_stratified_benchmark,
    sample_phase3_credit_estimator_pairs,
    sample_stratified_state_samples,
    select_phase3_validation_problems,
    stable_pool_seed,
    with_shared_parent_baseline,
)
from oracle.evaluation import build_reasoning_dag
from oracle.float_policy_evaluation import FloatPolicyEvaluator

REPOSITORY_ROOT = Path(__file__).parents[1]
PHASE3_RUNPOD_LAUNCHER = REPOSITORY_ROOT / "scripts/runpod_phase3_credit_estimator_pilot.sh"


def test_select_phase3_validation_problems_uses_sorted_hashes_and_four_number_rows(
    tmp_path,
):
    dataset_path = tmp_path / "validation.jsonl"
    dataset_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "canonical_hash": "z-last",
                    "n_values": 4,
                    "target": 19,
                    "values": [2, 3, 4, 5],
                },
                {
                    "canonical_hash": "five-number",
                    "n_values": 5,
                    "target": 42,
                    "values": [1, 2, 3, 4, 5],
                },
                {
                    "canonical_hash": "a-first",
                    "n_values": 4,
                    "target": 10,
                    "values": [1, 2, 3, 4],
                },
                {
                    "canonical_hash": "m-middle",
                    "n_values": 4,
                    "target": 25,
                    "values": [4, 5, 7, 9],
                },
            )
        )
        + "\n"
    )

    selected_problems = select_phase3_validation_problems(dataset_path, count=2)

    assert [
        (problem.canonical_hash, problem.source_row_index)
        for problem in selected_problems
    ] == [("a-first", 2), ("m-middle", 3)]


def test_sample_phase3_credit_estimator_pairs_records_exact_advantage_and_pools():
    root_state = State(values=(1, 2), target=3)
    exact_evaluator = FloatPolicyEvaluator(
        _UniformPolicy(),
        build_reasoning_dag(root_state),
    )

    sampled_pairs = sample_phase3_credit_estimator_pairs(
        root_state,
        exact_evaluator,
        pair_count=3,
        estimator_pool_size=4,
        seed=7,
    )

    assert len(sampled_pairs) == 3
    for sampled_pair in sampled_pairs:
        assert sampled_pair.exact_advantage == pytest.approx(
            exact_evaluator.a_pi(sampled_pair.state, sampled_pair.action)
        )
        assert len(sampled_pair.parent_reset_outcomes) == 4
        assert len(sampled_pair.child_reset_outcomes) == 4
        assert set(sampled_pair.parent_reset_outcomes) <= {0, 1}
        assert set(sampled_pair.child_reset_outcomes) <= {0, 1}


class _UniformPolicy:
    def action_probs(self, state, actions):
        probability = 1.0 / len(actions)
        return {action: probability for action in actions}


def test_compute_credit_estimator_fidelity_scores_each_predeclared_estimator():
    benchmark_pairs = (
        PairSamples((1,) * 16, (0,) * 16, 1.0),
        PairSamples((0,) * 16, (1,) * 16, -1.0),
        PairSamples((1,) * 16, (1,) * 16, 0.0),
    )

    fidelity_by_estimator = compute_credit_estimator_fidelity(benchmark_pairs)

    assert tuple(fidelity_by_estimator) == METHOD_NAMES
    assert fidelity_by_estimator["mc_reset_k16"] == pytest.approx(1.0)


def test_assess_phase3_pilot_gate_requires_material_reliable_improvement():
    benchmark_pairs = (
        PairSamples((1,) * 16, (0,) * 16, 1.0),
        PairSamples((0,) * 16, (1,) * 16, -1.0),
        PairSamples((1,) * 16, (1,) * 16, 0.0),
    )

    pilot_gate = assess_phase3_pilot_gate(
        benchmark_pairs,
        bootstrap_replicates=25,
        bootstrap_seed=7,
    )

    assert pilot_gate.decision == "CONTINUE"
    assert pilot_gate.median_fidelity_improvement >= 0.10
    assert pilot_gate.bootstrap_support == 1.0


def test_phase3_benchmark_command_describes_required_pilot_arguments():
    completed_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.benchmark_credit_estimator_fidelity",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--validation-dataset" in completed_process.stdout
    assert "--problem-count" in completed_process.stdout
    assert "--pairs-per-problem" in completed_process.stdout


def test_phase3_policy_scoring_defaults_to_measured_safe_batch_size():
    parsed_arguments = build_arg_parser().parse_args(
        [
            "--validation-dataset",
            "data/v1/validation.jsonl",
            "--output",
            "artifacts/phase3/smoke",
        ]
    )

    assert parsed_arguments.batch_size == 1


def test_phase3_benchmark_dry_run_writes_selected_problem_metadata(tmp_path):
    dataset_path = tmp_path / "validation.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "canonical_hash": "pilot-root",
                "n_values": 4,
                "target": 10,
                "values": [1, 2, 3, 4],
            }
        )
        + "\n"
    )
    output_directory = tmp_path / "artifact"

    main(
        [
            "--validation-dataset",
            str(dataset_path),
            "--output",
            str(output_directory),
            "--problem-count",
            "1",
            "--dry-run",
        ]
    )

    artifact = json.loads(
        (output_directory / "phase3_credit_estimator_pilot.json").read_text()
    )
    assert artifact["status"] == "dry_run"
    assert artifact["selected_problems"] == [
        {
            "canonical_hash": "pilot-root",
            "root": {"target": 10, "values": [1, 2, 3, 4]},
            "source_row_index": 0,
        }
    ]


class _FakeScorer:
    """A deterministic scorer that exposes immutable fake model metadata."""

    load_calls = 0
    score_calls: ClassVar[list[str]] = []

    def __init__(self, model_revision="fake-model-revision"):
        self.metadata = SimpleNamespace(
            model_repository="fake/repository",
            model_revision=model_revision,
            tokenizer_repository="fake/repository",
            tokenizer_revision="fake-tokenizer-revision",
        )

    @classmethod
    def load(cls, *args, **kwargs):
        cls.load_calls += 1
        return cls()

    def score_states(self, states, *, batch_size, menu_ordering_id):
        self.score_calls.append(menu_ordering_id)
        return tuple(
            SimpleNamespace(
                state=state,
                probabilities={
                    action: 1.0 / len(allowed_actions(state))
                    for action in allowed_actions(state)
                },
            )
            for state in states
        )


def _write_phase3_dataset(tmp_path):
    dataset_path = tmp_path / "validation.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "canonical_hash": "pilot-root",
                "n_values": 4,
                "target": 10,
                "values": [1, 2, 3, 4],
            }
        )
        + "\n"
    )
    return dataset_path


def _phase3_cli_arguments(dataset_path, output_directory):
    return [
        "--validation-dataset",
        str(dataset_path),
        "--output",
        str(output_directory),
        "--problem-count",
        "1",
        "--pairs-per-problem",
        "3",
        "--estimator-pool-size",
        "16",
        "--bootstrap-replicates",
        "5",
    ]


def test_phase3_benchmark_scores_all_menu_orderings_and_writes_derived_artifacts(
    tmp_path, monkeypatch
):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    _FakeScorer.load_calls = 0
    _FakeScorer.score_calls = []
    monkeypatch.setattr(
        "inference.huggingface_allowed_action_scorer.HuggingFaceAllowedActionScorer",
        _FakeScorer,
    )

    main(_phase3_cli_arguments(dataset_path, output_directory))

    assert _FakeScorer.load_calls == 1
    assert _FakeScorer.score_calls == [
        "phase2-v1",
        "phase2-sensitivity-1",
        "phase2-sensitivity-2",
        "phase2-sensitivity-3",
        "phase2-sensitivity-4",
    ]
    checkpoint = json.loads(
        (output_directory / "problems" / "pilot-root.json").read_text()
    )
    assert checkpoint["scorer_metadata"] == {
        "model_repository": "fake/repository",
        "model_revision": "fake-model-revision",
        "tokenizer_repository": "fake/repository",
        "tokenizer_revision": "fake-tokenizer-revision",
    }
    assert (output_directory / "phase3_credit_estimator_pilot_summary.md").is_file()
    assert (output_directory / "phase3_credit_estimator_pilot_summary.json").is_file()
    assert {
        path.name for path in (output_directory / "plots").glob("*.svg")
    } == {
        "fidelity_by_rollout_count.svg",
        "fidelity_by_problem_horizon.svg",
        "pivotal_action_detection.svg",
        "fidelity_cost_frontier.svg",
    }


def test_phase3_benchmark_reuses_complete_cache_without_loading_scorer(
    tmp_path, monkeypatch
):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    monkeypatch.setattr(
        "inference.huggingface_allowed_action_scorer.HuggingFaceAllowedActionScorer",
        _FakeScorer,
    )
    main(_phase3_cli_arguments(dataset_path, output_directory))

    class _ScorerMustNotLoad:
        @classmethod
        def load(cls, *args, **kwargs):
            raise AssertionError("a complete checkpoint must not load the scorer")

    monkeypatch.setattr(
        "inference.huggingface_allowed_action_scorer.HuggingFaceAllowedActionScorer",
        _ScorerMustNotLoad,
    )

    main(_phase3_cli_arguments(dataset_path, output_directory))

    assert json.loads(
        (output_directory / "phase3_credit_estimator_pilot.json").read_text()
    )["status"] == "completed"
    compact_summary = json.loads(
        (output_directory / "phase3_credit_estimator_pilot_summary.json").read_text()
    )
    assert compact_summary["status"] == "completed"
    assert "problems" not in compact_summary


def test_phase3_benchmark_rejects_checkpoint_with_mismatched_immutable_metadata(
    tmp_path, monkeypatch
):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    monkeypatch.setattr(
        "inference.huggingface_allowed_action_scorer.HuggingFaceAllowedActionScorer",
        _FakeScorer,
    )
    main(_phase3_cli_arguments(dataset_path, output_directory))
    checkpoint_path = output_directory / "problems" / "pilot-root.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["scorer_metadata"]["model_revision"] = "mutable-or-corrupt"
    checkpoint_path.write_text(json.dumps(checkpoint))

    with pytest.raises(ValueError, match="immutable scorer metadata"):
        main(_phase3_cli_arguments(dataset_path, output_directory))


def test_stratified_checkpoint_persists_complete_menu_averaged_policy(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    scorer = _FakeScorer()

    run_stratified_benchmark(
        build_arg_parser().parse_args(_phase3_cli_arguments(dataset_path, output_directory)),
        scorer=scorer,
    )

    checkpoint_path = (
        output_directory / "stratified" / "problems" / "pilot-root.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    expected_nonterminal_states = sum(
        bool(allowed_actions(state))
        for state in build_reasoning_dag(State(values=(1, 2, 3, 4), target=10)).nodes
    )
    assert len(checkpoint["menu_averaged_policy"]) == expected_nonterminal_states
    assert checkpoint["selection"]["depths"] == [0, 1, 2]
    assert all(
        len(record["actions"]) == len(
            allowed_actions(
                State(
                    values=tuple(record["state"]["values"]),
                    target=record["state"]["target"],
                )
            )
        )
        for record in checkpoint["menu_averaged_policy"]
    )
    assert checkpoint["node_count"] == build_reasoning_dag(
        State(values=(1, 2, 3, 4), target=10)
    ).node_count
    assert isinstance(checkpoint["scoring_duration_seconds"], float)
    assert checkpoint["scoring_duration_seconds"] >= 0.0


def test_stratified_checkpoint_persists_raw_parent_and_child_pools(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )

    run_stratified_benchmark(arguments, scorer=_FakeScorer())

    checkpoint = json.loads(
        (
            output_directory / "stratified" / "problems" / "pilot-root.json"
        ).read_text()
    )
    persisted = checkpoint["selection"]["selected_credit_states"]
    for credit_state in persisted:
        assert len(credit_state["shared_parent_outcomes"]) == 16
        for action in credit_state["actions"]:
            assert len(action["parent_outcomes"]) == 16
            assert len(action["child_outcomes"]) == 16
            pools = derive_estimator_pools(
                action["parent_outcomes"], action["child_outcomes"]
            )
            assert pools["uniform"] == (action["child_outcomes"][0],)
            for rollout_count in (1, 2, 4, 8, 16):
                estimator = f"mc_reset_k{rollout_count}"
                assert pools[estimator] == pools["mc_reset_k16"][:rollout_count]

    # The persisted pools are exactly what a model-free rerun regenerates.
    root_state = State(values=(1, 2, 3, 4), target=10)
    reasoning_dag = build_reasoning_dag(root_state)
    regenerated = sample_stratified_state_samples(
        Phase3PilotProblem(
            canonical_hash="pilot-root", source_row_index=0, state=root_state
        ),
        arguments,
        reasoning_dag,
        FloatPolicyEvaluator(_UniformPolicy(), reasoning_dag),
    )
    assert [
        (
            credit_state["shared_parent_outcomes"],
            [action["parent_outcomes"] for action in credit_state["actions"]],
            [action["child_outcomes"] for action in credit_state["actions"]],
        )
        for credit_state in persisted
    ] == [
        (
            list(sample.shared_parent_outcomes),
            [list(credit.parent_outcomes) for credit in sample.credit_state.action_credits],
            [list(credit.child_outcomes) for credit in sample.credit_state.action_credits],
        )
        for sample in regenerated
    ]


def test_stratified_analysis_is_invariant_to_checkpoint_key_order(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )
    in_canonical_order = run_stratified_benchmark(arguments, scorer=_FakeScorer())

    checkpoint_path = output_directory / "stratified" / "problems" / "pilot-root.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    shuffled = dict(reversed(list(checkpoint.items())))
    shuffled["menu_averaged_policy"] = list(
        reversed(shuffled["menu_averaged_policy"])
    )
    checkpoint_path.write_text(json.dumps(shuffled, indent=2))

    in_shuffled_order = run_stratified_benchmark(arguments, scorer=_FakeScorer())

    assert (
        in_shuffled_order["fidelity_by_estimator"]
        == in_canonical_order["fidelity_by_estimator"]
    )
    assert in_shuffled_order["stratified_gate"] == in_canonical_order["stratified_gate"]


def test_stable_pool_seed_separates_every_pool_identity():
    state = State(values=(1, 2, 3, 4), target=10)
    ordered_actions = sorted(allowed_actions(state), key=Action.sort_key)
    action, other_action = ordered_actions[0], ordered_actions[1]
    seed_for = functools.partial(stable_pool_seed, 11, "pilot-root", 0, state)

    assert seed_for(action, "parent") == seed_for(action, "parent")
    assert len(
        {
            seed_for(action, "parent"),
            seed_for(action, "child"),
            seed_for(other_action, "parent"),
            seed_for(None, "shared_parent"),
            stable_pool_seed(12, "pilot-root", 0, state, action, "parent"),
            stable_pool_seed(11, "other-root", 0, state, action, "parent"),
        }
    ) == 6


def test_compute_policy_visit_probabilities_propagates_policy_mass():
    root_state = State(values=(1, 2), target=3)
    reasoning_dag = build_reasoning_dag(root_state)
    evaluator = FloatPolicyEvaluator(_UniformPolicy(), reasoning_dag)

    occupancy = compute_policy_visit_probabilities(reasoning_dag, evaluator)

    assert occupancy[root_state] == pytest.approx(1.0)
    terminal_mass = sum(
        probability
        for state, probability in occupancy.items()
        if not allowed_actions(state)
    )
    assert terminal_mass == pytest.approx(1.0)
    assert all(
        occupancy[transition(root_state, action)]
        == pytest.approx(1.0 / len(allowed_actions(root_state)))
        for action in allowed_actions(root_state)
    )


def test_build_state_candidates_covers_every_nonterminal_action_without_pools():
    root_state = State(values=(1, 2, 3), target=6)
    reasoning_dag = build_reasoning_dag(root_state)
    evaluator = FloatPolicyEvaluator(_UniformPolicy(), reasoning_dag)
    occupancy = compute_policy_visit_probabilities(reasoning_dag, evaluator)

    candidates = build_state_candidates(
        reasoning_dag, evaluator, occupancy, canonical_hash="pilot-root"
    )

    nonterminal_states = [
        state for state in reasoning_dag.topological_order() if allowed_actions(state)
    ]
    assert len(candidates) == len(nonterminal_states)
    assert [candidate.depth for candidate in candidates] == [
        reasoning_dag.nodes[state].depth for state in nonterminal_states
    ]
    assert [len(candidate.action_credits) for candidate in candidates] == [
        len(allowed_actions(state)) for state in nonterminal_states
    ]
    # Selection must not be able to see rollout outcomes.
    assert all(
        credit.parent_outcomes == () and credit.child_outcomes == ()
        for candidate in candidates
        for credit in candidate.action_credits
    )
    assert all(
        candidate.action_credits[0].exact_advantage
        == pytest.approx(
            evaluator.a_pi(state, min(allowed_actions(state), key=Action.sort_key))
        )
        for candidate, state in zip(candidates, nonterminal_states)
    )


def test_stratified_selection_picks_the_widest_exact_spread_state_at_each_depth(
    tmp_path,
):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )
    root_state = State(values=(1, 2, 3, 4), target=10)
    reasoning_dag = build_reasoning_dag(root_state)
    evaluator = FloatPolicyEvaluator(_UniformPolicy(), reasoning_dag)

    samples = sample_stratified_state_samples(
        Phase3PilotProblem(
            canonical_hash="pilot-root", source_row_index=0, state=root_state
        ),
        arguments,
        reasoning_dag,
        evaluator,
    )

    occupancy = compute_policy_visit_probabilities(reasoning_dag, evaluator)
    candidates = build_state_candidates(
        reasoning_dag, evaluator, occupancy, canonical_hash="pilot-root"
    )
    for sample in samples:
        widest = max(
            candidate.advantage_spread
            for candidate in candidates
            if candidate.depth == sample.credit_state.depth
            and len(candidate.action_credits) >= 3
        )
        assert sample.credit_state.advantage_spread == pytest.approx(widest)
    assert [sample.credit_state.depth for sample in samples] == [0, 1, 2]


def test_stratified_selected_state_action_pairs_are_unique(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )

    artifact = run_stratified_benchmark(arguments, scorer=_FakeScorer())

    assert artifact["selected_action_count"] > 0
    assert (
        artifact["unique_state_action_pair_count"] == artifact["selected_action_count"]
    )


def test_stratified_aggregate_records_checkpoint_paths_instead_of_contents(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )

    artifact = run_stratified_benchmark(arguments, scorer=_FakeScorer())

    stratified_directory = output_directory / "stratified"
    aggregate_text = (
        stratified_directory / "phase3_credit_estimator_stratified.json"
    ).read_text()
    assert "menu_averaged_policy" not in aggregate_text
    for problem in artifact["problems"]:
        checkpoint_path = stratified_directory / problem["checkpoint_path"]
        assert checkpoint_path.is_file()
        assert (
            hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            == problem["checkpoint_sha256"]
        )
        assert json.loads(checkpoint_path.read_text())["canonical_hash"] == problem[
            "canonical_hash"
        ]


def test_stratified_cache_only_run_does_not_rewrite_unchanged_checkpoints(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )
    run_stratified_benchmark(arguments, scorer=_FakeScorer())
    checkpoint_path = output_directory / "stratified" / "problems" / "pilot-root.json"
    modified_before = checkpoint_path.stat().st_mtime_ns

    run_stratified_benchmark(arguments, scorer=_FakeScorer())

    assert checkpoint_path.stat().st_mtime_ns == modified_before


def test_stratified_cache_migrates_the_legacy_benchmark_a_identity_key(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )
    run_stratified_benchmark(arguments, scorer=_FakeScorer())
    stratified_directory = output_directory / "stratified"
    for path in (
        stratified_directory / "run_manifest.json",
        stratified_directory / "problems" / "pilot-root.json",
    ):
        document = json.loads(path.read_text())
        document["checkpoint_configuration"]["pairs_per_problem"] = 32
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    run_stratified_benchmark(arguments)

    assert "pairs_per_problem" not in json.loads(
        (stratified_directory / "run_manifest.json").read_text()
    )["checkpoint_configuration"]
    assert "pairs_per_problem" not in json.loads(
        (stratified_directory / "problems" / "pilot-root.json").read_text()
    )["checkpoint_configuration"]


def test_stratified_sensitivity_analyses_are_labelled_and_leave_the_gate_alone(
    tmp_path,
):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )

    artifact = run_stratified_benchmark(arguments, scorer=_FakeScorer())

    assert artifact["primary_baseline"].startswith("per-action parent rollout pool")
    assert [
        sensitivity["key"] for sensitivity in artifact["sensitivity_analyses"]
    ] == [
        "shared_per_state_baseline",
        "stratum_balanced",
        "nondegenerate_states",
        "depths_0_and_1",
    ]
    by_key = {
        sensitivity["key"]: sensitivity
        for sensitivity in artifact["sensitivity_analyses"]
    }
    assert by_key["depths_0_and_1"]["state_count"] == sum(
        artifact["selection_coverage_by_depth"][str(depth)]["selected_problem_count"]
        for depth in (0, 1)
    )
    assert by_key["depths_0_and_1"]["action_count"] == sum(
        artifact["selected_action_count_by_depth"][str(depth)] for depth in (0, 1)
    )
    assert all(
        sensitivity["is_post_hoc_diagnostic"]
        and sensitivity["gate"]["minimum_median_fidelity_improvement"]
        == artifact["stratified_gate"]["minimum_median_fidelity_improvement"]
        for sensitivity in artifact["sensitivity_analyses"]
    )
    assert artifact["stratified_gate"]["decision"] in {"CAPABLE", "NOT-DEMONSTRATED"}


def test_shared_parent_baseline_replaces_per_action_pools_at_a_state(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )
    root_state = State(values=(1, 2, 3, 4), target=10)
    reasoning_dag = build_reasoning_dag(root_state)
    samples = sample_stratified_state_samples(
        Phase3PilotProblem(
            canonical_hash="pilot-root", source_row_index=0, state=root_state
        ),
        arguments,
        reasoning_dag,
        FloatPolicyEvaluator(_UniformPolicy(), reasoning_dag),
    )

    shared = with_shared_parent_baseline(samples)

    for sample, shared_state in zip(samples, shared, strict=True):
        assert all(
            credit.parent_outcomes == sample.shared_parent_outcomes
            for credit in shared_state.action_credits
        )
        assert [credit.child_outcomes for credit in shared_state.action_credits] == [
            credit.child_outcomes for credit in sample.credit_state.action_credits
        ]


def test_stratified_markdown_names_the_median_paired_difference(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )

    run_stratified_benchmark(arguments, scorer=_FakeScorer())

    markdown = (
        output_directory / "stratified" / "phase3_credit_estimator_stratified.md"
    ).read_text()
    compact_summary = json.loads(
        (
            output_directory
            / "stratified"
            / "phase3_credit_estimator_stratified_summary.json"
        ).read_text()
    )
    assert "Typical K=16 improvement over one rollout" in markdown
    assert "How often each method picked the truly best move" in markdown
    assert "Extra checks run after seeing the result" in markdown
    assert "K means the number of games simulated both" in markdown
    assert "before and after a move" in markdown
    assert "by_estimator" not in compact_summary["bootstrap"]
    assert "interval_by_estimator" in compact_summary["bootstrap"]


def test_stratified_run_manifest_has_complete_scorer_identity(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )

    run_stratified_benchmark(arguments, scorer=_FakeScorer())

    manifest_path = output_directory / "stratified" / "run_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["scorer_metadata"] == {
        "model_repository": "fake/repository",
        "model_revision": "fake-model-revision",
        "tokenizer_repository": "fake/repository",
        "tokenizer_revision": "fake-tokenizer-revision",
    }


@pytest.mark.parametrize(
    "mutate_manifest",
    [
        lambda manifest: manifest["scorer_metadata"].pop("model_revision"),
        lambda manifest: manifest["scorer_metadata"].__setitem__("unexpected", "value"),
    ],
)
def test_stratified_cache_only_resume_rejects_incomplete_manifest(
    tmp_path, mutate_manifest
):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )
    run_stratified_benchmark(arguments, scorer=_FakeScorer())
    manifest_path = output_directory / "stratified" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="run manifest"):
        run_stratified_benchmark(arguments)


def test_stratified_cache_only_resume_rejects_joint_checkpoint_identity_mutation(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )
    run_stratified_benchmark(arguments, scorer=_FakeScorer())
    checkpoint_path = (
        output_directory / "stratified" / "problems" / "pilot-root.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["scorer_metadata"]["model_repository"] = "jointly-mutated/repository"
    checkpoint["checkpoint_configuration"]["model_revision"] = "jointly-mutated-revision"
    checkpoint["scorer_metadata"]["model_revision"] = "jointly-mutated-revision"
    checkpoint_path.write_text(json.dumps(checkpoint))

    with pytest.raises(ValueError, match="run manifest"):
        run_stratified_benchmark(arguments)


def test_stratified_checkpoint_rejects_metadata_drift(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )
    run_stratified_benchmark(arguments, scorer=_FakeScorer())
    checkpoint_path = (
        output_directory / "stratified" / "problems" / "pilot-root.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["checkpoint_configuration"]["prompt_version"] = "stale-prompt"
    checkpoint_path.write_text(json.dumps(checkpoint))

    with pytest.raises(ValueError, match="Checkpoint does not match"):
        run_stratified_benchmark(arguments, scorer=_FakeScorer())


def test_stratified_cache_reuse_never_scores_and_preserves_benchmark_a(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    benchmark_a_arguments = _phase3_cli_arguments(dataset_path, output_directory)
    _FakeScorer.load_calls = 0
    _FakeScorer.score_calls = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "inference.huggingface_allowed_action_scorer.HuggingFaceAllowedActionScorer",
        _FakeScorer,
    )
    try:
        main(benchmark_a_arguments)
    finally:
        monkeypatch.undo()
    benchmark_a_path = output_directory / "phase3_credit_estimator_pilot.json"
    aggregate_hash_before = hashlib.sha256(benchmark_a_path.read_bytes()).hexdigest()
    arguments = build_arg_parser().parse_args(benchmark_a_arguments)
    run_stratified_benchmark(arguments, scorer=_FakeScorer())

    class _ScorerMustNotScore:
        metadata = _FakeScorer().metadata

        def score_states(self, *args, **kwargs):
            raise AssertionError("a complete checkpoint must not score the policy")

    run_stratified_benchmark(arguments, scorer=_ScorerMustNotScore())

    assert hashlib.sha256(benchmark_a_path.read_bytes()).hexdigest() == aggregate_hash_before


def _stratified_bootstrap(uniform_draws, counterfactual_draws):
    return ProblemClusteredFidelity(
        by_estimator={
            "uniform": tuple(uniform_draws),
            "mc_reset_k16": tuple(counterfactual_draws),
        },
        problem_draw_count=len(uniform_draws),
        dropped_replicate_count=0,
    )


def test_stratified_gate_uses_paired_delta_boundaries():
    capable_bootstrap = _stratified_bootstrap(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.1, 0.1, 0.1, 0.1, -0.1),
    )
    below_improvement_boundary = _stratified_bootstrap(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.099, 0.099, 0.099, 0.099, 0.099),
    )
    below_support_boundary = _stratified_bootstrap(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.1, 0.1, 0.1, 0.0, -0.1),
    )

    assert evaluate_stratified_gate(capable_bootstrap).decision == "CAPABLE"
    assert evaluate_stratified_gate(
        capable_bootstrap
    ).median_fidelity_improvement == pytest.approx(0.1)
    assert evaluate_stratified_gate(capable_bootstrap).bootstrap_support == pytest.approx(
        0.8
    )
    assert (
        evaluate_stratified_gate(below_improvement_boundary).decision
        == "NOT-DEMONSTRATED"
    )
    assert (
        evaluate_stratified_gate(below_support_boundary).decision
        == "NOT-DEMONSTRATED"
    )


def test_stratified_gate_uses_median_paired_delta_not_difference_of_medians():
    bootstrap = _stratified_bootstrap(
        (0.0, 100.0, 101.0),
        (100.0, 101.0, 0.0),
    )

    assert evaluate_stratified_gate(bootstrap).median_fidelity_improvement == 1.0


def test_benchmark_b_fidelity_bootstrap_interval_summarizes_draws():
    assert _benchmark_b_fidelity_bootstrap_interval((0.1, 0.2, 0.3)) == {
        "lower_95": 0.1,
        "median": 0.2,
        "upper_95": 0.3,
    }


def test_stratified_benchmark_writes_isolated_report_artifacts(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )

    artifact = run_stratified_benchmark(arguments, scorer=_FakeScorer())

    assert artifact["stratified_gate"]["decision"] in {"CAPABLE", "NOT-DEMONSTRATED"}
    assert artifact["selection_coverage_by_depth"]["0"]["selected_problem_count"] == 1
    markdown_path = (
        output_directory / "stratified" / "phase3_credit_estimator_stratified.md"
    )
    assert markdown_path.is_file()
    assert all(not line.endswith(" ") for line in markdown_path.read_text().splitlines())
    assert {
        path.name for path in (output_directory / "stratified" / "plots").glob("*.svg")
    } == {
        "fidelity_by_rollout_count.svg",
        "fidelity_by_depth.svg",
        "pivotal_action_detection.svg",
        "fidelity_cost_frontier.svg",
    }


def _plot_artifact():
    fidelity_by_estimator = {
        estimator: (index + 1) / 10
        for index, estimator in enumerate(PHASE3_ESTIMATOR_ORDER)
    }
    return {
        "fidelity_by_estimator": fidelity_by_estimator,
        "fidelity_by_estimator_and_depth": {
            estimator: {"0": 0.1, "1": 0.2, "2": 0.3}
            for estimator in PHASE3_ESTIMATOR_ORDER
        },
        "pivotal_action_detection": {
            estimator: {
                "detected_state_count": index,
                "pivotal_state_count": 10,
                "detection_rate": index / 10,
            }
            for index, estimator in enumerate(PHASE3_ESTIMATOR_ORDER)
        },
    }


def test_stratified_estimator_plots_pair_display_labels_with_ordered_values(tmp_path):
    artifact = _plot_artifact()
    fidelity_by_estimator = artifact["fidelity_by_estimator"]
    output_directory = tmp_path / "plots"

    _write_stratified_plots(artifact, output_directory)

    plot = (output_directory / "plots" / "fidelity_by_rollout_count.svg").read_text()
    label_positions = [
        plot.index(phase3_estimator_plot_label(estimator))
        for estimator in PHASE3_ESTIMATOR_ORDER
    ]
    value_positions = [
        plot.index(f">{fidelity_by_estimator[estimator]:.3f}</text>")
        for estimator in PHASE3_ESTIMATOR_ORDER
    ]
    assert label_positions == sorted(label_positions)
    assert value_positions == sorted(value_positions)
    assert "mc_reset_k" not in plot
    assert "How accurately each method ranks the available moves" in plot
    assert phase3_estimator_plot_label("uniform") == "One rollout"
    assert phase3_estimator_plot_label("mc_reset_k16") == "Before/after ×16"


def test_fidelity_cost_frontier_plots_actual_rollout_costs(tmp_path):
    artifact = _plot_artifact()
    output_directory = tmp_path / "plots"

    _write_stratified_plots(artifact, output_directory)

    frontier = (output_directory / "plots" / "fidelity_cost_frontier.svg").read_text()
    by_rollout_count = (
        output_directory / "plots" / "fidelity_by_rollout_count.svg"
    ).read_text()
    cost_label_positions = [
        frontier.index(">1 rollout</text>"),
        *(frontier.index(f">{cost} rollouts</text>") for cost in (2, 4, 8, 16, 32)),
    ]
    assert cost_label_positions == sorted(cost_label_positions)
    assert ROLLOUT_COST_BY_ESTIMATOR["uniform"] == 1
    assert ROLLOUT_COST_BY_ESTIMATOR["mc_reset_k16"] == 32
    # The frontier is a cost axis, not a second copy of the estimator chart.
    assert phase3_estimator_plot_label("uniform") not in frontier
    assert frontier != by_rollout_count


def test_pivotal_action_detection_plot_shows_detection_rates(tmp_path):
    artifact = _plot_artifact()
    output_directory = tmp_path / "plots"

    _write_stratified_plots(artifact, output_directory)

    plot = (output_directory / "plots" / "pivotal_action_detection.svg").read_text()
    assert "How often each method picks the truly best move" in plot
    assert "Defined-state coverage" not in plot
    for estimator, counts in artifact["pivotal_action_detection"].items():
        assert phase3_estimator_plot_label(estimator) in plot
        assert f">{counts['detection_rate']:.3f}</text>" in plot


def test_pivotal_action_detection_plot_reports_real_detection_not_coverage(tmp_path):
    dataset_path = _write_phase3_dataset(tmp_path)
    output_directory = tmp_path / "artifact"
    arguments = build_arg_parser().parse_args(
        _phase3_cli_arguments(dataset_path, output_directory)
    )

    artifact = run_stratified_benchmark(arguments, scorer=_FakeScorer())

    detection = artifact["pivotal_action_detection"]
    assert set(detection) == set(PHASE3_ESTIMATOR_ORDER)
    for counts in detection.values():
        assert counts["pivotal_state_count"] <= sum(
            coverage["selected_state_count"]
            for coverage in artifact["defined_state_coverage"]["uniform"].values()
        )
        assert counts["detected_state_count"] <= counts["pivotal_state_count"]


def test_phase3_runpod_command_targets_stratified_directory():
    launcher = PHASE3_RUNPOD_LAUNCHER.read_text()

    assert "--benchmark stratified" in launcher
    assert "--output /workspace/results/stratified" in launcher
    assert "--batch-size 1" in launcher
    assert "NETWORK_VOLUME_ID" in launcher


def _isolated_benchmark_a(tmp_path):
    """Build a miniature tracked Benchmark A root with its own hash manifest."""
    benchmark_a_root = tmp_path / "credit-estimator-pilot"
    (benchmark_a_root / "qwen3.5-4b" / "problems").mkdir(parents=True)
    files = {
        "qwen3.5-4b/phase3_credit_estimator_pilot.json": '{"status": "completed"}\n',
        "qwen3.5-4b/problems/pilot-root.json": '{"canonical_hash": "pilot-root"}\n',
    }
    for relative_path, contents in files.items():
        (benchmark_a_root / relative_path).write_text(contents)
    (benchmark_a_root / "benchmark_a_sha256sums.txt").write_text(
        "".join(
            f"{hashlib.sha256(contents.encode()).hexdigest()}  {relative_path}\n"
            for relative_path, contents in files.items()
        )
    )
    return benchmark_a_root


def _launcher_environment(benchmark_a_root, tmp_path, **overrides):
    return {
        **os.environ,
        "BENCHMARK_A_ROOT": str(benchmark_a_root),
        "STRATIFIED_DESTINATION": str(benchmark_a_root / "qwen3.5-4b" / "stratified"),
        "STAGING_ROOT": str(tmp_path / "staging"),
        "RUNPOD_STATE_DIR": str(tmp_path / "runpod-state"),
        **overrides,
    }


def _benchmark_a_snapshot(benchmark_a_root):
    return {
        str(path.relative_to(benchmark_a_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(benchmark_a_root.rglob("*"))
        if path.is_file() and "stratified" not in path.relative_to(benchmark_a_root).parts
    }


def test_phase3_launcher_verifies_the_tracked_benchmark_a_manifest(tmp_path):
    benchmark_a_root = _isolated_benchmark_a(tmp_path)
    environment = _launcher_environment(benchmark_a_root, tmp_path)

    verified = subprocess.run(
        [str(PHASE3_RUNPOD_LAUNCHER), "verify-benchmark-a"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    (benchmark_a_root / "qwen3.5-4b" / "phase3_credit_estimator_pilot.json").write_text(
        '{"status": "tampered"}\n'
    )
    corrupted = subprocess.run(
        [str(PHASE3_RUNPOD_LAUNCHER), "verify-benchmark-a"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert verified.returncode == 0
    assert corrupted.returncode == 3
    assert "manifest verification failed" in corrupted.stderr


def test_phase3_launcher_rejects_benchmark_a_files_the_manifest_omits(tmp_path):
    benchmark_a_root = _isolated_benchmark_a(tmp_path)
    environment = _launcher_environment(benchmark_a_root, tmp_path)
    # Every listed file still hashes correctly; only the path set changed.
    (benchmark_a_root / "qwen3.5-4b" / "problems" / "smuggled.json").write_text("{}\n")

    completed_process = subprocess.run(
        [str(PHASE3_RUNPOD_LAUNCHER), "verify-benchmark-a"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 3
    assert "files the manifest does not list" in completed_process.stderr
    assert "qwen3.5-4b/problems/smuggled.json" in completed_process.stderr


def test_phase3_launcher_promotes_only_the_stratified_subtree(tmp_path):
    benchmark_a_root = _isolated_benchmark_a(tmp_path)
    before = _benchmark_a_snapshot(benchmark_a_root)
    staging_directory = tmp_path / "staging" / "phase3-credit-estimator-pilot"
    (staging_directory / "stratified" / "problems").mkdir(parents=True)
    (staging_directory / "stratified" / "problems" / "pilot-root.json").write_text("{}\n")
    (staging_directory / "stratified" / "run_manifest.json").write_text("{}\n")
    # An archive that also carries Benchmark A paths must not be able to reach them.
    (staging_directory / "qwen3.5-4b").mkdir(parents=True)
    (staging_directory / "qwen3.5-4b" / "phase3_credit_estimator_pilot.json").write_text(
        '{"status": "overwritten"}\n'
    )

    completed_process = subprocess.run(
        [str(PHASE3_RUNPOD_LAUNCHER), "promote", str(staging_directory)],
        env=_launcher_environment(benchmark_a_root, tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0
    destination = benchmark_a_root / "qwen3.5-4b" / "stratified"
    assert (destination / "problems" / "pilot-root.json").is_file()
    assert (destination / "run_manifest.json").is_file()
    assert _benchmark_a_snapshot(benchmark_a_root) == before


def test_phase3_launcher_promotion_replaces_rather_than_merges_the_destination(
    tmp_path,
):
    benchmark_a_root = _isolated_benchmark_a(tmp_path)
    destination = benchmark_a_root / "qwen3.5-4b" / "stratified"
    (destination / "problems").mkdir(parents=True)
    (destination / "problems" / "retired-problem.json").write_text('{"stale": true}\n')
    (destination / "phase3_credit_estimator_stratified.json").write_text(
        '{"status": "stale"}\n'
    )
    staging_directory = tmp_path / "staging" / "phase3-credit-estimator-pilot"
    (staging_directory / "stratified" / "problems").mkdir(parents=True)
    (staging_directory / "stratified" / "problems" / "pilot-root.json").write_text("{}\n")
    (staging_directory / "stratified" / "run_manifest.json").write_text("{}\n")

    completed_process = subprocess.run(
        [str(PHASE3_RUNPOD_LAUNCHER), "promote", str(staging_directory)],
        env=_launcher_environment(benchmark_a_root, tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 0
    assert {
        str(path.relative_to(destination))
        for path in destination.rglob("*")
        if path.is_file()
    } == {"problems/pilot-root.json", "run_manifest.json"}
    assert not list(benchmark_a_root.rglob("*.incoming.*"))
    assert not list(benchmark_a_root.rglob("*.replaced.*"))


@pytest.mark.parametrize(
    ("build_staging", "expected_error"),
    [
        (lambda staging: staging.mkdir(parents=True), "no complete stratified subtree"),
        (
            lambda staging: (
                staging.mkdir(parents=True),
                (staging / "stratified").symlink_to(staging),
            ),
            "symbolic link",
        ),
    ],
)
def test_phase3_launcher_refuses_to_promote_a_partial_or_unsafe_archive(
    tmp_path, build_staging, expected_error
):
    benchmark_a_root = _isolated_benchmark_a(tmp_path)
    before = _benchmark_a_snapshot(benchmark_a_root)
    staging_directory = tmp_path / "staging" / "phase3-credit-estimator-pilot"
    build_staging(staging_directory)

    completed_process = subprocess.run(
        [str(PHASE3_RUNPOD_LAUNCHER), "promote", str(staging_directory)],
        env=_launcher_environment(benchmark_a_root, tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 3
    assert expected_error in completed_process.stderr
    assert _benchmark_a_snapshot(benchmark_a_root) == before
    assert not (benchmark_a_root / "qwen3.5-4b" / "stratified").exists()


def test_phase3_launcher_refuses_to_promote_onto_an_already_corrupt_benchmark_a(tmp_path):
    benchmark_a_root = _isolated_benchmark_a(tmp_path)
    (benchmark_a_root / "qwen3.5-4b" / "phase3_credit_estimator_pilot.json").write_text(
        '{"status": "tampered"}\n'
    )
    staging_directory = tmp_path / "staging" / "phase3-credit-estimator-pilot"
    (staging_directory / "stratified").mkdir(parents=True)
    (staging_directory / "stratified" / "run_manifest.json").write_text("{}\n")

    completed_process = subprocess.run(
        [str(PHASE3_RUNPOD_LAUNCHER), "promote", str(staging_directory)],
        env=_launcher_environment(benchmark_a_root, tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 3
    assert not (benchmark_a_root / "qwen3.5-4b" / "stratified").exists()


def test_phase3_failed_run_downloads_into_staging_and_leaves_benchmark_a_intact(tmp_path):
    benchmark_a_root = _isolated_benchmark_a(tmp_path)
    before = _benchmark_a_snapshot(benchmark_a_root)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_runpodctl = fake_bin / "runpodctl"
    fake_runpodctl.write_text(
        """#!/usr/bin/env python3
import json
import sys

if sys.argv[1:3] == ["pod", "create"]:
    print(json.dumps({"id": "pod-123"}))
elif sys.argv[1:3] == ["pod", "logs"]:
    print("CUDA out of memory")
"""
    )
    fake_runpodctl.chmod(0o755)
    state_directory = tmp_path / "runpod-state"
    state_directory.mkdir()
    (state_directory / "image-digest.txt").write_text(
        "registry.example/research@sha256:" + "a" * 64 + "\n"
    )

    completed_process = subprocess.run(
        [str(PHASE3_RUNPOD_LAUNCHER), "run"],
        env=_launcher_environment(
            benchmark_a_root,
            tmp_path,
            PATH=f"{fake_bin}:{os.environ['PATH']}",
            NETWORK_VOLUME_ID="phase3-volume",
            PROJECTED_HOURS="0.25",
            HOURLY_RATE_USD="2",
            RUNPOD_MAX_HOURS="6",
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 1
    pod_state = json.loads(
        (state_directory / "phase3-credit-estimator-pilot.json").read_text()
    )
    assert pod_state["local_output"] == str(
        tmp_path / "staging" / "phase3-credit-estimator-pilot"
    )
    assert str(benchmark_a_root) not in pod_state["local_output"]
    assert _benchmark_a_snapshot(benchmark_a_root) == before
    assert not (benchmark_a_root / "qwen3.5-4b" / "stratified").exists()


@pytest.mark.parametrize(
    ("image_override", "expected_error"),
    [
        (None, "immutable image digest"),
        ("registry.example/research:latest", "immutable image digest"),
        ("registry.example/research:v1", "immutable image digest"),
    ],
)
def test_phase3_runpod_run_rejects_unpinned_image_references(
    tmp_path, image_override, expected_error
):
    environment = {
        **os.environ,
        "NETWORK_VOLUME_ID": "phase3-volume",
        "PROJECTED_HOURS": "0.25",
        "HOURLY_RATE_USD": "2",
        "RUNPOD_STATE_DIR": str(tmp_path / "runpod-state"),
    }
    environment.pop("IMAGE", None)
    if image_override is not None:
        environment["IMAGE"] = image_override

    completed_process = subprocess.run(
        [str(PHASE3_RUNPOD_LAUNCHER), "run"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 2
    assert expected_error in completed_process.stderr


@pytest.mark.parametrize("maximum_hours", ("12", "0", "not-a-number"))
def test_phase3_runpod_run_rejects_invalid_maximum_hours(tmp_path, maximum_hours):
    state_directory = tmp_path / "runpod-state"
    state_directory.mkdir()
    (state_directory / "image-digest.txt").write_text(
        "registry.example/research@sha256:" + "a" * 64 + "\n"
    )

    completed_process = subprocess.run(
        [str(PHASE3_RUNPOD_LAUNCHER), "run"],
        env={
            **os.environ,
            "NETWORK_VOLUME_ID": "phase3-volume",
            "PROJECTED_HOURS": "0.25",
            "HOURLY_RATE_USD": "2",
            "RUNPOD_MAX_HOURS": maximum_hours,
            "RUNPOD_STATE_DIR": str(state_directory),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 2
    assert "between 1 and 6" in completed_process.stderr


def test_phase3_runpod_run_launches_with_a_six_hour_termination_bound(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_arguments = tmp_path / "runpodctl-args.json"
    fake_runpodctl = fake_bin / "runpodctl"
    fake_runpodctl.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if sys.argv[1:3] == ["pod", "create"]:
    Path(os.environ["CAPTURED_ARGS"]).write_text(json.dumps(sys.argv[1:]))
    print(json.dumps({"id": "pod-123"}))
elif sys.argv[1:3] == ["pod", "logs"]:
    print("CUDA out of memory")
"""
    )
    fake_runpodctl.chmod(0o755)
    state_directory = tmp_path / "runpod-state"
    state_directory.mkdir()
    (state_directory / "image-digest.txt").write_text(
        "registry.example/research@sha256:" + "a" * 64 + "\n"
    )
    started_at = datetime.now(UTC)

    completed_process = subprocess.run(
        [str(PHASE3_RUNPOD_LAUNCHER), "run"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CAPTURED_ARGS": str(captured_arguments),
            "NETWORK_VOLUME_ID": "phase3-volume",
            "PROJECTED_HOURS": "0.25",
            "HOURLY_RATE_USD": "2",
            "RUNPOD_MAX_HOURS": "6",
            "RUNPOD_STATE_DIR": str(state_directory),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 1
    arguments = json.loads(captured_arguments.read_text())
    terminate_after = arguments[arguments.index("--terminate-after") + 1]
    deadline = datetime.fromisoformat(terminate_after)
    assert deadline <= started_at + timedelta(hours=6, seconds=1)


def test_phase3_runpod_preflight_refuses_a_50_problem_run_over_the_budget():
    completed_process = subprocess.run(
        [
            str(PHASE3_RUNPOD_LAUNCHER),
            "preflight",
            "--problem-count",
            "50",
            "--projected-hours",
            "3",
            "--hourly-rate-usd",
            "20",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed_process.returncode == 2
    assert "projected cost: $60.00" in completed_process.stderr
    assert "$50" in completed_process.stderr


def test_phase3_runpod_preflight_reports_a_non_paid_smoke_projection():
    completed_process = subprocess.run(
        [
            str(PHASE3_RUNPOD_LAUNCHER),
            "preflight",
            "--problem-count",
            "1",
            "--projected-hours",
            "0.25",
            "--hourly-rate-usd",
            "2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "projected cost: $0.50" in completed_process.stdout
    assert "No Pod was launched" in completed_process.stdout
