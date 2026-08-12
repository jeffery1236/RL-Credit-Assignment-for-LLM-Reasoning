"""Tests for analysis/artifacts.py: canonical JSON/DOT generation and
byte-identical regeneration.
"""

import csv
import json
from fractions import Fraction
from pathlib import Path

from analysis.artifacts import (
    build_dag_artifact,
    flatten_problem_profile,
    to_canonical_json,
    to_dot,
    write_csv_rows,
    write_json_rows,
)
from analysis.profiles import compute_problem_profile
from env.countdown import Action, Operator, Problem
from env.generator import canonical_problem_hash
from oracle.evaluation import (
    ExactPolicyEvaluator,
    build_reasoning_dag,
    compute_min_steps_to_success,
    compute_q_star,
    compute_v_star,
)
from policies.baselines import UniformPolicy

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _small_artifact() -> dict:
    problem = Problem(values=(1, 2), target=3)
    return build_dag_artifact(
        problem.to_state(),
        policies={"uniform": UniformPolicy()},
        generator_version="fixture-v1",
    )


# ---------------------------------------------------------------------------
# Determinism: regenerating the same artifact twice is byte-identical
# ---------------------------------------------------------------------------


def test_json_regeneration_is_byte_identical():
    a = to_canonical_json(_small_artifact())
    b = to_canonical_json(_small_artifact())
    assert a == b


def test_dot_regeneration_is_byte_identical():
    a = to_dot(_small_artifact())
    b = to_dot(_small_artifact())
    assert a == b


# ---------------------------------------------------------------------------
# Snapshot tests against committed fixture files
# ---------------------------------------------------------------------------


def test_json_matches_committed_snapshot():
    expected = (FIXTURES_DIR / "small_dag.json").read_text()
    actual = to_canonical_json(_small_artifact())
    assert actual == expected


def test_dot_matches_committed_snapshot():
    expected = (FIXTURES_DIR / "small_dag.dot").read_text()
    actual = to_dot(_small_artifact())
    assert actual == expected


# ---------------------------------------------------------------------------
# Round-trip: parsing the JSON back out reproduces the same oracle values
# ---------------------------------------------------------------------------


def test_json_round_trip_reproduces_v_star_q_star_and_advantage():
    problem = Problem(values=(1, 2), target=3)
    root = problem.to_state()
    dag = build_reasoning_dag(root)
    evaluator = ExactPolicyEvaluator(UniformPolicy(), dag)

    artifact = build_dag_artifact(
        root, policies={"uniform": UniformPolicy()}, generator_version="roundtrip-test"
    )
    parsed = json.loads(to_canonical_json(artifact))

    root_node = next(n for n in parsed["nodes"] if n["id"] == 0)
    assert root_node["v_star"] == 1
    assert root_node["values"] == [1, 2]

    add_edge = next(
        e
        for e in parsed["edges"]
        if e["source_id"] == 0 and e["operator"] == "+" and e["left"] == 1 and e["right"] == 2
    )
    assert add_edge["q_star"] == 1

    add_action = Action(Operator.ADD, 1, 2)
    expected_advantage = evaluator.a_pi(root, add_action)
    parsed_advantage = Fraction(
        add_edge["policy_advantage"]["uniform"]["num"],
        add_edge["policy_advantage"]["uniform"]["den"],
    )
    assert parsed_advantage == expected_advantage == Fraction(2, 3)

    mul_edge = next(
        e
        for e in parsed["edges"]
        if e["source_id"] == 0 and e["operator"] == "*" and e["left"] == 1 and e["right"] == 2
    )
    assert mul_edge["q_star"] == 0
    parsed_mul_advantage = Fraction(
        mul_edge["policy_advantage"]["uniform"]["num"],
        mul_edge["policy_advantage"]["uniform"]["den"],
    )
    assert parsed_mul_advantage == Fraction(-1, 3)


def test_json_round_trip_win_probability():
    problem = Problem(values=(1, 2), target=3)
    root = problem.to_state()
    artifact = build_dag_artifact(
        root, policies={"uniform": UniformPolicy()}, generator_version="roundtrip-test"
    )
    parsed = json.loads(to_canonical_json(artifact))
    root_node = next(n for n in parsed["nodes"] if n["id"] == 0)
    win_prob = root_node["policy_win_probability"]["uniform"]
    assert Fraction(win_prob["num"], win_prob["den"]) == Fraction(1, 3)


# ---------------------------------------------------------------------------
# Structural sanity on the DOT output
# ---------------------------------------------------------------------------


def test_dot_preserves_parallel_edges_for_2_2_fixture():
    problem = Problem(values=(2, 2), target=99)
    artifact = build_dag_artifact(
        problem.to_state(), policies={"uniform": UniformPolicy()}
    )
    dot_text = to_dot(artifact)
    # Both 2+2 and 2*2 land on the same child -- both edges must appear.
    assert dot_text.count("2 + 2 = 4") == 1
    assert dot_text.count("2 * 2 = 4") == 1
    assert artifact["edge_count"] == 2


def test_artifact_includes_optimal_and_policy_profiles_as_separate_sections():
    artifact = _small_artifact()
    root_node = artifact["nodes"][0]
    assert "optimal_action_profile" in root_node
    assert "policy_credit_profiles" in root_node
    # Structural enforcement of Part 2.5: these must be separate sections,
    # not merged into one blob.
    assert set(root_node["optimal_action_profile"].keys()).isdisjoint(
        root_node["policy_credit_profiles"]["uniform"].keys()
    )


# ---------------------------------------------------------------------------
# ProblemProfile flattening + CSV/JSON row output
# ---------------------------------------------------------------------------


def _small_problem_row() -> dict:
    problem = Problem(values=(1, 2), target=3)
    root = problem.to_state()
    dag = build_reasoning_dag(root)
    v_star = compute_v_star(dag)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)
    evaluators = {"uniform": ExactPolicyEvaluator(UniformPolicy(), dag)}
    profile = compute_problem_profile(dag, v_star, q_star, min_steps, evaluators)
    return flatten_problem_profile(2, problem, canonical_problem_hash(problem), profile)


def test_flatten_problem_profile_has_expected_scalar_fields():
    row = _small_problem_row()
    assert row["n_values"] == 2
    assert row["values"] == "1,2"
    assert row["target"] == 3
    assert row["solvable"] is True
    assert row["node_count"] == 4
    assert row["min_solution_depth"] == 1
    assert row["unweighted_uniform_positive_count"] == 1
    assert row["unweighted_uniform_negative_count"] == 2
    # Occupancy-weighted keys are absent entirely when not requested --
    # flatten_problem_profile never invents zero-filled columns for data
    # that was never computed.
    assert not any(k.startswith("occupancy_weighted_") for k in row)


def test_flatten_problem_profile_values_are_json_safe():
    row = _small_problem_row()
    text = json.dumps(row)  # must not raise (no Fraction, no Action, etc.)
    assert json.loads(text) == row


def test_write_json_rows_round_trips(tmp_path):
    rows = [_small_problem_row()]
    path = tmp_path / "rows.json"
    write_json_rows(rows, path)
    loaded = json.loads(path.read_text())
    assert loaded == rows


def test_write_csv_rows_round_trips(tmp_path):
    rows = [_small_problem_row()]
    path = tmp_path / "rows.csv"
    write_csv_rows(rows, path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        loaded_rows = list(reader)
    assert len(loaded_rows) == 1
    assert loaded_rows[0]["target"] == "3"  # CSV round-trips as strings
    assert loaded_rows[0]["values"] == "1,2"


def test_write_csv_rows_handles_heterogeneous_columns(tmp_path):
    rows = [{"a": 1, "b": 2}, {"a": 3, "c": 4}]
    path = tmp_path / "rows.csv"
    write_csv_rows(rows, path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["a", "b", "c"]
        loaded_rows = list(reader)
    assert loaded_rows[0] == {"a": "1", "b": "2", "c": ""}
    assert loaded_rows[1] == {"a": "3", "b": "", "c": "4"}
