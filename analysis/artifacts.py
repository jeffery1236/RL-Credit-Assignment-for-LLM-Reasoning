"""Canonical, byte-reproducible JSON and DOT artifacts for one Countdown
problem's complete reasoning DAG.

The JSON and DOT outputs are built from the SAME in-memory artifact
dictionary, so they can never disagree about node IDs, edges, or values.

Canonical means byte-identical regeneration: every collection is emitted in
a fixed, sorted order (never a bare set or unordered dict iteration), and
JSON is dumped with sort_keys=True and a fixed separator/indent style. This
matters because Phase 1's determinism check (Part 5 of the plan) is
literally "regenerate everything and confirm `git status` is empty."

Fraction values are serialized as {"num": ..., "den": ...} HERE, at the
artifact boundary only -- this encoding must never leak into the Policy
protocol or the oracle, which stay generic over `Real`.

DAG artifact schema: nodes hold AGGREGATES only, edges are the single
source of truth for per-action facts. Every action already appears as an
edge with its own q_star/policy_action_probability/policy_advantage, so a
node's profile does NOT repeat a full action-by-action breakdown -- only
counts, fragility, and mean/max summaries. Earlier revisions duplicated
the full per-action Q*/probability/advantage mapping on BOTH the node and
every one of its outgoing edges, which roughly doubled artifact size for
no benefit (1.3MB for a 236-node/420-edge example) and created two sources
of truth for the same numbers. If you need "which actions are ruining from
this node," filter its outgoing edges for q_star == 0 (only meaningful
when the node's own v_star == 1) rather than adding a list back here. See
docs/project_plan.md's DAG artifact schema note for the full rationale.
"""

from __future__ import annotations

import csv
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Mapping

from analysis.profiles import (
    OptimalActionProfile,
    PolicyCreditProfile,
    ProblemProfile,
    compute_optimal_action_profile,
    compute_policy_credit_profile,
)
from env.countdown import (
    DEFAULT_RULES,
    Problem,
    RuleSet,
    State,
    is_success,
    is_terminal,
)
from oracle.evaluation import (
    ExactPolicyEvaluator,
    Policy,
    build_reasoning_dag,
    compute_min_steps_to_success,
    compute_q_star,
    compute_v_star,
)

ARTIFACT_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Fraction serialization + aggregate-only profile serializers
# ---------------------------------------------------------------------------


def _serialize_fraction(value: Fraction) -> dict:
    return {"num": value.numerator, "den": value.denominator}


def _serialize_optimal_action_profile(profile: OptimalActionProfile) -> dict:
    """Aggregate-only. The exact Q* for each action is NOT repeated here --
    it lives on this state's outgoing edges (see module docstring).
    """
    return {
        "allowed_action_count": profile.allowed_action_count,
        "solution_preserving_count": profile.solution_preserving_count,
        "ruining_action_count": len(profile.ruining_actions),
        "fraction_of_moves_that_ruin_solution": profile.fraction_of_moves_that_ruin_solution,
    }


def _serialize_policy_credit_profile(profile: PolicyCreditProfile) -> dict:
    """Aggregate-only -- see _serialize_optimal_action_profile and the
    module docstring. Each action's exact probability/advantage lives on
    the corresponding edge, not duplicated here.
    """
    return {
        "negative_count": profile.negative_count,
        "zero_count": profile.zero_count,
        "positive_count": profile.positive_count,
        "mean_abs_advantage": _serialize_fraction(profile.mean_abs_advantage),
        "max_abs_advantage": _serialize_fraction(profile.max_abs_advantage),
    }


# ---------------------------------------------------------------------------
# Building the artifact
# ---------------------------------------------------------------------------


def build_dag_artifact(
    root: State,
    policies: Mapping[str, "Policy[Fraction]"],
    rules: RuleSet = DEFAULT_RULES,
    generator_version: str = "manual",
    max_nodes: int = 100_000,
    max_edges: int = 1_000_000,
) -> dict:
    """Build the complete, canonical artifact dictionary for `root`'s
    reasoning DAG: every node, every edge (including parallel edges),
    V*/Q*/min_steps_to_success, and every named policy's exact Vpi/action
    probabilities/Api. `policies` keys become the stable names used
    throughout the artifact (e.g. "uniform", "greedy_distance").
    """
    dag = build_reasoning_dag(root, rules, max_nodes, max_edges)
    v_star = compute_v_star(dag, rules)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)

    # Stable node IDs: sequential integers assigned in the DAG's own
    # deterministic traversal order (sorted by (depth, values)).
    node_order = dag.topological_order()
    node_id = {state: index for index, state in enumerate(node_order)}

    evaluators = {
        name: ExactPolicyEvaluator(policy, dag, rules)
        for name, policy in sorted(policies.items())
    }
    policy_names = sorted(evaluators.keys())

    nodes_json = []
    for state in node_order:
        optimal_profile = compute_optimal_action_profile(state, v_star, q_star, min_steps)
        policy_credit_profiles = {
            name: compute_policy_credit_profile(state, evaluators[name], rules)
            for name in policy_names
        }
        nodes_json.append(
            {
                "id": node_id[state],
                "values": list(state.values),
                "target": state.target,
                "depth": dag.nodes[state].depth,
                "is_terminal": is_terminal(state, rules),
                "is_success": is_success(state),
                "v_star": v_star[state],
                "min_steps_to_success": min_steps[state],
                "optimal_action_profile": _serialize_optimal_action_profile(
                    optimal_profile
                ),
                "policy_win_probability": {
                    name: _serialize_fraction(evaluators[name].v_pi(state))
                    for name in policy_names
                },
                "policy_credit_profiles": {
                    name: _serialize_policy_credit_profile(policy_credit_profiles[name])
                    for name in policy_names
                },
            }
        )

    edges_json = []
    for state in node_order:
        q_star_for_state = q_star.get(state, {})
        edges = sorted(dag.out_edges(state), key=lambda e: e.action.sort_key())
        for edge in edges:
            edges_json.append(
                {
                    "source_id": node_id[edge.source],
                    "target_id": node_id[edge.target_state],
                    "operator": edge.action.operator.value,
                    "left": edge.action.left,
                    "right": edge.action.right,
                    "result": edge.action.result,
                    "q_star": q_star_for_state[edge.action],
                    "policy_action_probability": {
                        name: _serialize_fraction(
                            evaluators[name].action_probs(state)[edge.action]
                        )
                        for name in policy_names
                    },
                    "policy_advantage": {
                        name: _serialize_fraction(
                            evaluators[name].a_pi(state, edge.action)
                        )
                        for name in policy_names
                    },
                }
            )

    return {
        "artifact_version": ARTIFACT_VERSION,
        "generator_version": generator_version,
        "problem": {"values": list(root.values), "target": root.target},
        "rules": {
            "operators": [op.value for op in rules.operators],
            "min_intermediate": rules.min_intermediate,
        },
        "policy_names": policy_names,
        "node_count": dag.node_count,
        "edge_count": dag.edge_count,
        "nodes": nodes_json,
        "edges": edges_json,
    }


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def to_canonical_json(artifact: dict) -> str:
    """sort_keys=True is a safety net on top of the already-deterministic
    list ordering built into `artifact` -- together they guarantee
    byte-identical regeneration.
    """
    return json.dumps(artifact, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def write_json(artifact: dict, path: Path) -> None:
    path.write_text(to_canonical_json(artifact))


# ---------------------------------------------------------------------------
# DOT output
# ---------------------------------------------------------------------------


_NODE_COLOR_SUCCESS = "palegreen"
_NODE_COLOR_TERMINAL_FAILURE = "lightcoral"
_NODE_COLOR_SOLVABLE = "honeydew"
_NODE_COLOR_UNSOLVABLE = "mistyrose"


def _node_color(node: dict) -> str:
    if node["is_success"]:
        return _NODE_COLOR_SUCCESS
    if node["is_terminal"]:
        return _NODE_COLOR_TERMINAL_FAILURE
    return _NODE_COLOR_SOLVABLE if node["v_star"] == 1 else _NODE_COLOR_UNSOLVABLE


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def to_dot(artifact: dict) -> str:
    """Build DOT text from the SAME artifact dict used for JSON, so the two
    representations of a problem can never disagree. Nodes are grouped
    into {rank=same; ...} blocks by depth so the rendered graph reads
    top-to-bottom as the reasoning unfolds. Parallel edges (two actions
    landing on the same child) are both emitted.
    """
    lines = [
        "digraph reasoning_dag {",
        "  rankdir=TB;",
        '  node [shape=box, fontname="monospace", fontsize=10];',
        '  edge [fontname="monospace", fontsize=9];',
    ]

    nodes_by_depth: dict[int, list[dict]] = {}
    for node in artifact["nodes"]:
        nodes_by_depth.setdefault(node["depth"], []).append(node)

    for node in artifact["nodes"]:
        bag = ", ".join(str(v) for v in node["values"])
        first_line = _dot_escape(f"{{{bag}}}  target={node['target']}")
        second_line = _dot_escape(f"V*={node['v_star']}")
        # The literal two-character sequence \n (added AFTER escaping) is
        # Graphviz's own label line-break syntax -- if this were run through
        # _dot_escape too, the backslash would get doubled and Graphviz
        # would render the literal text "\n" instead of a line break.
        label = f"{first_line}\\n{second_line}"
        lines.append(
            f'  n{node["id"]} [label="{label}", style=filled, '
            f'fillcolor="{_node_color(node)}"];'
        )

    for depth in sorted(nodes_by_depth):
        ids = " ".join(f'n{n["id"]}' for n in sorted(nodes_by_depth[depth], key=lambda n: n["id"]))
        lines.append(f"  {{ rank=same; {ids} }}")

    sorted_edges = sorted(
        artifact["edges"],
        key=lambda e: (e["source_id"], e["operator"], e["left"], e["right"], e["target_id"]),
    )
    for edge in sorted_edges:
        label = _dot_escape(
            f'{edge["left"]} {edge["operator"]} {edge["right"]} = {edge["result"]}'
        )
        lines.append(f'  n{edge["source_id"]} -> n{edge["target_id"]} [label="{label}"];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_dot(artifact: dict, path: Path) -> None:
    path.write_text(to_dot(artifact))


def render_svg(dot_path: Path, svg_path: Path) -> bool:
    """Shell out to the native `dot` binary if it's installed. Returns
    False (and prints a clear message) rather than raising if Graphviz
    isn't available -- SVG rendering is a nice-to-have, not a dependency.
    """
    try:
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True
        )
        return True
    except FileNotFoundError:
        print(
            "Graphviz 'dot' binary not found on PATH -- skipping SVG rendering. "
            "Install Graphviz (e.g. `brew install graphviz`) to enable it."
        )
        return False


# ---------------------------------------------------------------------------
# ProblemProfile flattening + CSV output (for bulk exploratory studies,
# e.g. experiments/profile_environment.py -- NOT the per-problem DAG
# artifact above, which stays a one-off, illustrative-only structure)
# ---------------------------------------------------------------------------


def flatten_problem_profile(
    n_values: int,
    problem: Problem,
    canonical_hash: str,
    profile: ProblemProfile,
) -> dict:
    """Flatten one problem's ProblemProfile into a single scalar-valued
    row, suitable for both a JSON list entry and a CSV row. Column names
    are explicit and stable so an analysis notebook can load this
    directly (e.g. pandas.read_csv) with no nested structure to unpack.
    """
    row: dict = {
        "n_values": n_values,
        "values": ",".join(str(v) for v in problem.values),
        "target": problem.target,
        "canonical_hash": canonical_hash,
        "solvable": profile.solvable,
        "initially_solved": profile.initially_solved,
        "node_count": profile.node_count,
        "edge_count": profile.edge_count,
        "branching_mean": profile.branching.mean if profile.branching else None,
        "branching_min": profile.branching.minimum if profile.branching else None,
        "branching_max": profile.branching.maximum if profile.branching else None,
        "min_solution_depth": profile.min_solution_depth,
        "solution_preserving_root_action_count": profile.solution_preserving_root_action_count,
        "ruining_move_state_count": profile.ruining_move_state_count,
        "ruining_move_state_rate": profile.ruining_move_state_rate,
        "ruining_action_count": profile.ruining_action_count,
        "ruining_action_rate": profile.ruining_action_rate,
        "fraction_of_moves_that_ruin_solution_mean": profile.fraction_of_moves_that_ruin_solution_mean,
    }

    for bucket_label, count in sorted(
        profile.fraction_of_moves_that_ruin_solution_histogram.items()
    ):
        row[f"fragility_histogram_{bucket_label}"] = count

    for policy_name in sorted(profile.policy_advantage_summaries):
        summary = profile.policy_advantage_summaries[policy_name]
        prefix = f"unweighted_{policy_name}"
        row[f"{prefix}_negative_count"] = summary.negative_count
        row[f"{prefix}_zero_count"] = summary.zero_count
        row[f"{prefix}_positive_count"] = summary.positive_count
        row[f"{prefix}_mean_abs_advantage"] = summary.mean_abs_advantage

    for policy_name in sorted(profile.occupancy_weighted_advantage_summaries):
        summary = profile.occupancy_weighted_advantage_summaries[policy_name]
        prefix = f"occupancy_weighted_{policy_name}"
        row[f"{prefix}_negative_mass"] = summary.negative_mass
        row[f"{prefix}_zero_mass"] = summary.zero_mass
        row[f"{prefix}_positive_mass"] = summary.positive_mass
        row[f"{prefix}_mean_abs_advantage"] = summary.mean_abs_advantage

    return row


def write_json_rows(rows: list[dict], path: Path) -> None:
    path.write_text(json.dumps(rows, sort_keys=True, indent=2, separators=(",", ": ")) + "\n")


def write_csv_rows(rows: list[dict], path: Path) -> None:
    """Canonical CSV: a fixed, sorted column order (the union of every
    row's keys), so re-running the same generation produces a
    byte-identical file. Rows may have different keys (e.g. different
    policy names across strata) -- DictWriter fills missing ones blank.
    """
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
