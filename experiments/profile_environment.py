"""Generate the 750-problem exploratory manifest (250 each of 4/5/6-number
problems) and report structural + policy credit-assignment diagnostics.

This is Step 7 of the Phase 1 plan -- the go/no-go gate. It is a
MEASUREMENT pass, not a search: generation settings are broad and fixed,
and results are reported as-is against predeclared diagnostic bands (see
docs/project_plan.md Section 16 / the Phase 1 implementation plan Part
4 Step 7c). Do not tune parameters based on what this script prints.

Deliberately uses analysis/profiles.py's lightweight compute_problem_profile
for every candidate, NOT analysis/artifacts.py's build_dag_artifact -- see
docs/project_plan.md Section 31.1 for why building a full per-node/per-edge
DAG artifact for each of 750 problems would be the wrong tool here.

Usage:
    uv run python -m experiments.profile_environment --output artifacts/phase1/exploratory
"""

import argparse
import json
import time
from pathlib import Path

from analysis.artifacts import flatten_problem_profile, write_csv_rows, write_json_rows
from analysis.profiles import compute_problem_profile
from env.countdown import DEFAULT_RULES, RuleSet
from env.generator import GenerationConfig, generate_candidates
from oracle.evaluation import (
    DagLimitExceeded,
    ExactPolicyEvaluator,
    build_reasoning_dag,
    compute_min_steps_to_success,
    compute_q_star,
    compute_v_star,
    compute_visit_probabilities,
)
from policies.baselines import EpsilonGreedyDistancePolicy, GreedyDistancePolicy, UniformPolicy

GENERATOR_VERSION = "v1.0-exploratory"
VALUE_POOL = tuple(range(1, 11)) + (25, 50, 75, 100)
TARGET_RANGE = (100, 999)
STRATA = (4, 5, 6)
MAX_NODES = 100_000

POLICY_FACTORIES = {
    "uniform": UniformPolicy,
    "greedy_distance": GreedyDistancePolicy,
    "epsilon_greedy_distance": EpsilonGreedyDistancePolicy,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for exploratory JSON/CSV summaries.",
    )
    parser.add_argument(
        "--per-stratum",
        type=int,
        default=250,
        help="Number of candidates to generate per number-count stratum.",
    )
    return parser


def profile_one_candidate(candidate, rules: RuleSet = DEFAULT_RULES):
    """Returns (profile: ProblemProfile | None, hit_node_limit: bool)."""
    root = candidate.problem.to_state()
    try:
        dag = build_reasoning_dag(root, rules, max_nodes=MAX_NODES)
    except DagLimitExceeded:
        return None, True

    v_star = compute_v_star(dag, rules)
    q_star = compute_q_star(dag, v_star)
    min_steps = compute_min_steps_to_success(dag, v_star, q_star)

    evaluators = {
        name: ExactPolicyEvaluator(factory(), dag, rules)
        for name, factory in POLICY_FACTORIES.items()
    }
    visit_probabilities = {
        name: compute_visit_probabilities(dag, evaluator, rules)
        for name, evaluator in evaluators.items()
    }

    profile = compute_problem_profile(
        dag, v_star, q_star, min_steps, evaluators, rules, visit_probabilities
    )
    return profile, False


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    node_limit_hits = 0
    total_candidates = 0
    depths_by_stratum: dict[int, set[int]] = {n: set() for n in STRATA}

    start = time.time()
    for n_values in STRATA:
        config = GenerationConfig(
            n_values=n_values,
            value_pool=VALUE_POOL,
            target_range=TARGET_RANGE,
            generator_version=GENERATOR_VERSION,
            seed_namespace=f"exploratory-{n_values}",
        )
        candidates = generate_candidates(config, args.per_stratum)
        print(f"[n={n_values}] generated {len(candidates)} unique candidates")

        for i, candidate in enumerate(candidates):
            total_candidates += 1
            profile, hit_limit = profile_one_candidate(candidate)
            if hit_limit:
                node_limit_hits += 1
                print(f"[n={n_values}] candidate {i}: DagLimitExceeded, skipping")
                continue

            row = flatten_problem_profile(
                n_values, candidate.problem, candidate.canonical_hash, profile
            )
            rows.append(row)
            if profile.min_solution_depth is not None:
                depths_by_stratum[n_values].add(profile.min_solution_depth)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"[n={n_values}] {i + 1}/{len(candidates)} done ({elapsed:.0f}s elapsed)")

    write_json_rows(rows, output_dir / "problems.json")
    write_csv_rows(rows, output_dir / "problems.csv")

    # Predeclared diagnostic summary (docs/project_plan.md Section 31 /
    # the Phase 1 plan Part 4 Step 7c). This script only REPORTS these
    # numbers -- interpretation and the proceed/revise/stop decision are
    # written up separately in research/findings_phase1.md.
    ruining_move_rates = [
        r["ruining_move_state_rate"]
        for r in rows
        if r["ruining_move_state_rate"] is not None
    ]
    ruining_action_rates = [
        r["ruining_action_rate"] for r in rows if r["ruining_action_rate"] is not None
    ]

    summary = {
        "total_candidates": total_candidates,
        "node_limit_hits": node_limit_hits,
        "node_limit_hit_rate": node_limit_hits / total_candidates if total_candidates else None,
        "solvable_count": sum(1 for r in rows if r["solvable"]),
        "initially_solved_count": sum(1 for r in rows if r["initially_solved"]),
        "mean_ruining_move_state_rate": (
            sum(ruining_move_rates) / len(ruining_move_rates) if ruining_move_rates else None
        ),
        "mean_ruining_action_rate": (
            sum(ruining_action_rates) / len(ruining_action_rates)
            if ruining_action_rates
            else None
        ),
        "distinct_min_solution_depths_by_stratum": {
            str(n): sorted(depths_by_stratum[n]) for n in STRATA
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s. Wrote {len(rows)} rows to {output_dir}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
