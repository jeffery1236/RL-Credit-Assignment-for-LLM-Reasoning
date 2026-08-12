"""Produce canonical, hash-deduplicated, eligibility-checked train /
validation / test splits under data/v1/.

Only run this AFTER the Step 7 exploratory review (research/findings_phase1.md)
has been written and a decision to proceed has been made -- see
docs/project_plan.md's go/no-go gate. Do not tune the generation config here
to chase a preferred result; any tuning must be justified in
research/findings_phase1.md first.

Each split is generated from the SAME underlying distribution (same value
pool, same target range, same 4/5/6-number horizon mix), varying only the
seed namespace -- this is what makes the splits "matched" (comparable
profile distributions) without needing an explicit balancing step. A
post-hoc audit re-profiles a random subsample of each split and reports
mean branching / mean min_solution_depth in the manifest as evidence.

Every frozen row satisfies is_eligible_for_formal_split: the target is not
already in the initial bag, and the problem is solvable under optimal play.
Rejected candidates (initially solved, unsolvable, or duplicate) are
counted, not silently discarded -- their rates are recorded in the
manifest.

Usage:
    uv run python -m experiments.freeze_dataset --output data/v1
"""

import argparse
import json
import random
from pathlib import Path

from env.countdown import DEFAULT_RULES
from env.generator import GenerationConfig, generate_candidate, is_eligible_for_formal_split
from oracle.evaluation import (
    build_reasoning_dag,
    compute_min_steps_to_success,
    compute_q_star,
    compute_v_star,
)

GENERATOR_VERSION = "v1.0-frozen"
VALUE_POOL = tuple(range(1, 11)) + (25, 50, 75, 100)
TARGET_RANGE = (100, 999)
STRATA = (4, 5, 6)
AUDIT_SAMPLE_SIZE = 50


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory, e.g. data/v1"
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=1000)
    return parser


def _stratum_targets(total: int) -> list[int]:
    """Split `total` as evenly as possible across len(STRATA) strata,
    handing the remainder to the first strata so counts never drift by
    more than 1.
    """
    base, remainder = divmod(total, len(STRATA))
    return [base + (1 if i < remainder else 0) for i in range(len(STRATA))]


def collect_split(
    split_name: str, size: int, global_seen_hashes: set[str]
) -> tuple[list[dict], dict]:
    """Generate `size` eligible, deduplicated candidates for one split,
    split evenly across the 4/5/6-number strata. Mutates
    global_seen_hashes in place so a later split can detect (and reject)
    any accidental collision with an earlier one.
    """
    accepted: list[dict] = []
    rejections = {
        "initially_solved": 0,
        "unsolvable": 0,
        "duplicate_within_split": 0,
        "duplicate_cross_split": 0,
    }

    for n_values, target_for_stratum in zip(STRATA, _stratum_targets(size)):
        config = GenerationConfig(
            n_values=n_values,
            value_pool=VALUE_POOL,
            target_range=TARGET_RANGE,
            generator_version=GENERATOR_VERSION,
            seed_namespace=f"{split_name}-{n_values}",
        )
        seen_within_stratum: set[str] = set()
        index = 0
        found = 0
        while found < target_for_stratum:
            candidate = generate_candidate(config, index)
            index += 1

            if candidate.canonical_hash in seen_within_stratum:
                rejections["duplicate_within_split"] += 1
                continue
            seen_within_stratum.add(candidate.canonical_hash)

            if candidate.canonical_hash in global_seen_hashes:
                rejections["duplicate_cross_split"] += 1
                continue

            if candidate.initially_solved:
                rejections["initially_solved"] += 1
                continue
            if not candidate.solvable:
                rejections["unsolvable"] += 1
                continue

            assert is_eligible_for_formal_split(candidate.problem, candidate.solvable)
            global_seen_hashes.add(candidate.canonical_hash)
            accepted.append(
                {
                    "n_values": n_values,
                    "values": list(candidate.problem.values),
                    "target": candidate.problem.target,
                    "canonical_hash": candidate.canonical_hash,
                }
            )
            found += 1

    return accepted, rejections


def audit_split(rows: list[dict], rules=DEFAULT_RULES, sample_size: int = AUDIT_SAMPLE_SIZE) -> dict:
    """Re-profile a random subsample of an already-frozen split to report
    mean branching and mean min_solution_depth -- empirical evidence that
    splits drawn from the same generation config really are comparable,
    without re-running full policy evaluation on all 6500+ rows.
    """
    from env.countdown import Problem, allowed_actions, is_terminal

    rng = random.Random(0)
    sample = rng.sample(rows, min(sample_size, len(rows)))

    depths = []
    branching_means = []
    for row in sample:
        problem = Problem(values=tuple(row["values"]), target=row["target"])
        root = problem.to_state()
        dag = build_reasoning_dag(root, rules)
        v_star = compute_v_star(dag, rules)
        q_star = compute_q_star(dag, v_star)
        min_steps = compute_min_steps_to_success(dag, v_star, q_star)
        depths.append(min_steps[root])

        non_terminal_states = [s for s in dag.nodes if not is_terminal(s, rules)]
        if non_terminal_states:
            counts = [len(allowed_actions(s, rules)) for s in non_terminal_states]
            branching_means.append(sum(counts) / len(counts))

    return {
        "audit_sample_size": len(sample),
        "audit_mean_min_solution_depth": (
            sum(depths) / len(depths) if depths else None
        ),
        "audit_mean_branching": (
            sum(branching_means) / len(branching_means) if branching_means else None
        ),
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": args.train_size,
        "validation": args.validation_size,
        "test": args.test_size,
    }

    global_seen_hashes: set[str] = set()
    manifest_splits: dict[str, dict] = {}

    for split_name, size in splits.items():
        print(f"[{split_name}] collecting {size} eligible candidates...")
        rows, rejections = collect_split(split_name, size, global_seen_hashes)
        write_jsonl(rows, output_dir / f"{split_name}.jsonl")

        by_stratum = {n: sum(1 for r in rows if r["n_values"] == n) for n in STRATA}
        audit = audit_split(rows)

        manifest_splits[split_name] = {
            "requested_size": size,
            "actual_size": len(rows),
            "by_stratum": {str(n): count for n, count in by_stratum.items()},
            "rejections": rejections,
            **audit,
        }
        print(
            f"[{split_name}] wrote {len(rows)} rows "
            f"(rejected: {rejections})"
        )

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "value_pool": list(VALUE_POOL),
        "target_range": list(TARGET_RANGE),
        "strata": list(STRATA),
        "eligibility_predicate": "target not in initial values AND solvable (V*(s0) == 1)",
        "splits": manifest_splits,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )

    total_hashes = len(global_seen_hashes)
    total_rows = sum(s["actual_size"] for s in manifest_splits.values())
    assert total_hashes == total_rows, (
        f"cross-split hash count ({total_hashes}) != total rows ({total_rows}) "
        "-- a duplicate slipped through deduplication"
    )
    print(f"\nDone. {total_rows} total problems across all splits, all cross-split-unique.")


if __name__ == "__main__":
    main()
