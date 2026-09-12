"""Compare complete-DAG policies produced with different menu orderings."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

from env.countdown import Operator, RuleSet, State, allowed_actions, is_terminal
from experiments.evaluate_allowed_action_llm_policy import (
    validate_numerical_policy_invariants,
)
from oracle.evaluation import (
    PolicyValidationError,
    build_reasoning_dag,
    compute_q_star,
    compute_v_star,
)
from oracle.float_policy_evaluation import FloatPolicyEvaluator
from policies.allowed_action_llm_policy import (
    CachedAllowedActionLLMPolicy,
    load_cache_snapshot,
    serialize_action,
    serialize_state,
    validate_cache_coverage,
)

JSON_OUTPUT_NAME = "menu_ordering_sensitivity.json"
MARKDOWN_OUTPUT_NAME = "menu_ordering_sensitivity.md"


def compare_menu_ordering_caches(
    named_cache_paths: Mapping[str, Path],
) -> dict[str, object]:
    if len(named_cache_paths) < 2:
        raise ValueError("Menu-ordering comparison requires at least two caches")
    if any(not name for name in named_cache_paths):
        raise ValueError("Every cache must have a non-empty name")

    snapshots = {
        name: load_cache_snapshot(path)
        for name, path in sorted(named_cache_paths.items())
    }
    first_name = next(iter(snapshots))
    first_snapshot = snapshots[first_name]
    comparable_metadata = replace(first_snapshot.metadata, menu_ordering_id="")
    for name, snapshot in snapshots.items():
        if (
            replace(snapshot.metadata, menu_ordering_id="")
            != comparable_metadata
        ):
            raise PolicyValidationError(
                f"Cache {name!r} differs beyond menu_ordering_id"
            )

    try:
        root = State(
            values=tuple(first_snapshot.metadata.root_problem["values"]),
            target=int(first_snapshot.metadata.root_problem["target"]),
        )
        rules = RuleSet(
            operators=tuple(
                Operator(symbol)
                for symbol in first_snapshot.metadata.rules["operators"]
            ),
            min_intermediate=int(
                first_snapshot.metadata.rules["min_intermediate"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PolicyValidationError(
            f"Malformed shared cache metadata: {error}"
        ) from error

    dag = build_reasoning_dag(root, rules)
    nonterminal_states = {
        state for state in dag.nodes if not is_terminal(state, rules)
    }
    for snapshot in snapshots.values():
        validate_cache_coverage(snapshot, nonterminal_states)

    v_star = compute_v_star(dag, rules)
    q_star = compute_q_star(dag, v_star)
    winning_actions = [
        action for action, value in q_star[root].items() if value == 1
    ]
    if len(winning_actions) != 1:
        raise PolicyValidationError(
            "Sensitivity comparison requires one unique winning root action; "
            f"found {len(winning_actions)}"
        )
    winning_action = winning_actions[0]
    root_actions = allowed_actions(root, rules)

    ordering_summaries: dict[str, object] = {}
    root_vectors: dict[str, tuple[float, ...]] = {}
    for name, snapshot in snapshots.items():
        policy = CachedAllowedActionLLMPolicy(snapshot.records)
        evaluator = FloatPolicyEvaluator(policy, dag, rules)
        validate_numerical_policy_invariants(dag, evaluator, rules)
        root_probabilities = evaluator.action_probs(root)
        root_vectors[name] = tuple(
            root_probabilities[action] for action in root_actions
        )
        root_record = snapshot.records[root]
        root_labels = {
            option.action: option.label for option in root_record.menu
        }
        top_action = max(
            root_actions,
            key=lambda action: root_probabilities[action],
        )

        label_totals: dict[str, float] = {}
        label_counts: dict[str, int] = {}
        for record in snapshot.records.values():
            for option in record.menu:
                label_totals[option.label] = (
                    label_totals.get(option.label, 0.0)
                    + record.probabilities[option.action]
                )
                label_counts[option.label] = label_counts.get(option.label, 0) + 1
        label_summary = {
            label: {
                "appearance_count": label_counts[label],
                "mean_probability": label_totals[label] / label_counts[label],
            }
            for label in sorted(label_totals)
        }

        entropy = -math.fsum(
            probability * math.log(probability)
            for probability in root_probabilities.values()
            if probability > 0.0
        )
        ordering_summaries[name] = {
            "menu_ordering_id": snapshot.metadata.menu_ordering_id,
            "root_v_pi": evaluator.v_pi(root),
            "root_entropy_nats": entropy,
            "winning_root_action_label": root_labels[winning_action],
            "winning_root_action_probability": root_probabilities[
                winning_action
            ],
            "top_root_action": serialize_action(top_action),
            "top_root_action_label": root_labels[top_action],
            "top_root_action_probability": root_probabilities[top_action],
            "root_action_probabilities": [
                {
                    "action": serialize_action(action),
                    "label": root_labels[action],
                    "probability": root_probabilities[action],
                }
                for action in root_actions
            ],
            "average_probability_by_label": label_summary,
        }

    pairwise_total_variation = []
    for first, second in combinations(sorted(root_vectors), 2):
        distance = 0.5 * math.fsum(
            abs(first_probability - second_probability)
            for first_probability, second_probability in zip(
                root_vectors[first],
                root_vectors[second],
                strict=True,
            )
        )
        pairwise_total_variation.append(
            {"first": first, "second": second, "distance": distance}
        )

    return {
        "artifact_version": "1.0",
        "root_problem": serialize_state(root),
        "nonterminal_state_count": len(nonterminal_states),
        "winning_root_action": serialize_action(winning_action),
        "menu_orderings": ordering_summaries,
        "pairwise_total_variation": pairwise_total_variation,
    }


def comparison_to_markdown(comparison: Mapping[str, object]) -> str:
    root = comparison["root_problem"]
    winning = comparison["winning_root_action"]
    menu_orderings = comparison["menu_orderings"]
    pairwise = comparison["pairwise_total_variation"]
    lines = [
        "# Menu-Order Sensitivity",
        "",
        (
            f"Root: numbers `{root['values']}`, target `{root['target']}`. "
            f"Compared `{len(menu_orderings)}` complete-DAG policies over "
            f"`{comparison['nonterminal_state_count']}` nonterminal states."
        ),
        "",
        (
            "Winning root action: "
            f"`{winning['left']} {winning['operator']} {winning['right']} "
            f"= {winning['result']}`."
        ),
        "",
        "## Root policy by menu ordering",
        "",
        "| Menu ordering | Vπ(root) | P(winning action) | Top root action | Label | P(top) |",
        "|---|---:|---:|---|---|---:|",
    ]
    for name, summary in menu_orderings.items():
        top = summary["top_root_action"]
        top_text = (
            f"{top['left']} {top['operator']} {top['right']} = {top['result']}"
        )
        lines.append(
            f"| `{name}` | {summary['root_v_pi']:.8g} | "
            f"{summary['winning_root_action_probability']:.8f} | "
            f"`{top_text}` | `{summary['top_root_action_label']}` | "
            f"{summary['top_root_action_probability']:.8f} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise root total-variation distance",
            "",
            "| First ordering | Second ordering | TV distance |",
            "|---|---|---:|",
        ]
    )
    for row in pairwise:
        lines.append(
            f"| `{row['first']}` | `{row['second']}` | "
            f"{row['distance']:.8f} |"
        )

    lines.extend(["", "## Average probability by option label", ""])
    for name, summary in menu_orderings.items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("| Label | Appearances | Mean probability |")
        lines.append("|---|---:|---:|")
        for label, label_summary in summary[
            "average_probability_by_label"
        ].items():
            lines.append(
                f"| `{label}` | {label_summary['appearance_count']} | "
                f"{label_summary['mean_probability']:.8f} |"
            )
        lines.append("")

    lines.extend(
        [
            "Root distributions are matched by arithmetic `Action`, not by "
            "option label. Large variation is reported as a result rather than "
            "treated as a failed run.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named policy cache; repeat once per menu ordering.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    named_paths: dict[str, Path] = {}
    for value in args.cache:
        if "=" not in value:
            parser.error(f"--cache must use NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            parser.error(f"--cache must use non-empty NAME=PATH, got {value!r}")
        if name in named_paths:
            parser.error(f"duplicate cache name {name!r}")
        named_paths[name] = Path(raw_path)

    comparison = compare_menu_ordering_caches(named_paths)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_OUTPUT_NAME
    markdown_path = output_dir / MARKDOWN_OUTPUT_NAME
    json_path.write_text(
        json.dumps(
            comparison,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    )
    markdown_path.write_text(comparison_to_markdown(comparison))
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
