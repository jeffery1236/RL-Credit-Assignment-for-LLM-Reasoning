"""Render the complete reasoning DAG for a single Countdown problem to JSON
and DOT (and optionally SVG via the native `dot` binary).

Usage:
    uv run python -m experiments.render_dag --numbers 3 7 10 25 --target 245 \\
        --output artifacts/phase1/example --max-nodes 100000
"""

import argparse
from pathlib import Path

from analysis.artifacts import render_svg, to_dot, to_canonical_json, build_dag_artifact
from env.countdown import Problem
from policies.baselines import (
    EpsilonGreedyDistancePolicy,
    GreedyDistancePolicy,
    UniformPolicy,
)

DEFAULT_POLICIES = {
    "uniform": UniformPolicy(),
    "greedy_distance": GreedyDistancePolicy(),
    "epsilon_greedy_distance": EpsilonGreedyDistancePolicy(),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numbers", type=int, nargs="+", required=True, help="Starting numbers."
    )
    parser.add_argument("--target", type=int, required=True, help="Target value.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path stem, e.g. artifacts/phase1/example",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=100_000,
        help="Raise DagLimitExceeded past this many distinct states.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    problem = Problem(values=tuple(args.numbers), target=args.target)
    root = problem.to_state()

    artifact = build_dag_artifact(
        root,
        policies=DEFAULT_POLICIES,
        generator_version="manual",
        max_nodes=args.max_nodes,
    )

    output_stem = Path(args.output)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_stem.with_suffix(".json")
    dot_path = output_stem.with_suffix(".dot")
    svg_path = output_stem.with_suffix(".svg")

    json_path.write_text(to_canonical_json(artifact))
    dot_path.write_text(to_dot(artifact))
    rendered = render_svg(dot_path, svg_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {dot_path}")
    if rendered:
        print(f"Wrote {svg_path}")
    print(
        f"DAG: {artifact['node_count']} nodes, {artifact['edge_count']} edges "
        f"(root solvable={bool(artifact['nodes'][0]['v_star'])})"
    )


if __name__ == "__main__":
    main()
