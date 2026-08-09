"""Render the complete reasoning DAG for a single Countdown problem to JSON
and DOT (and optionally SVG via the native `dot` binary).

Usage:
    uv run python -m experiments.render_dag --numbers 3 7 10 25 --target 245 \\
        --output artifacts/phase1/example --max-nodes 100000
"""

import argparse


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
    raise NotImplementedError(
        "render_dag is implemented in Step 6 (artifacts + DAG rendering)"
    )


if __name__ == "__main__":
    main()
