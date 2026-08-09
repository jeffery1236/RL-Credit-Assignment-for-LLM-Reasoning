"""Generate the 750-problem exploratory manifest (250 each of 4/5/6-number
problems) and report structural + policy credit-assignment diagnostics.

Usage:
    uv run python -m experiments.profile_environment --output artifacts/phase1/exploratory
"""

import argparse


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


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    raise NotImplementedError(
        "profile_environment is implemented in Step 7 (exploratory study)"
    )


if __name__ == "__main__":
    main()
