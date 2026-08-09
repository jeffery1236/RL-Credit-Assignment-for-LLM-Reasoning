"""Produce canonical, hash-deduplicated, eligibility-checked train/validation/test
splits under data/v1/.

Usage:
    uv run python -m experiments.freeze_dataset --output data/v1
"""

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory, e.g. data/v1"
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    raise NotImplementedError(
        "freeze_dataset is implemented in Step 8 (tune, freeze, and validate)"
    )


if __name__ == "__main__":
    main()
