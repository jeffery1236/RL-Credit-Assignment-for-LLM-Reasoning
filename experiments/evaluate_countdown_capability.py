"""Literature-style free-form Countdown capability evaluation.

This is separate from the state-only exact-credit policy. It asks a model to
produce a complete expression using every supplied number exactly once, matching
the setup used by TinyZero and related Countdown work.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

DEFAULT_DATASET = Path("data/countdown-capability/tinyzero-test-12.jsonl")
RESULTS_FILE = "countdown_capability_results.json"


@dataclass(frozen=True, slots=True)
class CapabilityProblem:
    numbers: tuple[int, ...]
    target: int
    source_row_index: int


@dataclass(frozen=True, slots=True)
class ResponseScore:
    format_valid: bool
    equation_valid: bool
    correct: bool
    equation: str | None


def load_capability_problems(path: Path) -> tuple[CapabilityProblem, ...]:
    problems = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            problem = CapabilityProblem(
                numbers=tuple(int(number) for number in row["numbers"]),
                target=int(row["target"]),
                source_row_index=int(row["source_row_index"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Malformed capability problem on line {line_number}: {error}"
            ) from error
        problems.append(problem)
    if not problems:
        raise ValueError(f"No capability problems found in {path}")
    return tuple(problems)


def build_capability_messages(
    problem: CapabilityProblem,
) -> tuple[dict[str, str], ...]:
    return (
        {
            "role": "system",
            "content": (
                "You are a careful mathematical reasoner. Solve the Countdown "
                "problem and follow the requested output format."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Using the numbers {list(problem.numbers)}, create an equation "
                f"that equals {problem.target}. You may use +, -, *, / and "
                "parentheses. Use each number exactly once. Show your reasoning "
                "inside <think>...</think> tags, then put only the final equation "
                "expression inside <answer>...</answer> tags. The answer tag "
                "must contain only the arithmetic expression; do not include an "
                "equals sign or the target value."
            ),
        },
    )


def _evaluate_expression(node: ast.AST, used_numbers: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body, used_numbers)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("Only integer input numbers are allowed")
        used_numbers.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_expression(node.operand, used_numbers)
        return value if isinstance(node.op, ast.UAdd) else -value
    if not isinstance(node, ast.BinOp):
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    left = _evaluate_expression(node.left, used_numbers)
    right = _evaluate_expression(node.right, used_numbers)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        if right == 0:
            raise ValueError("Division by zero")
        return left / right
    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")


def score_response(
    response: str,
    problem: CapabilityProblem,
) -> ResponseScore:
    matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not matches:
        return ResponseScore(False, False, False, None)
    equation = matches[-1].strip()
    if not equation:
        return ResponseScore(True, False, False, equation)

    try:
        parsed = ast.parse(equation, mode="eval")
        if sum(1 for _ in ast.walk(parsed)) > 100:
            raise ValueError("Expression is too large")
        used_numbers: list[int] = []
        value = _evaluate_expression(parsed, used_numbers)
        if sorted(used_numbers) != sorted(problem.numbers):
            raise ValueError(
                f"Used numbers {used_numbers!r}; expected {problem.numbers!r}"
            )
    except (SyntaxError, ValueError, ZeroDivisionError):
        return ResponseScore(True, False, False, equation)
    return ResponseScore(
        format_valid=True,
        equation_valid=True,
        correct=value == problem.target,
        equation=equation,
    )


def summarize_problem_results(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not results:
        raise ValueError("At least one problem result is required")
    all_scores = [
        score
        for result in results
        for score in result["scores"]
    ]
    if not all_scores:
        raise ValueError("At least one generated response is required")
    return {
        "problem_count": len(results),
        "sample_count": len(all_scores),
        "samples_per_problem": len(all_scores) // len(results),
        "accuracy": sum(score.correct for score in all_scores) / len(all_scores),
        "pass_at_k": (
            sum(any(score.correct for score in result["scores"]) for result in results)
            / len(results)
        ),
        "all_correct_at_k": (
            sum(all(score.correct for score in result["scores"]) for result in results)
            / len(results)
        ),
        "format_rate": (
            sum(score.format_valid for score in all_scores) / len(all_scores)
        ),
        "valid_equation_rate": (
            sum(score.equation_valid for score in all_scores) / len(all_scores)
        ),
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def _write_results(path: Path, payload: Mapping[str, object]) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(_canonical_json(payload))
    temporary_path.replace(path)


def run_capability_evaluation(args: argparse.Namespace) -> None:
    # Model imports stay local so verifier tests and ``--help`` remain lightweight.
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer

    problems = load_capability_problems(Path(args.dataset))
    if args.problem_limit is not None:
        problems = problems[: args.problem_limit]
    if not problems:
        raise ValueError("problem_limit selected no problems")

    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        device = "mps"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    resolved_revision = model_info(args.model).sha
    if not resolved_revision:
        raise ValueError(f"Could not resolve model revision for {args.model}")
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=resolved_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=resolved_revision,
        dtype=dtype,
    ).to(device)
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / RESULTS_FILE
    metadata = {
        "artifact_version": "1.0",
        "dataset": str(args.dataset),
        "model_repository": args.model,
        "model_revision": resolved_revision,
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "problem_count": len(problems),
        "samples_per_problem": args.samples_per_problem,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "generation_batch_size": args.generation_batch_size,
        "seed": args.seed,
        "native_thinking_enabled": False,
    }

    completed: dict[int, dict[str, object]] = {}
    if output_path.exists() and not args.overwrite:
        existing = json.loads(output_path.read_text())
        if existing.get("metadata") != metadata:
            raise ValueError(
                "Existing capability results use different configuration; "
                "pass --overwrite or choose another output directory"
            )
        completed = {
            int(result["source_row_index"]): result
            for result in existing.get("problems", [])
        }

    for problem in problems:
        if problem.source_row_index in completed:
            continue
        prompt = tokenizer.apply_chat_template(
            build_capability_messages(problem),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        responses: list[str] = []
        for batch_start in range(
            0,
            args.samples_per_problem,
            args.generation_batch_size,
        ):
            batch_count = min(
                args.generation_batch_size,
                args.samples_per_problem - batch_start,
            )
            torch.manual_seed(
                args.seed + problem.source_row_index * 1000 + batch_start
            )
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=batch_count,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            responses.extend(
                tokenizer.decode(
                    sequence[input_length:],
                    skip_special_tokens=True,
                )
                for sequence in generated
            )

        scores = [score_response(response, problem) for response in responses]
        completed[problem.source_row_index] = {
            "source_row_index": problem.source_row_index,
            "numbers": list(problem.numbers),
            "target": problem.target,
            "responses": [
                {"text": response, "score": asdict(score)}
                for response, score in zip(responses, scores, strict=True)
            ],
        }
        ordered_results = [
            completed[index] for index in sorted(completed)
        ]
        summary_input = [
            {
                "problem": CapabilityProblem(
                    numbers=tuple(result["numbers"]),
                    target=int(result["target"]),
                    source_row_index=int(result["source_row_index"]),
                ),
                "scores": [
                    ResponseScore(**response["score"])
                    for response in result["responses"]
                ],
            }
            for result in ordered_results
        ]
        _write_results(
            output_path,
            {
                "metadata": metadata,
                "summary": summarize_problem_results(summary_input),
                "problems": ordered_results,
            },
        )
        print(
            f"Completed problem {problem.source_row_index}: "
            f"{sum(score.correct for score in scores)}/{len(scores)} correct"
        )

    print(f"Wrote {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-problem", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--problem-limit", type=int)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.samples_per_problem < 1:
        parser.error("--samples-per-problem must be at least 1")
    if args.generation_batch_size < 1:
        parser.error("--generation-batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        parser.error("--temperature must be finite and greater than zero")
    run_capability_evaluation(args)


if __name__ == "__main__":
    main()
