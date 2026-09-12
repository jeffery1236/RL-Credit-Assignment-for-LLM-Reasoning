"""Deterministic, index-seeded generation of Countdown problem candidates.

Two properties matter more than anything else here:

  1. Determinism via index-derived LOCAL RNGs, not one global RNG advanced
     sequentially. A global RNG makes candidate i depend on how many
     candidates came before it, so you can't regenerate a single item or
     change a count without silently changing every other item. Each
     candidate gets its own RNG seeded from a stable hash of
     (generator_version, seed_namespace, index).

  2. Numbers and target are sampled INDEPENDENTLY, never by walking a
     solution path backward from the target. Constructive targets bias the
     difficulty distribution toward easy problems in a way that's hard to
     see and hard to undo later.

This module only GENERATES and LABELS candidates -- it does not filter or
freeze a dataset. Exploratory manifests (Step 7) keep unsolvable and
initially-solved candidates around so their rates can be measured. The
formal train/validation/test eligibility predicate lives here
(is_eligible_for_formal_split) but is applied separately, in
experiments/freeze_dataset.py (Step 8), only after that exploratory review.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from env.countdown import DEFAULT_RULES, Problem, RuleSet, is_success
from oracle.evaluation import build_reasoning_dag, compute_v_star


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    n_values: int
    value_pool: tuple[int, ...]
    target_range: tuple[int, int]  # inclusive (low, high)
    generator_version: str
    seed_namespace: str  # e.g. "exploratory" / "train" / "validation" / "test"


@dataclass(frozen=True, slots=True)
class GeneratedCandidate:
    """One generated candidate plus the oracle-derived labels needed to
    measure exploratory rates or decide formal-split eligibility, without
    requiring the caller to rebuild the DAG themselves.
    """

    problem: Problem
    solvable: bool
    initially_solved: bool
    canonical_hash: str


def _rng_for_index(config: GenerationConfig, index: int) -> random.Random:
    """A fresh, independent RNG for this index. Uses hashlib.blake2b, NOT
    Python's built-in hash() -- the built-in is randomized per
    process and would make "the same index" produce a different problem
    every run.
    """
    key = f"{config.generator_version}|{config.seed_namespace}|{index}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def canonical_problem_hash(problem: Problem) -> str:
    """A stable identity for a problem's (values, target) pair, used to
    deduplicate within a generation run and across dataset splits. Problem
    already canonicalizes `values` as a sorted tuple in __post_init__, so
    two problems that are permutations of the same bag hash identically.
    """
    key = f"{problem.target}|{','.join(str(v) for v in problem.values)}"
    return hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()


def generate_candidate(
    config: GenerationConfig, index: int, rules: RuleSet = DEFAULT_RULES
) -> GeneratedCandidate:
    """Generate and label the candidate at `index`. Pure function of
    (config, index, rules) -- calling this twice with the same arguments,
    in any order relative to other indices, always produces the same
    candidate.
    """
    rng = _rng_for_index(config, index)
    values = tuple(
        sorted(rng.choice(config.value_pool) for _ in range(config.n_values))
    )
    target = rng.randint(config.target_range[0], config.target_range[1])
    problem = Problem(values=values, target=target)

    root = problem.to_state()
    dag = build_reasoning_dag(root, rules)
    v_star = compute_v_star(dag, rules)

    return GeneratedCandidate(
        problem=problem,
        solvable=bool(v_star[root]),
        initially_solved=is_success(root),
        canonical_hash=canonical_problem_hash(problem),
    )


def generate_candidates(
    config: GenerationConfig, count: int, rules: RuleSet = DEFAULT_RULES
) -> tuple[GeneratedCandidate, ...]:
    """Generate `count` candidates, deduplicated by canonical_hash within
    this call, by scanning indices 0, 1, 2, ... and skipping duplicates.
    Does NOT deduplicate against any other generate_candidates() call or
    seed namespace -- cross-split deduplication is the caller's job (see
    experiments/freeze_dataset.py, Step 8).
    """
    seen_hashes: set[str] = set()
    candidates: list[GeneratedCandidate] = []
    index = 0
    while len(candidates) < count:
        candidate = generate_candidate(config, index, rules)
        index += 1
        if candidate.canonical_hash in seen_hashes:
            continue
        seen_hashes.add(candidate.canonical_hash)
        candidates.append(candidate)
    return tuple(candidates)


def is_eligible_for_formal_split(problem: Problem, solvable: bool) -> bool:
    """The frozen train/validation/test eligibility predicate: the target
    must not already be sitting in the initial bag, and the problem must
    be solvable under optimal play. Kept standalone and independently
    tested so it can be re-applied to already-generated candidates without
    redoing any DAG work.
    """
    return problem.target not in problem.values and solvable
