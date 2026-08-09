"""Tests for env/generator.py: deterministic, index-seeded problem
generation, canonical hashing, and the formal-split eligibility predicate.
"""

from env.countdown import Problem
from env.generator import (
    GenerationConfig,
    canonical_problem_hash,
    generate_candidate,
    generate_candidates,
    is_eligible_for_formal_split,
)

SMALL_POOL_CONFIG = GenerationConfig(
    n_values=4,
    value_pool=tuple(range(1, 11)),
    target_range=(10, 200),
    generator_version="v-test",
    seed_namespace="unit",
)


def test_generate_candidate_is_deterministic():
    a = generate_candidate(SMALL_POOL_CONFIG, 42)
    b = generate_candidate(SMALL_POOL_CONFIG, 42)
    assert a.problem == b.problem
    assert a.canonical_hash == b.canonical_hash
    assert a.solvable == b.solvable
    assert a.initially_solved == b.initially_solved


def test_generate_candidate_is_index_independent():
    isolated = generate_candidate(SMALL_POOL_CONFIG, 7)
    from_batch = [generate_candidate(SMALL_POOL_CONFIG, i) for i in range(10)][7]
    assert isolated.problem == from_batch.problem
    assert isolated.canonical_hash == from_batch.canonical_hash


def test_different_indices_usually_differ():
    # Not a strict guarantee, but with a reasonably sized pool, the first
    # 20 indices should not all collide onto one problem.
    problems = {generate_candidate(SMALL_POOL_CONFIG, i).problem for i in range(20)}
    assert len(problems) > 1


def test_different_seed_namespace_changes_the_result():
    other_config = GenerationConfig(
        n_values=SMALL_POOL_CONFIG.n_values,
        value_pool=SMALL_POOL_CONFIG.value_pool,
        target_range=SMALL_POOL_CONFIG.target_range,
        generator_version=SMALL_POOL_CONFIG.generator_version,
        seed_namespace="a-different-namespace",
    )
    a = generate_candidate(SMALL_POOL_CONFIG, 0)
    b = generate_candidate(other_config, 0)
    # Not a mathematical guarantee, but should hold for a real hash-derived
    # RNG seed -- if this ever collides, seeding is broken.
    assert a.problem != b.problem


def test_canonical_hash_is_permutation_invariant():
    p1 = Problem(values=(3, 7, 10, 25), target=245)
    p2 = Problem(values=(25, 10, 7, 3), target=245)  # Problem sorts internally
    assert canonical_problem_hash(p1) == canonical_problem_hash(p2)


def test_canonical_hash_distinguishes_different_problems():
    p1 = Problem(values=(3, 7, 10, 25), target=245)
    p2 = Problem(values=(3, 7, 10, 25), target=246)
    p3 = Problem(values=(3, 7, 10, 26), target=245)
    assert len({canonical_problem_hash(p) for p in (p1, p2, p3)}) == 3


def test_generate_candidates_deduplicates_and_reaches_requested_count():
    config = GenerationConfig(
        n_values=3,
        value_pool=(1, 2, 3, 4, 5, 6, 7, 8),
        target_range=(1, 50),
        generator_version="v-test",
        seed_namespace="dedup-check",
    )
    candidates = generate_candidates(config, 20)
    hashes = [c.canonical_hash for c in candidates]
    assert len(candidates) == 20
    assert len(set(hashes)) == 20


# ---------------------------------------------------------------------------
# Formal-split eligibility predicate
# ---------------------------------------------------------------------------


def test_eligibility_accepts_solvable_problem_with_absent_target():
    problem = Problem(values=(1, 2), target=3)
    assert is_eligible_for_formal_split(problem, solvable=True) is True


def test_eligibility_rejects_unsolvable_problem():
    problem = Problem(values=(1, 2), target=3)
    assert is_eligible_for_formal_split(problem, solvable=False) is False


def test_eligibility_rejects_initially_solved_problem_even_if_solvable():
    problem = Problem(values=(1, 3), target=3)  # target already in the bag
    assert is_eligible_for_formal_split(problem, solvable=True) is False
