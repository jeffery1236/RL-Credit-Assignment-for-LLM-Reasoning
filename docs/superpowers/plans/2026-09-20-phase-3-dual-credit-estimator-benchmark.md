# Phase 3 Dual Credit-Estimator Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the completed on-policy prevalence benchmark and add a separate stratified capability benchmark over consequential state-action choices.

**Architecture:** Keep Phase 3 experiment lifecycle in `experiments/benchmark_credit_estimator_fidelity.py`, while moving selection, within-state ranking, and problem-clustered bootstrap mathematics into `analysis/credit_estimator_fidelity.py`. Store complete menu-averaged policy tables per problem so Benchmark B can be regenerated without model inference.

**Tech Stack:** Python 3.12, existing Countdown DAG/oracle, Qwen3.5-4B allowed-action scorer, pytest, canonical JSON/Markdown, dependency-free SVG.

## Global Constraints

- The authoritative design is `docs/superpowers/specs/2026-09-13-phase-3-stratified-credit-estimator-benchmark-design.md`.
- Benchmark A artifacts and `REWORK` decision remain unchanged.
- Benchmark B selection uses exact constrained-policy facts only, never sampled estimator outcomes.
- User-facing terminology is “Sampled Counterfactual Advantage”; preserve internal Phase 2 `mc_reset_k*` keys.
- Keep `experiments/evaluate_policy_bridge.py` unchanged.
- No generic `utils` module.
- Paid inference requires a passing `$50` preflight and persistent per-problem checkpoints.

---

### Task 1: Stratified selection and within-state fidelity

**Files:**
- Create: `analysis/credit_estimator_fidelity.py`
- Create: `tests/test_credit_estimator_fidelity.py`

**Interfaces:**
- Produces: `StateActionCredit`, `SelectedCreditState`, `select_widest_credit_states(...)`, `within_state_rank_fidelity(...)`, and `bootstrap_problem_clustered_fidelity(...)`.
- Consumes: opaque canonical problem/state/action identifiers plus exact advantages, occupancies, depths, and rollout pools.

- [ ] **Step 1: Write failing deterministic-selection tests**

```python
def test_selects_widest_spread_state_then_occupancy_then_canonical_key():
    selected = select_widest_credit_states(candidate_states, depths=(0, 1, 2))
    assert selected[0].state_key == "widest"
    assert selected[1].state_key == "higher-occupancy"
    assert selected[2].state_key == "canonical-first"


def test_unavailable_depth_is_recorded_without_fallback():
    selected = select_widest_credit_states(depth_zero_only, depths=(0, 1, 2))
    assert [row.depth for row in selected] == [0]
```

- [ ] **Step 2: Run selection tests and verify missing-interface failures**

Run: `uv run pytest tests/test_credit_estimator_fidelity.py -q`

Expected: import failure for `analysis.credit_estimator_fidelity`.

- [ ] **Step 3: Implement exact-depth selection**

```python
def select_widest_credit_states(candidates, *, depths=(0, 1, 2)):
    selected = []
    for depth in depths:
        eligible = [
            candidate
            for candidate in candidates
            if candidate.depth == depth and len(candidate.advantage_by_action) >= 3
        ]
        if eligible:
            selected.append(min(
                eligible,
                key=lambda candidate: (
                    -candidate.advantage_spread,
                    -candidate.occupancy,
                    candidate.canonical_state_key,
                ),
            ))
    return tuple(selected)
```

- [ ] **Step 4: Write and implement within-state ranking tests**

```python
def test_within_state_rank_fidelity_removes_cross_problem_baselines():
    fidelity = within_state_rank_fidelity(grouped_credit_rows, "counterfactual_k16")
    assert fidelity == pytest.approx(1.0)
```

Convert each state's exact and estimated values independently to average percentile ranks, pool those ranks, then call the existing Spearman implementation.

- [ ] **Step 5: Write and implement clustered-bootstrap tests**

```python
def test_problem_clustered_bootstrap_is_reproducible_and_keeps_state_groups():
    first = bootstrap_problem_clustered_fidelity(problems, replicates=25, seed=7)
    second = bootstrap_problem_clustered_fidelity(problems, replicates=25, seed=7)
    assert first == second
    assert first.problem_draw_count == len(problems)
```

Each replicate resamples problems, includes every selected state/action from each sampled problem, resamples each rollout pool, and computes all estimator fidelities on the same draw.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run pytest tests/test_credit_estimator_fidelity.py -q`

Expected: all tests pass.

Commit:

```bash
git add analysis/credit_estimator_fidelity.py tests/test_credit_estimator_fidelity.py
git commit -m "feat: add stratified credit fidelity analysis"
```

---

### Task 2: Complete policy checkpoints and Benchmark B orchestration

**Files:**
- Modify: `experiments/benchmark_credit_estimator_fidelity.py`
- Modify: `tests/test_benchmark_credit_estimator_fidelity.py`

**Interfaces:**
- Consumes: Task 1 selection and fidelity interfaces; existing scorer and `FloatPolicyEvaluator`.
- Produces: `stratified/problems/<canonical_hash>.json` complete policy checkpoints and `run_stratified_benchmark(...)`.

- [ ] **Step 1: Write failing policy-checkpoint round-trip tests**

```python
def test_stratified_checkpoint_persists_complete_menu_averaged_policy(tmp_path):
    run_stratified_benchmark(fake_args, scorer=fake_scorer)
    checkpoint = json.loads(problem_checkpoint.read_text())
    assert len(checkpoint["menu_averaged_policy"]) == expected_nonterminal_states
    assert checkpoint["selection"]["depths"] == [0, 1, 2]
```

Also assert metadata drift fails and a cache-only rerun never invokes the scorer.

- [ ] **Step 2: Verify checkpoint tests fail**

Run: `uv run pytest tests/test_benchmark_credit_estimator_fidelity.py -q`

Expected: failure because the current checkpoint stores sampled pairs but not the complete policy.

- [ ] **Step 3: Persist and reload complete policy tables**

Serialize every nonterminal state and semantic action exactly once with its averaged probability. Validate model/tokenizer revisions, prompt version, rules, temperature, menu orderings, root, and canonical hash before reuse.

- [ ] **Step 4: Build exact candidates and selections from cached policy**

For each problem:

```python
evaluator = FloatPolicyEvaluator(cached_policy, reasoning_dag)
occupancy = compute_policy_visit_probabilities(reasoning_dag, cached_policy)
selected_states = select_widest_credit_states(
    build_state_candidates(reasoning_dag, evaluator, occupancy),
)
```

Evaluate every legal action at each selected state; seed parent/child pools by canonical problem, depth, state, action, and pool identity.

- [ ] **Step 5: Keep Benchmark A immutable**

Add a regression test that hashes the completed Benchmark A aggregate before and after Benchmark B generation and asserts equality. Benchmark B writes only beneath `stratified/`.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run pytest tests/test_benchmark_credit_estimator_fidelity.py -q`

Expected: all tests pass.

Commit:

```bash
git add experiments/benchmark_credit_estimator_fidelity.py tests/test_benchmark_credit_estimator_fidelity.py
git commit -m "feat: add stratified capability benchmark"
```

---

### Task 3: Dual reporting and bounded RunPod command

**Files:**
- Modify: `experiments/benchmark_credit_estimator_fidelity.py`
- Modify: `scripts/runpod_phase3_credit_estimator_pilot.sh`
- Modify: `tests/test_benchmark_credit_estimator_fidelity.py`
- Modify: `tests/test_runpod_scripts.py`
- Modify: `README.md`
- Create: `docs/phase3_credit_estimator_findings.md`

**Interfaces:**
- Produces: Benchmark B JSON/Markdown/SVG artifacts and one joint findings document linking Benchmark A and B.

- [ ] **Step 1: Write failing gate and artifact-isolation tests**

```python
def test_stratified_gate_uses_capability_labels():
    assert evaluate_stratified_gate(passing_bootstrap).decision == "CAPABLE"
    assert evaluate_stratified_gate(failing_bootstrap).decision == "NOT-DEMONSTRATED"


def test_runpod_command_targets_stratified_directory():
    assert "--benchmark stratified" in decoded_runpod_command
    assert "/workspace/results/stratified" in decoded_runpod_command
```

- [ ] **Step 2: Implement canonical aggregate reporting**

Write selection coverage, unique pair count, advantage distribution, fidelity by estimator/depth, defined-state coverage, clustered bootstrap intervals/support, and the Benchmark B decision. Regenerate four SVG plots from aggregate values.

- [ ] **Step 3: Update bounded RunPod lifecycle**

Retain the immutable image digest, persistent volume, safe scoring batch size `1`, six-hour cutoff, and `$50` refusal. Write Benchmark B to `stratified/` without modifying Benchmark A files.

- [ ] **Step 4: Write joint findings**

Record Benchmark A's existing `REWORK` result verbatim. Add Benchmark B results only after the run. Apply the four-case interpretation table from the design.

- [ ] **Step 5: Run focused/full verification and commit**

Run:

```bash
uv run pytest tests/test_credit_estimator_fidelity.py \
  tests/test_benchmark_credit_estimator_fidelity.py \
  tests/test_runpod_scripts.py -q
uv run pytest -q
git diff --check
```

Expected: all tests pass and diff check is empty.

Commit:

```bash
git add README.md docs/phase3_credit_estimator_findings.md \
  experiments/benchmark_credit_estimator_fidelity.py \
  scripts/runpod_phase3_credit_estimator_pilot.sh \
  tests/test_benchmark_credit_estimator_fidelity.py tests/test_runpod_scripts.py
git commit -m "docs: report dual Phase 3 benchmarks"
```

---

### Task 4: Run Benchmark B and finalize conclusions

**Files:**
- Create: `artifacts/phase3/credit-estimator-pilot/qwen3.5-4b/stratified/**`
- Modify: `docs/phase3_credit_estimator_findings.md`

**Interfaces:**
- Consumes: Tasks 1–3 and current RunPod pricing.
- Produces: completed Benchmark B artifacts and final dual-benchmark interpretation.

- [ ] **Step 1: Run model-free dry run**

Run the stratified CLI with `--dry-run`; verify 50 canonical problems, three requested depths, and a separate output path.

- [ ] **Step 2: Run fake-scorer smoke**

Run focused integration tests and regenerate reports from saved fake checkpoints without loading a model.

- [ ] **Step 3: Query current pricing and run non-launching preflight**

Use the current host price and the six-hour maximum. Refuse the run if projected compute exceeds `$50`.

- [ ] **Step 4: Build a pinned image and run Benchmark B**

Attach network volume `m1xhmssb78` in `EU-RO-1`. Monitor through artifact download and automatic Pod termination. Diagnose failures before retrying.

- [ ] **Step 5: Validate artifacts and findings**

Check 50 policy checkpoints, selection coverage, unique pairs, gate evidence, all four plots, and joint findings. Run the full suite and lints.

- [ ] **Step 6: Commit final artifacts**

```bash
git add artifacts/phase3 docs/phase3_credit_estimator_findings.md
git commit -m "research: complete Phase 3 dual benchmark"
```
