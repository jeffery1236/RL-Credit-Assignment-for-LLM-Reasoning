# Experiment log

Format per `docs/project_plan.md` Section 24: Question / Hypothesis / Independent variable / Dependent variable / Controls / Expected result / Actual result / Interpretation / Next experiment.

---

## E001 — Manual verification of one complete reasoning DAG

**Date:** 2026-08-12

**Question:** Does the exact oracle (DAG enumeration, V*, Q*, min_steps_to_success) produce correct, internally consistent output on a real, non-trivial problem?

**Hypothesis:** Given the existing unit/property test suite already cross-checks the oracle against an independent brute-force solver on small synthetic cases, a real problem of realistic size (4 numbers, a 3-digit target) should show zero invariant violations and should be independently reconstructible by hand.

**Independent variable:** The problem `(3, 7, 10, 25) → 245` (the project plan's own worked example).

**Dependent variable:** Structural correctness of every node and edge in the resulting DAG; agreement between the oracle's claimed values and hand-recomputed arithmetic.

**Controls:** Default `RuleSet` (`+`, `−`, `×`, no division); `max_nodes=100_000`.

**Command:**

```bash
uv run python -m experiments.render_dag --numbers 3 7 10 25 --target 245 \
    --output artifacts/phase1/example --max-nodes 100000
```

**Expected result:** A DAG of a few hundred nodes; the root state solvable (`V*=1`); a short winning line reachable; no structural inconsistencies.

**Actual result:**

- **Graph size:** 236 nodes, 420 edges.
- **Root:** `V*(root) = 1`, `min_steps_to_success(root) = 2`.
- **Root fragility:** 18 legal actions from the root; only **1** (`10 + 25 = 35`) preserves the win (`Q* = 1`); the other **17** are irreversibly destructive (`Q* = 0`). Fragility ≈ 0.944 — an extremely narrow "correct doorway."
- **A 236-node DAG is too large to genuinely eyeball node-by-node**, so verification was done two ways instead of literal manual inspection of all 236 nodes:
  1. **Exhaustive automated invariant sweep** over every node and edge in this specific DAG (not the general test suite — a dedicated one-off script run directly against `artifacts/phase1/example.json`): for every node, recomputed `allowed_actions(state)` from the environment and confirmed it exactly matches the node's outgoing edge set; for every edge, recomputed the action's arithmetic result and confirmed it matches the recorded value; confirmed every node's `is_terminal` flag matches `is_terminal(state)`. **Result: 236/236 nodes and 420/420 edges checked, zero errors.**
  2. **Manual hand-verification of two specific paths:**
     - Winning path: `10 + 25 = 35` → bag `(3, 7, 35)` → `7 × 35 = 245` → bag `(3, 245)`, target present, success. Two steps — matches `min_steps_to_success = 2` exactly.
     - A destructive root action: `3 × 7 = 21` → bag `(10, 21, 25)`. Hand-enumerated all 9 possible next actions from this bag (`10+21, 10×21, 21−10, 10+25, 10×25, 25−10, 21+25, 21×25, 25−21`) — none produces 245 and none of their single-remaining-number results equal 245 either. Confirms `V*((10,21,25)) = 0`, matching the oracle's recorded `Q*(root, 3×7) = 0`.
- The full existing test suite (231 tests at the time of this experiment, including the independent brute-force cross-check in `tests/reference_solver.py`, the exact Bellman-residual and zero-sum-advantage property tests, and the 20k-rollout Monte-Carlo-vs-exact sanity check) was already green going into this experiment and remained green after.

**Interpretation:** The oracle is structurally sound on a realistic, non-trivial problem, not just on the small hand-constructed fixtures used in unit tests. The fragility observed here (17/18 root actions lose immediately) is a first hint — confirmed at scale by E002 below — that this environment's default generation settings may produce very narrow "correct paths," which is scientifically interesting (rich credit structure) but worth checking isn't *so* narrow that it degenerates into "there is exactly one right answer and everything else is identical noise."

**Next experiment:** E002 — measure whether this fragility pattern holds across a broad, unbiased sample of generated problems, and whether the three baseline policies produce visibly different credit structure.

---

## E002 — 750-problem exploratory profile (the go/no-go gate)

**Date:** 2026-08-12

**Question:** Does the fragility pattern found in E001 hold across a broad, unbiased sample, and do the three baseline policies produce visibly different, non-degenerate credit structure?

**Hypothesis:** Fragility (irreversibly-destructive action rate) will be high and fairly consistent across problems, matching E001; the three baselines will show *some* differentiation in advantage distribution, more visible once weighted by actual on-policy visitation than in the raw unweighted statistic.

**Independent variable:** Number-count stratum (`n_values ∈ {4, 5, 6}`), 250 candidates each.

**Dependent variable:** Solvability rate, fragility/decision-sensitivity rates, unweighted and occupancy-weighted `Aπ` sign/magnitude distributions per baseline policy.

**Controls:** Fixed value pool (`1–10` + `{25,50,75,100}`), fixed target range (`100–999`), fixed `RuleSet`, `max_nodes = 100,000`. Six diagnostic bands predeclared in `experiments/profile_environment.py` before this run.

**Command:**

```bash
uv run python -m experiments.profile_environment --output artifacts/phase1/exploratory --per-stratum 250
```

**Expected result:** High fragility consistent with E001; no computational-degeneracy triggers; some visible policy differentiation once occupancy-weighted.

**Actual result:** 750/750 candidates profiled in 24 minutes, 0 node-limit hits. Overall solvability 42.0% (6.0% / 37.6% / 82.4% at n=4/5/6 respectively — an unanticipated, strongly horizon-dependent pattern). Pooled decision-sensitivity among solvable non-terminal states: 99.99%. Pooled ruining-action rate: 84.1%, stable across horizons (83–85%) when conditioned on solvability. Raw unweighted nonzero-`Aπ` fraction was very low for all three baselines (0.56–1.71%) and all three looked statistically indistinguishable by the predeclared thresholds — but the occupancy-weighted view showed uniform and epsilon-greedy with 30–37% of visited probability mass on nonzero-advantage actions, and an order-of-magnitude difference in weighted mean `|Aπ|` between greedy-distance (0.0018) and epsilon-greedy (0.0170). Full results, per-trigger interpretation, and the go/no-go decision are in `research/findings_phase1.md`.

**Interpretation:** See `research/findings_phase1.md` for the full write-up. Summary: the raw unweighted statistics are dominated by dead-subtree zero-inflation and are misleading on their own; the occupancy-weighted companion view (built specifically for this purpose) reveals a rich, clearly policy-differentiated credit signal, at least for policies with non-trivial entropy. Decision: proceed to Step 8 with the current generation config; the horizon/solvability confound is documented as a design note for any future horizon-specific experiment (RQ3), not a Phase 1 blocker.

**Next experiment:** Step 8 — freeze train/validation/test splits using the same generation config, then begin Phase 2 (LLM policy integration).
