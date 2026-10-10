# Phase 3 Credit-Estimator Findings

Phase 3 reports two complementary benchmarks. Neither benchmark replaces or
gates the other.

The plots use short names:

- **One rollout:** score a move from one simulated game after taking it.
- **Before/after ×K:** compare K simulated games from the starting position
  with K simulated games after taking the move.
- **Ranking accuracy:** how well those scores order the available moves
  compared with the exact answer; `1.0` is perfect.

## Benchmark A: on-policy prevalence

Artifact directory:
`artifacts/phase3/credit-estimator-pilot/qwen3.5-4b/`

Benchmark A is complete. Its result is recorded verbatim:

> Pilot gate: **REWORK**

Its gate evidence records a median fidelity improvement of `-0.023` and paired
bootstrap support of `1.6%`. Naturally sampled four-number decisions were
dominated by zero or below-resolution exact advantages, so Sampled
Counterfactual Advantage at `K <= 16` did not reliably outperform Uniform
Outcome Credit at this budget.

Benchmark A cannot be regenerated without paid inference. Its compact summary
and checksum manifest are committed. The 61 raw files are stored in the GitHub
Release named `phase3-credit-estimator-data-v1`; the release archive's URL and
SHA-256 digest are recorded in
`artifacts/phase3/credit-estimator-pilot/raw_artifacts_manifest.json`.

## Benchmark B: stratified capability

Artifact directory:
`artifacts/phase3/credit-estimator-pilot/qwen3.5-4b/stratified/`

Do not infer a Benchmark B result from Benchmark A. Read
`phase3_credit_estimator_stratified_summary.json` and its Markdown/SVG report.
To pass, Before/after ×16 had to improve ranking accuracy over One rollout by
at least `0.10` in a typical resampled trial and win at least `80%` of those
trials. Otherwise the result is `NOT-DEMONSTRATED`.

### Primary predeclared result

Benchmark B is complete: **NOT-DEMONSTRATED**. Across all 50 frozen validation
roots, the benchmark selected one state at each requested depth (`0`, `1`, and
`2`) and evaluated 1,409 unique state-action pairs — 809 at depth 0, 450 at
depth 1, and 150 at depth 2.

| Method | Ranking accuracy |
| --- | --- |
| One rollout | `0.490` |
| Before/after ×1 | `0.320` |
| Before/after ×2 | `0.429` |
| Before/after ×4 | `0.464` |
| Before/after ×8 | `0.503` |
| Before/after ×16 | `0.494` |

The median paired difference (K=16 minus Uniform) was `-0.0227` and K=16 won
`38.1%` of clustered bootstrap replicates, against the predeclared `0.10` and
`80%` thresholds. No replicate was dropped.

These numbers differ slightly from the first reported run (`-0.038`, `30.5%`)
because rollout pools are now drawn from one RNG stream per pool identity
instead of adjacent per-draw seeds. The decision is unchanged.

### Where the aggregate is diluted

The aggregate combines distinct depth regimes. At depth 1, Uniform strongly
leads (`0.999` versus K=16 `0.573`), while at depth 0 K=16 leads (`0.418`
versus Uniform `0.278`). At depth 2 every estimator scores exactly `1.0`. The
aggregate should not be read as a uniform effect across decision depths.

**Why depth 2 separates no estimators.** An earlier draft of this document said
depth-2 exact advantages are all tied. That is wrong: only `10` of the `50`
depth-2 states are all-tied, and the other `40` have genuinely distinct exact
advantages. The real mechanism is that a depth-2 state in a four-number problem
has one merge left, so the rollout from each child state is terminal and
deterministic — all `50` states have constant child pools — and the evaluated
policy is near-deterministic on that last merge, so the parent rollout is
effectively deterministic too (`49` of `50` states have constant parent pools).
Every estimator therefore observes the exact terminal reward of each action and
reproduces the exact ranking: per-state fidelity is `1.0` on all `40` untied
states and undefined on the `10` tied ones, and every estimator detects `40/40`
pivotal actions there. Depth 2 costs a third of the selection budget and
separates nothing, which is a selection-design problem rather than a tie
problem.

### Picking the truly best move

A position counts here when exactly one move is truly best. The method succeeds
when it picks that same move. Of the 150 selected positions, 140 have one
unambiguously best move. Before/after ×16 picks it in `125/140` (`89.3%`) and
×8 also picks `125/140`, while One rollout picks `109/140` (`77.9%`). Accuracy
rises monotonically with K above
K=1 (`68.6%`, `78.6%`, `83.6%`, `89.3%`, `89.3%`).

40 of those pivotal states are the depth-2 states every estimator gets right.
Restricted to the 100 pivotal states at depths 0 and 1, the gap is wider:
Uniform `69/100`, K=1 `56/100`, K=2 `70/100`, K=4 `77/100`, K=8 `85/100`,
K=16 `85/100`.

### Extra checks run after seeing the result

These are diagnostics. They do not redefine the gate above and were not used to
choose it. All four were added after the run, and all four share the primary
bootstrap seed (`20260913`) and the same 1,000 replicate stream, so they are
correlated with the primary result and with each other. Read each one against
the primary, not as an independent replication.

| Sensitivity | States / actions | Decision | Median paired difference | Support |
| --- | --- | --- | --- | --- |
| Shared per-state baseline | 150 / 1,409 | `CAPABLE` | `0.3479` | `100.0%` |
| Stratum-balanced depths | 150 / 1,409 | `NOT-DEMONSTRATED` | `-0.1077` | `0.2%` |
| Nondegenerate states only | 140 / 1,379 | `NOT-DEMONSTRATED` | `-0.0098` | `45.2%` |
| Depths 0 and 1 only | 100 / 1,259 | `NOT-DEMONSTRATED` | `0.0006` | `50.1%` |

The **shared-per-state-baseline** variant replaces the predeclared independent
per-action parent pool with a single parent pool shared by every action at a
state. Under it, Uniform's fidelity is unchanged at `0.490` while K=16 rises to
`0.913` (K=1 `0.490`, K=2 `0.635`, K=4 `0.750`, K=8 `0.865`).

The **stratum-balanced** variant weights each populated depth equally and puts
Uniform ahead (`0.759` versus K=16 `0.664`), which shows how much depth 1 and
the uninformative depth 2 drive the pooled number.

The **nondegenerate-states** filter removes exactly the `10` states — `30`
actions — whose exact advantages are all tied and on which no estimator can be
scored. All `10` happen to sit at depth 2, but the filter does not remove the
depth-2 stratum: `40` depth-2 states remain. It barely moves the primary result
(`0.449` Uniform versus `0.473` K=16).

The **depths 0 and 1 only** variant is the one that actually excludes the
nondiscriminating stratum, dropping all `50` depth-2 states and `150` actions.
Fidelity becomes Uniform `0.4003`, K=1 `0.2225`, K=2 `0.3528`, K=4 `0.3929`,
K=8 `0.4406`, K=16 `0.4321`. The median paired difference moves from `-0.0227`
to `+0.0006` and support from `38.1%` to `50.1%` — the sign flips, but the
result is still nowhere near the `0.10` / `80%` gate, so depth-2 dilution alone
does not explain the primary outcome.

### Qualified conclusion

Under the predeclared design, Benchmark B is `NOT-DEMONSTRATED`: Sampled
Counterfactual Advantage at `K <= 16` does not beat Uniform Outcome Credit by
the required margin on pooled within-state ranks.

That conclusion is specific to the predeclared design, and the diagnostics say
so. Drawing an independent parent pool per action injects baseline noise that
varies within a state; with one baseline per state the same rollout budget
makes K=16 clearly better than Uniform on every replicate. Pivotal-action
detection, which the pooled rank correlation is not sensitive to, also favours
larger K. Excluding depth 2 flips the sign of the paired difference but leaves
the gate far out of reach, so the baseline design — not depth dilution — is the
dominant factor.

The honest reading is that the predeclared pooled-rank gate was not met, and
that the estimator does show capability once the per-action baseline noise is
removed — which is a design question for the next benchmark, not a reason to
restate this one's decision.

## Run provenance

Both benchmarks were produced by a single paid RunPod launch; all later
analysis is cache-only and model-free.

- Image digest actually used:
  `us-central1-docker.pkg.dev/gen-lang-client-0367367639/cloud-run-source-deploy/rl-research-runpod@sha256:5915d6ff0feaa3952cae4cb439a10d4259e919de2472df1bd3fd18e9bcba4e9a`
  (Cloud Build `d1916fc5-07a7-4985-a553-04e481c42b21`).
- Preflight projection: `$1.44` worst case at `$0.24/hour` for six hours,
  below the `$50` ceiling.
- Pod `8e49ansek76x04` ran an earlier digest, exited with experiment status
  `2` (`--benchmark stratified` unrecognized) after roughly six minutes, and
  was terminated.
- Pod `zcpa1es49yb4m5` ran the pinned digest, completed with experiment status
  `0` after 74m22s on an NVIDIA RTX 2000 Ada Generation in `EU-RO-1`, and was
  terminated.
- Duration-based cost estimate, not an invoice: about `$0.30` and `$0.02`
  respectively.

## Reproducing Benchmark B

Every Benchmark B number above can be regenerated deterministically from the
raw checkpoint archive without loading a model or paying for another run.
Restore the release archive using the commands in the README first.

```bash
uv run python -m experiments.benchmark_credit_estimator_fidelity \
  --validation-dataset data/v1/validation.jsonl \
  --benchmark stratified \
  --output artifacts/phase3/credit-estimator-pilot/qwen3.5-4b \
  --problem-count 50
```

## Joint interpretation

The two decisions stay separate. The observed pair is Benchmark A `REWORK` and
Benchmark B `NOT-DEMONSTRATED`, the third row below, qualified by the
sensitivity analyses above.

| Benchmark A | Benchmark B | Interpretation |
| --- | --- | --- |
| `REWORK` | `CAPABLE` | The estimator works on consequential choices, but ordinary trajectories do not contain enough measurable credit at this budget; consider curriculum, harder tasks, informative-decision targeting, or larger `K`. |
| `CONTINUE` | `CAPABLE` | The estimator is both capable and practically relevant under ordinary behavior. |
| `REWORK` | `NOT-DEMONSTRATED` | Neither practical relevance nor diagnostic capability has been established at this K budget. |
| `CONTINUE` | `NOT-DEMONSTRATED` | Investigate a benchmark or aggregation inconsistency before proceeding. |
