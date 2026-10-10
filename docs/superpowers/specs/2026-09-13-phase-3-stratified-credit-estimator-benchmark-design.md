# Phase 3 Dual Credit-Estimator Benchmark Design

**Date:** 2026-09-13

## Goal

Preserve the initial on-policy Phase 3 pilot as the prevalence benchmark and add
a stratified diagnostic benchmark that can distinguish Uniform Outcome Credit
from Sampled Counterfactual Advantage at `K = 1, 2, 4, 8, 16`.

The completed on-policy pilot remains an immutable primary artifact for its
question. It measured how often informative credit occurs under sampled behavior,
and its
1,600 observations contained 847 exact-zero advantages, 1,191 advantages with
`|A| < 0.01`, and 41% duplicate state-action pairs. Its `REWORK` decision is
preserved as evidence that `K <= 16` offers little practical benefit over
Uniform on naturally sampled decisions.

The new stratified benchmark asks a different question: when consequential
action choices are present, can the estimator recover their relative credit?
Neither benchmark replaces or gates the other.

## Benchmark A — On-Policy Prevalence

Keep the existing on-policy implementation, raw per-problem artifacts, aggregate
JSON, summary, plots, and gate unchanged. Do not rerun it, alter its bootstrap,
or retroactively filter its selected decisions.

Its conclusion is:

> Naturally sampled four-number decisions are dominated by zero or
> below-resolution advantages, so Sampled Counterfactual Advantage at
> `K <= 16` does not reliably outperform Uniform Outcome Credit.

## Benchmark B — Stratified Capability

### Selection Population

Use the same 50 deterministic four-number problems selected from
`data/v1/validation.jsonl`. Selection must depend only on the exact
menu-order-averaged constrained policy and never on sampled estimator outcomes.

For each problem and each decision depth in `{0, 1, 2}`:

1. enumerate states at exactly that depth with at least three legal actions;
2. compute every legal action's exact `Aπ`;
3. select the state with the largest action-advantage spread,
   `max(Aπ) - min(Aπ)`;
4. break spread ties by higher on-policy state occupancy, then canonical state
   order;
5. evaluate every legal action at the selected state.

If a problem has no eligible state at a depth, record that depth as unavailable;
do not substitute another depth. Report selection coverage by depth.

This produces unique state-action pairs and preserves the horizon comparison.
It does not impose a post-result magnitude threshold or discard selected
actions whose exact advantage happens to be zero.

### Policy Cache

Each problem checkpoint stores the menu-order-averaged action probabilities for
every nonterminal DAG state, together with:

- canonical problem hash and root state;
- model and tokenizer repositories and immutable revisions;
- prompt version, temperature, rules, and all five menu-ordering identifiers;
- node count and scoring duration;
- the raw rollout outcome pools described below.

The cache identity deliberately excludes `pairs_per_problem`, which only
controls Benchmark A's on-policy sampling. Older caches that recorded it are
migrated in place rather than invalidated.

Resumption validates every field and fails closed on mismatch. The complete
probability table allows selection, rollout simulation, bootstrap analysis, and
report regeneration without loading the model again. A cache-only run must not
rewrite a checkpoint whose bytes are unchanged.

### Estimator Samples

For every selected state-action pair, simulate independent policy rollouts from:

- the selected parent state, estimating `Vπ(s)`;
- the action's child state, estimating `Qπ(s,a)`.

Use one deterministic pool of 16 outcomes per arm, drawn from a single RNG
stream seeded by that pool's identity. Also draw one additional parent pool per
selected *state*, shared by all of its actions, for the sensitivity analysis
below.

Checkpoints persist those raw parent and child outcomes, never the derived
estimator samples. Derived samples are a function of the raw draw:

- Uniform Outcome Credit is the first child outcome, with no parent baseline.
- Sampled Counterfactual Advantage at `K` is `Q̂K(s,a) - V̂K(s)` over the first
  `K` paired outcomes, so every `K` is a nested prefix of the same draw.

Exact `Aπ(s,a)` is the reference. Storing derived pools instead would let the
bootstrap resample each estimator independently and destroy the pairing the
gate measures.

### Pivotal-Action Detection

A selected state is **pivotal** when exactly one of its actions attains the
maximum exact advantage. Only pivotal states can separate an estimator that
finds the right action from one that merely orders ties consistently.

An estimator **detects** the pivotal action at a state when its mean credit has
a unique argmax and that argmax is the same action. The reported metric is
`detected / pivotal` per estimator. Defined-state coverage is a separate
secondary diagnostic and must never be reported or plotted as pivotal-action
detection.

### Fidelity and Bootstrap

The primary fidelity metric compares action rankings within the same state:

1. convert exact advantages to average within-state percentile ranks;
2. convert each estimator's credits to the same rank representation;
3. pool those within-state ranks across selected states;
4. compute Spearman correlation between estimator and exact ranks.

This prevents differences in baseline puzzle difficulty from masquerading as
action-level credit fidelity. Report ordinary per-state Spearman correlations
and their defined-state coverage as secondary diagnostics.

Bootstrap 1,000 paired replicates. Each replicate:

1. resamples the 50 problems as clusters;
2. retains all selected states and actions for each sampled problem;
3. resamples the raw parent and child outcomes exactly once per state-action
   pair, before any estimator is derived;
4. derives and computes every estimator on that same resampled draw.

Benchmark B returns `CAPABLE` only when:

- the median of paired per-replicate `(K=16 fidelity - Uniform fidelity)`
  differences is at least `0.10`; and
- K=16 beats Uniform in at least `80%` of paired bootstrap replicates.

Otherwise it returns `NOT-DEMONSTRATED`. These labels are deliberately
different from Benchmark A's `CONTINUE` / `REWORK` gate because the benchmarks
answer different questions.

### Post-Hoc Sensitivity Analyses

**Amended after the run (2026-09-20).** This section was not part of the
predeclared design; it was added once the completed benchmark showed the
primary result depended on design choices the spec had not examined. The gate
above is predeclared and remains the decision. The variants below are computed
with identical statistics, reported alongside it, and explicitly labelled
post-hoc diagnostics. They qualify how the primary result should be read; they
never replace it, and the primary gate is never redefined after seeing them.

All four share the primary bootstrap seed and replicate stream, so they are
correlated with the primary result and with each other rather than being
independent replications.

1. **Shared per-state baseline.** One parent pool per selected state, shared by
   all of its actions, instead of an independent pool per action. The bootstrap
   resamples that one pool per state so the variant survives resampling. This
   isolates how much of the primary result comes from action-specific baseline
   noise rather than from the estimator.
2. **Stratum-balanced depths.** Per-depth fidelity averaged with equal weight
   per populated depth, so depths with many selected actions do not dominate a
   pooled rank correlation.
3. **Nondegenerate states only.** States whose exact advantages are not all
   tied. This removes only the states no estimator can be scored on; it is not
   a way to exclude a depth stratum.
4. **Depths 0 and 1 only.** Excludes the depth-2 stratum outright. A depth-2
   state in a four-number problem has one merge left, so its child rollouts are
   terminal and deterministic and the evaluated policy is near-deterministic on
   that merge. Every estimator therefore recovers the exact ranking there, and
   the stratum separates no estimators regardless of whether its exact
   advantages are tied.

## Joint Interpretation

Report both decisions without collapsing them into one gate:

- A `REWORK`, B `CAPABLE`: the estimator works on consequential choices, but
  ordinary trajectories do not contain enough measurable credit at this budget;
  consider curriculum, harder tasks, informative-decision targeting, or larger
  `K`.
- A `CONTINUE`, B `CAPABLE`: the estimator is both capable and practically
  relevant under ordinary behavior.
- A `REWORK`, B `NOT-DEMONSTRATED`: neither practical relevance nor diagnostic
  capability has been established.
- A `CONTINUE`, B `NOT-DEMONSTRATED`: investigate a benchmark or aggregation
  inconsistency before proceeding.

## Artifacts

Write to a new `stratified/` artifact directory:

- one resumable policy checkpoint per problem, carrying the raw rollout pools;
- canonical aggregate JSON, which records each checkpoint's relative path and
  SHA-256 rather than embedding its contents;
- concise Markdown findings;
- fidelity by `K`;
- fidelity by depth;
- pivotal-action detection rate by estimator;
- fidelity versus rollouts actually spent per decision, where Uniform costs `1`
  rollout and `K` costs `2K`, making this plot distinct from fidelity by `K`.

The on-policy artifact is not overwritten or silently mixed into Benchmark B.
Every Benchmark A file is committed and covered by a tracked SHA-256 manifest.

### Download Safety

The generic RunPod helper untars whatever a Pod produced into the directory it
is given, so the Phase 3 launcher must never give it the Benchmark A root. It
downloads into an isolated staging directory and copies only the `stratified/`
subtree into place. The Benchmark A manifest is verified before the launch,
before the copy, and after it; a mismatch, a missing subtree, or a symbolic
link in the staged output aborts without touching Benchmark A.
Add a concise joint findings document that links both artifact directories and
states both conclusions. User-facing labels use “Sampled Counterfactual
Advantage”; internal Phase 2 `mc_reset_k*` keys remain unchanged for artifact
compatibility.

## Module Boundaries

Keep `experiments/evaluate_policy_bridge.py` unchanged.

Extend `experiments/benchmark_credit_estimator_fidelity.py` only with Phase 3
orchestration and artifact lifecycle. Put reusable within-state ranking and
problem-clustered bootstrap calculations in
`analysis/credit_estimator_fidelity.py`. Do not create a generic utilities
module.

## Tests

Tests must prove:

- exact depth assignment and widest-spread state selection;
- occupancy and canonical tie-breaking;
- no duplicate selected state-action pairs, checked against the total number of
  selected actions;
- unavailable depths remain unavailable;
- policy visit probability propagation and candidate construction on their own;
- deterministic, pool-specific rollout seeding;
- policy checkpoint round-trip and metadata rejection;
- legacy cache-identity migration;
- regeneration without model loading, and without rewriting unchanged
  checkpoints;
- nested estimator prefixes derived from one raw draw;
- heterogeneous or unpaired rollout pools raise a descriptive `ValueError`;
- analysis is invariant to checkpoint key order and to estimator iteration
  order;
- within-state rank normalization;
- problem-clustered paired bootstrap reproducibility;
- the shared-baseline bootstrap keeps one parent pool per state;
- both Benchmark B gate boundaries;
- plots carry their stated semantics: real detection rates, and a cost axis on
  the frontier;
- the on-policy artifact is never overwritten or reinterpreted, its manifest
  verifies, and every manifest entry is committed;
- a failed or partial download cannot reach Benchmark A;
- smoke and RunPod commands target the stratified artifact directory.

Run focused tests, the full fast suite, diff checks, and IDE lints before a paid
rerun.
