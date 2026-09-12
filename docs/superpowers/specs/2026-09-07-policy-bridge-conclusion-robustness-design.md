# Policy Bridge Conclusion-Robustness Design

**Date:** 2026-09-07

## Goal

Replace the original complete-DAG Stage B with a compute-bounded benchmark that
asks:

> Do the practical conclusions about which credit estimators work best survive
> when evaluation uses full-solution Qwen completions instead of the constrained
> next-token policy?

The benchmark validates a conclusion, not equality between two policies. It
does not claim that constrained next-token probabilities reproduce natural
chain-of-thought behavior.

## Motivation

The original Stage B estimates a deliberative action distribution at every
nonterminal state in three complete DAGs. That requires roughly 50,000
sequential generations and costs far more than the project budget.

The single-action deliberative prompt also failed to produce a usable policy:
Qwen3.5-4B commonly spent the full token allowance searching and never emitted
the required action tag. A missing sampled policy makes the original Stage B
unmeasurable rather than demonstrating policy disagreement.

Phase 3 ultimately needs a narrower decision: whether Uniform outcome credit
and MC-reset at `K = 1, 2, 4, 8, 16` exhibit the same broad fidelity pattern
under constrained and more natural completion behavior.

## Scope

The revised benchmark:

- remains available as `experiments.evaluate_policy_bridge stage-b`;
- runs independently of the original Stage A gate;
- uses the same 12 roots selected deterministically from `data/v1/test.jsonl`;
- compares estimator-level conclusions on 24 stratified state-action pairs;
- uses exact constrained-policy credit on one side;
- uses held-out full-solution rollouts as a sampled deliberative reference;
- has a predeclared RunPod budget ceiling of `$10`;
- produces `GO`, `NO-GO`, or `INCONCLUSIVE`.

It does not:

- reconstruct a deliberative policy over a complete DAG;
- treat sampled deliberative credit as an exact oracle;
- require exact equality of estimator scores or ordering;
- change the model from `Qwen/Qwen3.5-4B`;
- change Stage A or reinterpret a failed Stage A as policy disagreement.

## Completion Contract

Full-solution rollouts use the project environment rather than TinyZero's
different arithmetic rules:

- allowed operators are `+`, `-`, and `*`;
- subtraction uses the larger operand minus the smaller operand;
- every intermediate result is positive;
- an input number may be used at most once;
- unused numbers are allowed once an expression reaches the target.

The prompt contains only the current state, rules, and requested output format.
It asks for ordinary generated reasoning followed by one final expression in an
`<answer>...</answer>` tag. The Qwen chat template uses
`enable_thinking=False`.

The parser reads only the final complete answer tag. It evaluates the expression
tree exactly, verifies input-number multiplicities, allowed operations, positive
intermediates, subtraction ordering, and equality to the target. Missing,
malformed, illegal, wrong, and truncated outputs are rollout failures. They are
saved verbatim and are never inferred, retried, or discarded.

A state that already contains the target succeeds without generation. A
terminal state that cannot reach the target fails without generation.

## Deterministic State-Action Selection

Build complete DAGs for the same 12 deterministic roots. Score the constrained
policy under the five predeclared menu orderings and average by semantic action.
Use the resulting exact constrained policy to compute occupancy and advantage.

Select one state per root:

- assign desired depths `0`, `1`, and `2` cyclically by canonical root index,
  yielding four roots per depth;
- among eligible states at the assigned depth, choose the state with highest
  constrained-policy occupancy;
- break ties by canonical state order;
- require at least three legal actions so top, middle, and bottom advantage
  strata are defined.

Select two actions from each state. Rotate the omitted stratum by canonical root
index so top-, middle-, and bottom-ranked constrained advantages each appear
eight times across the 24 pairs. Advantage ties use semantic `Action.sort_key`
ordering. Persist all candidates, selection ranks, occupancies, advantages, and
tie-break decisions.

Selection uses constrained credit only to guarantee informative coverage. It
does not inspect deliberative outcomes.

## Rollout Pools

There are 36 rollout arms:

- 12 state arms estimate `V(s)`;
- 24 child-state arms estimate `Q(s,a)`.

Each non-deterministic arm has two independent seed ranges:

1. **Reference pool:** 32 or 64 completions used only to estimate the
   deliberative reference advantage.
2. **Estimator pool:** 16 completions used only to evaluate Uniform and
   MC-reset `K = 1, 2, 4, 8, 16`.

The K estimators use nested prefixes of the 16-sample estimator pool. Sharing
prefixes makes differences between K values paired and avoids extra inference.
Bootstrap resampling represents repeated estimator draws.

Uniform outcome credit uses one post-action completion outcome without a state
baseline. MC-reset uses K post-action outcomes for `Q` and K current-state
outcomes for `V`, then reports `Q - V`.

For the constrained comparison, exact `Vπ`, `Qπ`, and `Aπ` are the reference.
Estimator sampling is simulated reproducibly from the exact policy, requiring
no additional model inference.

## Calibration and Budget Enforcement

The `calibrate` command runs four project-rule completions for each of the
canonical root states at indices `0`, `5`, and `11`, at token caps `1024`,
`2048`, and `3072`. It records generated token counts, truncation, format
validity, correctness, and local MPS throughput. Choose the smallest cap for
which at most `25%` of responses consume the full allowance. If no cap meets
that criterion, stop before paid inference and report that the completion
policy could not be measured within the predeclared range.

A tiny RunPod preflight then measures actual 4090 generation throughput using
the selected token cap and production batch size. Its cost is included in the
`$10` Stage B ceiling.

Using the observed 4090 rate and a 20% contingency:

- choose 64 reference samples if the projected total is at most `$10`;
- otherwise choose 32 if its projected total is at most `$10`;
- otherwise refuse the full run.

The projected response count is 2,880 with a 64-sample reference or 1,728 with
a 32-sample reference, before subtracting deterministic terminal arms.
Reference size is chosen from cost only, never from observed estimator results.

## Amendments After Calibration (2026-09-07)

Local MPS calibration ran before any paid inference and produced these rates
over 12 responses per cap:

| Token cap | Truncated | Answer tag | Solved | Mean tokens |
| --- | --- | --- | --- | --- |
| `1024` | `58.3%` | `50.0%` | `50.0%` | `849` |
| `2048` | `8.3%` | `91.7%` | `66.7%` | `954` |
| `3072` | `16.7%` | `100%` | `100%` | `1321` |

The predeclared rule selected `2048`, which is the cap the benchmark uses. The
`3072` row shows higher truncation than `2048` purely from sampling noise at
this sample size, and its `100%` success rate is a reason for caution rather
than preference: if the completion policy solves nearly every state then
`V ≈ 1` everywhere, every solution-preserving action has `A ≈ 0`, and the
sampled reference loses the spread the fidelity metric needs.

Three predeclared values are amended:

1. **Reference pool size is `256`, not `64` or `32`.** A reference advantage is
   a difference of two binomial means, so `64` samples resolve only
   `|A| >= 0.17`. Exact advantages under a uniform policy in these DAGs put the
   middle-ranked action at roughly `0.006` to `0.09`, so `64` cannot rank the
   mid-range at all. `256` resolves `|A| >= 0.087`. `REFERENCE_POOL_SIZES` is
   widened to `256, 128, 64, 32`, ordered so budget planning takes the most
   precise pool that fits.
2. **Cost is projected from measured wall time per response, not from a token
   cap and a token rate.** The original basis multiplied the worst-case token
   count by a throughput measured under realistic lengths, double-counting the
   pessimism and overstating spend by `2.23x` against the first 4090
   measurement. `project_stage_b_cost` now takes `seconds_per_response`, which
   already carries the real token distribution and the fact that a batch runs
   until its longest member finishes.
3. **The `$10` ceiling is retained but made explicit.** `plan-budget` and
   `preflight` take `--budget-ceiling-usd` so any override is stated at the
   call site instead of silently editing the predeclared constant. At the
   measured throughput the `256`-sample reference needs only `6.9` GPU-hours,
   so no override is required.

The paid preflight measures throughput at the production batch size and sweeps
batch size, because decoding a 4B model is memory-bandwidth bound at small
batches. Measured on an RTX 4090 at cap `2048`:

| Batch size | Seconds per response | Tokens per second | Mean tokens |
| --- | --- | --- | --- |
| `8` | `10.60` | `118.6` | `1257` |
| `16` | `6.43` | `166.1` | `1067` |
| `32` | `3.54` | `278.0` | `983` |
| `64` | `2.53` | `385.1` | `973` |

Stage B therefore runs at batch size `64`. An earlier measurement on a
different community host reported `21.64` seconds per response at batch `8`,
five times slower than the `4.2` seconds the same batch size implies here, so
host-to-host variance on community cloud is large enough that a throughput
measurement is only valid for the host that produced it.

Two infrastructure requirements follow from the run length:

- Pods reference the image by **digest**, not by a mutable tag. Three paid Pods
  exited on argument errors from code that predated the tag they pulled.
- `/workspace` is backed by a **network volume** so batch checkpoints and the
  cached model survive a host failure, and a resumed run continues instead of
  restarting. One Pod's GPU failed to initialise outright, which at hour 13 of
  an unvolumed run would discard the whole run.

Completion temperature stays at `0.6` and the estimator pool stays at `16`.
Lowering the temperature would make every rollout from a state identical, so
MC-reset would return the same estimate at every `K`, the fidelity slope
against `log2(K)` would be exactly zero, and decision criterion 2 would be
undefined rather than tested. The estimator pool is the `K = 16` ceiling under
evaluation, not a precision knob.

Model and tokenizer revision measured during calibration:
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, `float16` on `mps`, local
throughput `33.1` tokens per second.

Final Stage B configuration: token cap `2048`, reference pool `256`, estimator
pool `16`, batch size `64`, completion temperature `0.6`, on an RTX 4090 in
`EU-RO-1` with network volume `rl-research-bridge`.

## Metrics

For every bootstrap replicate, calculate estimator credit for all 24 pairs and
compare it with the corresponding reference advantages.

Primary fidelity metric:

- Spearman rank correlation across the 24 state-action pairs.

Secondary metrics:

- sign accuracy for pairs whose reference uncertainty excludes zero;
- fidelity by depth;
- fidelity by top/middle/bottom constrained-credit stratum;
- completion format, legality, success, and truncation rates;
- total generated tokens, wall time, and estimated cost.

Reference stability is measured with 1,000 deterministic replicates. Each
replicate randomly splits every held-out arm into equal halves, derives two
advantage vectors, and computes their rank correlation.

## Decision Rule

Return **INCONCLUSIVE** if the sampled deliberative reference is not stable
enough to evaluate estimator rankings. The predeclared stability threshold is a
median split-half Spearman correlation of at least `0.5`.

Given a stable reference, return **GO** only when all conditions hold:

1. MC-reset `K=16` has higher rank-correlation fidelity than Uniform under both
   constrained and deliberative evaluation, with at least `80%` bootstrap
   support in each setup.
2. The least-squares slope of fidelity against `log2(K)` is positive under both
   setups, with at least `80%` bootstrap support.
3. The Spearman correlation between the six method-level median fidelity scores
   (`Uniform`, `K=1`, `K=2`, `K=4`, `K=8`, `K=16`) in the two setups is at least
   `0.7`.

Return **NO-GO** when the reference is stable but any directional condition
fails.

Adjacent K values may swap order without failing the benchmark. The gate tests
the broad research conclusion, not an exact total ordering.

## Modules and Integration

Keep orchestration and CLI integration in
`experiments/evaluate_policy_bridge.py`.

Use focused support modules:

- `policies/countdown_completion_policy.py`: prompt and strict project-rule
  expression scoring;
- `inference/huggingface_completion_generator.py`: model loading and batched
  repeated generation;
- `analysis/conclusion_robustness_metrics.py`: selection, estimator simulation,
  bootstrap metrics, reference stability, and decision logic.

The original Stage A implementation and artifacts remain intact. Revised Stage
B does not read or require a Stage A artifact. Existing old-schema Stage B
artifacts fail with a clear incompatibility error rather than being reused.

## Artifacts and Resumption

Write canonical JSON and a concise Markdown summary. The JSON records:

- immutable model and tokenizer revisions;
- package versions, device, dtype, temperature, token cap, and batch size;
- prompt version and project rules;
- local calibration and RunPod preflight measurements;
- projected and observed spend;
- deterministic root, state, and action selection evidence;
- every raw completion, parse result, pool identity, batch seed, and sample
  index;
- exact constrained reference values;
- sampled deliberative reference values and uncertainty;
- per-method bootstrap distributions;
- aggregate and subgroup metrics;
- gate thresholds, evidence, and final three-way decision.

Checkpoint after every generated batch. Resumption requires exact metadata and
selection equality. Any mismatch fails rather than reusing stale responses.

## Tests and Verification

Focused tests cover:

- legal subset expressions and positive intermediate enforcement;
- rejection of division, repeated inputs, malformed tags, wrong targets, and
  illegal subtraction;
- deterministic depth, occupancy, action-stratum, and tie selection;
- independent reference and estimator seed ranges;
- nested K prefixes;
- Uniform and MC-reset calculations;
- constrained-policy estimator simulation;
- split-half stability;
- bootstrap reproducibility;
- all GO, NO-GO, and INCONCLUSIVE boundaries;
- deterministic terminal arms;
- checkpoint resumption and metadata mismatch;
- budget projection and refusal;
- fake-model batching;
- model-free imports and CLI help.

After implementation, run focused tests, tests for modified shared modules, the
fast full suite, and lints. Run local calibration before any paid preflight.
Do not launch the full paid benchmark until the projection including
contingency is within `$10`.
