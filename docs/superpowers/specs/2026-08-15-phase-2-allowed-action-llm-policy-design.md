# Phase 2 Allowed-Action LLM Policy Design

**Date:** 2026-08-15

## Goal

Integrate Qwen3-0.6B as a stationary, forced-choice policy over the existing
Countdown environment, compute numerical `Vπ`, `Qπ`, and `Aπ` over one complete
reasoning DAG, validate the result against Monte Carlo rollouts, and produce one
reproducible trajectory artifact suitable for explaining the research in a
blog post.

Phase 2 is complete when the pipeline is validated on the worked problem
`(3, 7, 10, 25) → 245` and produces the approved artifact. It does not require
evaluating the full frozen dataset.

## Research context

Phase 1 built a deterministic arithmetic environment whose complete future
state graph can be enumerated. It also built exact `Fraction`-based evaluators
for hand-coded stationary policies.

Phase 2 replaces a hand-coded policy with an LLM policy:

```text
canonical state
      │
      ▼
state-only prompt and legal-action menu
      │
      ▼
Qwen next-token scores
      │
      ▼
normalized π(a|s)
      │
      ▼
numerical DAG recurrence
      │
      ▼
Vπ(s), Qπ(s,a), Aπ(s,a)
```

The central correctness constraint is stationarity. The LLM's action
probabilities must depend only on the canonical current state and its legal
actions. If trajectory history or carried-over reasoning affects the policy,
different histories that reach the same merged DAG node are no longer
equivalent and the Phase 1 oracle structure is invalid.

## Scope

Phase 2 includes:

- a state-only, forced-choice prompt;
- deterministic action-menu randomization;
- tokenizer-validated single-token option labels;
- batched Hugging Face inference on Apple Silicon, portable to CUDA;
- a resumable, versioned action-probability cache;
- a floating-point policy evaluator separate from the exact `Fraction`
  evaluator;
- a throughput probe before any larger evaluation;
- Monte Carlo validation of the LLM policy's root value;
- one deterministic, transparently selected trajectory artifact.

Phase 2 excludes:

- free-form chain-of-thought;
- carrying conversation or reasoning history between steps;
- RL training;
- credit-estimator benchmarking;
- a full 200–300 problem diagnostic study;
- a permanent n=4/n=5/n=6 exact-versus-Monte-Carlo policy;
- custom inference servers or distributed infrastructure.

Free-form reasoning remains a later robustness extension. Under free-form
reasoning, `π(a|s)` is the marginal probability over all reasoning traces that
end in action `a`, so it must be estimated by sampling rather than read from
one model pass.

## Policy contract

The policy consumes:

- one canonical `State`;
- the complete tuple returned by `allowed_actions(state)`.

The rendered prompt contains:

- the current sorted numbers;
- the target;
- the Phase 1 arithmetic rules;
- every legal action exactly once;
- an instruction to answer with one option label.

Render the prompt through Qwen's chat template with
`add_generation_prompt=True` and `enable_thinking=False`. Qwen3 enables
thinking by default; leaving that default active would make the immediate
next-token distribution target a reasoning preamble rather than the requested
action label. The resolved tokenizer revision and the `enable_thinking` setting
are part of the prompt version.

The prompt must not contain:

- prior states or actions;
- trajectory depth except information already implied by the current state;
- previous model output or reasoning;
- `V*`, `Q*`, solution paths, or any other oracle information.

### Deterministic menu order

The menu permutation is seeded by a `blake2b` digest over:

- a canonical serialization of the state;
- a recorded menu ordering ID;
- the serialized active `RuleSet`.

Python's built-in `hash()` is forbidden because it is randomized per process.
The same state, rules, and ordering ID must render byte-identically in every process.

### Option labels

Six-number states can expose more than 26 legal actions, so `A`–`Z` alone is
insufficient. The scorer owns a fixed, versioned candidate label inventory.
At model load time, it filters the candidates to labels that satisfy all of
these conditions in the exact answer context:

1. the label extends the prompt by exactly one token;
2. its token ID has not already been assigned to another selected label;
3. decoding that token produces exactly the intended label;
4. at least 45 valid labels survive filtering.

Invalid candidates are excluded rather than treated as selected labels. If
fewer than 45 candidates survive, evaluation aborts. The implementation must
never truncate the legal-action set or switch silently to multi-token labels.

### Action probabilities

For each prompt, the scorer reads next-token logits for the option-label token
IDs and applies a masked softmax over only the labels corresponding to legal
actions. This renormalized distribution defines the experimental
forced-choice policy.

Temperature is explicit in run configuration and defaults to `1.0`. The
scorer must return one finite, nonnegative probability per legal action, with
the probabilities summing to one within `1e-6`.

This policy is intentionally constrained. Results must be described as
properties of the forced-choice LLM policy, not the model's unrestricted
generation distribution.

## Architecture

Use a two-stage scoring and evaluation pipeline:

```text
enumerate DAG
    │
    ▼
render all unique nonterminal states
    │
    ▼
batch-score missing states ──▶ resumable cache
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
          numerical evaluator             trajectory sampler
                   │                             │
                   ▼                             ▼
             Vπ / Qπ / Aπ                  blog artifact
```

Expensive model inference is separated from cheap value computation and
analysis. Once a complete cache exists, evaluation and artifact generation
must not load the model.

## Components

### `policies/allowed_action_llm_policy.py`

Owns:

- canonical state serialization for prompts and cache keys;
- deterministic menu permutation;
- prompt rendering;
- label-to-action mapping;
- cache metadata and record validation;
- a cache-backed implementation of the existing `Policy[float]` protocol.

It does not load model weights or perform tensor operations.

### `inference/huggingface_allowed_action_scorer.py`

Owns:

- loading the resolved Qwen model and tokenizer revisions;
- selecting MPS or CUDA;
- validating the single-token label inventory;
- batching prompts;
- extracting label logits and computing masked softmax probabilities;
- reporting throughput and device metadata.

The initial backend is Hugging Face Transformers with PyTorch MPS. The same
code path must accept CUDA without changing the policy or oracle layers. MLX
is not part of Phase 2.

The module exposes a deep scorer interface rather than module-level pipeline
steps:

- `ValidatedLabelInventory.from_tokenizer(...)` owns contextual label
  validation, and `token_ids_for(menu)` enforces label/token ordering;
- `ScorerMetadata` groups immutable model, tokenizer, device, dtype,
  temperature, and label information;
- `HuggingFaceAllowedActionScorer` exposes only `load`, `score_states`, and
  `benchmark`;
- `benchmark` owns representative-state selection and synchronization and
  returns both selected states and measurements as one result.

Device resolution and masked softmax are implementation details inside the
scorer. They are not separate module interfaces.

MPS does not support float64 tensors. For numerically stable normalization, the
scorer copies only the selected option logits (at most 45 values) to CPU before
converting them to float64 and applying softmax. Model inference and full-vocab
logits remain on MPS.

### `oracle/float_policy_evaluation.py`

Owns numerical policy evaluation over the existing `ReasoningDag`.

It remains separate from `oracle/evaluation.py` so the existing exact
`Fraction` semantics and tests do not acquire tolerance-based behavior.
It computes values through the same reverse-topological recurrence but stores
floating-point values.

Before evaluation, it validates:

- exact action-key coverage;
- finite probabilities;
- nonnegative probabilities;
- probability sum within `1e-6` of one.

Malformed policy output raises the existing `PolicyValidationError`. The
evaluator does not clip or renormalize malformed probabilities.

### `experiments/evaluate_allowed_action_llm_policy.py`

Owns orchestration:

1. construct the worked problem and DAG;
2. run or load the throughput probe;
3. score missing nonterminal states into the cache;
4. reject incomplete or stale caches;
5. run numerical policy evaluation;
6. perform the Monte Carlo check;
7. select the trajectory under the declared rule;
8. write machine-readable and blog-ready artifacts.

## Cache and reproducibility

Each run directory contains one `allowed_action_policy_cache.json` snapshot.
The snapshot has a `metadata` object for cache identity and a `records` array
with one entry per completed nonterminal state.

The metadata records:

- cache-schema version;
- model repository and resolved immutable revision;
- tokenizer repository and resolved immutable revision;
- prompt version;
- chat-template arguments, including `enable_thinking=False`;
- candidate labels and token IDs;
- menu ordering ID;
- temperature;
- serialized `RuleSet`;
- device type and tensor dtype;
- PyTorch and Transformers versions;
- canonical root problem.

Every metadata field is cache-identifying. Device, dtype, PyTorch version, and
Transformers version intentionally invalidate the cache because each can
change floating-point probabilities. Monte Carlo and trajectory seeds do not
affect model scoring, so they are recorded only in
`policy_evaluation_summary.json`. Operational details such as the invoked
command are recorded in run outputs but are not part of cache identity.

Each state record contains:

- canonical state;
- canonical allowed-action list;
- rendered menu order;
- probability for every action.

At startup, the scorer loads and validates the snapshot into an in-memory
state-to-record dictionary. After each successful batch, it writes the complete
updated snapshot to a temporary file and atomically replaces the previous
snapshot. At 103 states, rewriting the snapshot is inexpensive and avoids a
separate partial/final cache protocol. An interrupted batch loses at most that
in-flight batch while preserving the previous valid snapshot.

Before numerical evaluation, the orchestrator requires exact coverage of every
nonterminal DAG state. Any metadata mismatch, duplicate state, action-set
mismatch, malformed probability, or missing state is a hard error. Stale data
is never reused silently.

## Throughput gate

Before scoring the complete DAG, select eight deterministic representative
states spaced across the canonical nonterminal-state order. Benchmark only
those states using batch sizes `1`, `4`, and `8`, stopping before an
out-of-memory configuration. Use one warm-up batch before timing, record states
per second and wall-clock duration, and extrapolate the fastest successful
measurement to 103 states. The probe must remain cheaper than the run it gates.

The worked DAG has:

- 236 total nodes;
- 103 nonterminal states requiring model scores;
- 420 action edges.

The first full run proceeds on the Mac if all 103 states can be scored without
out-of-memory failure and the estimated scoring time is at most 30 minutes.
If that gate fails, estimate the cost of the same run on one CUDA EC2 instance
before launch. Phase 2 cloud spending must remain below the agreed $100
budget.

The benchmark also reports projected costs for representative n=4, n=5, and
n=6 DAG sizes, but Phase 2 does not commit to evaluating those strata. A
future horizon experiment must choose one evaluation methodology across
strata or explicitly model the uncertainty introduced by mixing exact-DAG and
Monte Carlo estimates.

## Menu-order sensitivity check

Before publishing the first research post, evaluate the complete worked DAG
under five deterministic menu orderings:

- `phase2-v1`;
- `phase2-sensitivity-1`;
- `phase2-sensitivity-2`;
- `phase2-sensitivity-3`;
- `phase2-sensitivity-4`.

Reuse the existing scorer and write one separate cache directory per menu ordering. A
focused `experiments/compare_menu_ordering_sensitivity.py` module loads the five
completed caches without model weights and produces canonical JSON and
Markdown comparisons.

The comparison reports:

- exact root `Vπ` for each menu ordering;
- probability assigned to the unique winning root action,
  `10 + 25 = 35`;
- each menu ordering's highest-probability root action, option label, and probability;
- pairwise stability of root probability vectors after matching by semantic
  `Action`, not option label;
- average probability assigned to each option label across all nonterminal
  states.

The comparison module has one deep responsibility: turn completed policy
caches into a menu-order robustness result. It does not score models, sample
trajectories, or add generic experiment infrastructure. The original
`phase2-v1` trajectory remains the illustrative artifact; sensitivity runs do
not search for additional showcase trajectories.

The sensitivity check passes as an experiment when all five caches cover the
same 103 nonterminal states and the summary is reproducible byte-for-byte. It
does not require the policy to be invariant. Large variation is a result that
must be reported, not a failure to hide.

## Dual Countdown evaluation

The project uses two deliberately separate evaluation modes.

### Exact-credit mode

This remains the primary research mode. The model sees one canonical state and
one complete allowed-action menu. The prompt explicitly explains that:

- the objective is to eventually make the target appear;
- selected operands are removed and their result is inserted;
- future moves continue from the resulting numbers;
- the model should consider future moves, not only immediate closeness.

The model still returns one label with thinking disabled, preserving the
stationary policy required by the exact DAG oracle.

### Literature-style capability mode

This mode checks whether a model can solve the conventional Countdown task at
all. It does not produce exact per-action credit.

- Dataset: 12 fixed rows beginning at TinyZero's documented held-out boundary
  in `Jiayi-Pan/Countdown-Tasks-3to4`;
- Prompt: produce a complete expression, use every number exactly once, allow
  `+`, `-`, `*`, `/`, and put reasoning/final expression in
  `<think>`/`<answer>` tags;
- Qwen3.5 native thinking mode is disabled; the tags are generated as ordinary
  response text, matching the non-reasoning-model setup used by prior work;
- Sampling: 32 outputs per problem, temperature `0.6`, maximum 1,024 generated
  tokens;
- Generation batch size: `1` on MPS; larger autoregressive batches caused
  severe unified-memory slowdown in measured smoke runs;
- Metrics: average accuracy, Pass@32, All-correct@32, answer-format rate, and
  valid-equation rate;
- Models: Qwen3.5-4B and Qwen3.5-9B evaluated separately.

Expressions are verified with a restricted AST evaluator and exact rational
arithmetic. Arbitrary Python evaluation is forbidden. The capability mode is a
model-selection diagnostic inspired by prior Countdown work, not a replacement
for the project's controlled environment.

## Numerical validation

Set `numerical_tolerance = 1e-9`. The numerical evaluator must satisfy:

- `-numerical_tolerance <= Vπ(s) <= 1 + numerical_tolerance`;
- `abs(Qπ(s,a) - Vπ(child(s,a))) <= numerical_tolerance`;
- `abs(Aπ(s,a) - (Qπ(s,a) - Vπ(s))) <= numerical_tolerance`;
- `abs(Σ_a π(a|s) Aπ(s,a)) <= numerical_tolerance`.

Use `math.fsum` for probability-weighted sums. Numerical results are described
as dynamic-programming values conditional on the cached floating-point policy
probabilities, not as exact rational values. Values are never clipped merely
to make an invariant pass.

For the integration sanity check, run 20,000 seeded trajectories from the root
using the cached policy. Compare the observed success rate with numerical
`Vπ(root)` using the same five-standard-error criterion as
`tests/test_mc_vs_exact.py`. A failure blocks artifact publication.

## Artifact selection and output

Trajectory sampling uses seed `20260815`. The selected trajectory is the first
sampled trajectory that:

1. ends in failure;
2. contains at least one action with `Aπ > 1e-9`;
3. contains at least one action with `Aπ < -1e-9`.

Search at most 10,000 trajectories. If none qualifies, report that result and
do not alter the rule or hand-pick a trajectory. Because the approved Phase 2
goal requires a compelling mixed-credit artifact, a no-match result means
Phase 2 is not complete; choosing a different predeclared problem-selection
protocol requires a new design decision.

The artifact includes, for every step:

- state and rendered action menu;
- selected action;
- selected-action probability;
- resulting state;
- final environment reward;
- outcome-only display credit;
- numerical `Vπ`, `Qπ`, and `Aπ`.

Environment reward remains `0/1`. For the explanatory comparison only,
outcome-only display credit is `+1` for a successful trajectory and `-1` for a
failed trajectory, assigned uniformly to every action. The artifact must label
this as a display transformation so it is not confused with the environment's
reward definition.

Outputs include canonical JSON for analysis and concise Markdown for the blog
draft. The Markdown should state that the trajectory is deterministically
selected by a predeclared illustrative rule and is not an estimate of
population frequency.

## Error handling

Evaluation fails closed on:

- model or tokenizer revision drift;
- insufficient or context-dependent option labels;
- nonstationary prompt inputs;
- duplicate, missing, or illegal actions;
- NaN, infinite, negative, or non-normalized probabilities;
- cache-schema or `RuleSet` mismatch;
- incomplete DAG cache coverage;
- Monte Carlo disagreement with numerical root value.

No failure may trigger uniform probabilities, action truncation, cache reuse,
or a different artifact-selection rule as a fallback.

## Testing strategy

Unit tests cover:

- state serialization and prompt determinism across processes;
- menu permutation determinism and ordering sensitivity;
- action/menu bijection, including duplicate values and actions sharing a
  child state;
- capacity for the maximum legal-action count;
- tokenizer-label validation using a controlled fake tokenizer;
- tokenizer decode round-trip and invalid-candidate filtering;
- explicit agreement between menu-label order and label-token-ID order;
- masked-softmax action mapping using synthetic logits;
- cache round-trip, resumability, canonical ordering, duplicate-state
  rejection, and every metadata invalidation key;
- numerical evaluator validation failures;
- hand-computed numerical value fixtures;
- Bellman and zero-mean-advantage invariants;
- deterministic trajectory selection and the no-match outcome.

One slow integration test loads the real model, scores a tiny state, and checks
that the returned policy is finite, complete, and normalized. The full worked
DAG and 20,000-rollout experiment are reproducible experiment commands rather
than default unit tests.

## Success criteria

Phase 2 is complete when:

1. Qwen3-0.6B produces a valid stationary forced-choice distribution for every
   nonterminal state in the worked DAG;
2. the cache is complete, reproducible, resumable, and tied to immutable model
   and prompt metadata;
3. numerical Bellman invariants pass;
4. the 20,000-rollout Monte Carlo check agrees with numerical `Vπ(root)`;
5. the predeclared artifact-selection procedure finds a qualifying trajectory
   without post-hoc substitution;
6. the JSON and Markdown artifacts can be regenerated from the cache without
   loading the model;
7. the complete run stays within the Mac-first resource gate or the approved
   sub-$100 cloud budget.
