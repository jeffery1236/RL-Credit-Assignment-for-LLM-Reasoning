# Causal Credit Assignment in LLM RL — Phase 1: Countdown Environment & Exact Oracle

## The central fact this repository can compute

Take the classic Countdown puzzle: numbers `(3, 7, 10, 25)`, target `245`.

From the starting position there are **18 legal moves**. Under perfect play, exactly **one** of them keeps a win possible:

```
10 + 25 = 35      <- the only move that preserves the win
```

The other **17** are already game-over the instant they're played — verified by exhaustive search, not sampling. From here, the win completes in one more step:

```
10 + 25 = 35   ->   7 × 35 = 245   ->   target reached, success
```

An outcome-only reward signal (the kind used by GRPO and most RL-for-reasoning setups) can only say "this trajectory won" or "this trajectory lost." It cannot express *"17 of your 18 opening options were already fatal, and only one of them wasn't."* This repository computes that fact exactly, for any state in this environment, and for any policy's actual behavior — not just under optimal play.

That's the whole point of Phase 1: build a reasoning environment small enough to compute **true causal credit** by brute force, so later phases can measure how badly practical credit-assignment methods get it wrong.

See `docs/project_plan.md` for the full 12-week research plan this Phase 1 supports.

---

## What's implemented (Phase 1 — no ML, no GPU, $0)

- A Countdown-style arithmetic environment with exact rules (below).
- Exact enumeration of the complete reasoning DAG for any starting problem, with transpositions merged.
- Exact `V*`, `Q*`, and `min_steps_to_success` (optimal-play facts, policy-independent).
- Exact `Vπ`, `Qπ`, `Aπ` for any stationary policy, computed in exact rational arithmetic (`fractions.Fraction` — no floating-point drift).
- Exact on-policy visit probabilities (how often a policy actually reaches each state / takes each action).
- Three baseline policies (uniform, greedy-distance, epsilon-greedy-distance) standing in for an imperfect model.
- Canonical, byte-reproducible JSON + DOT artifacts for visualizing one problem's full DAG.
- A 750-problem exploratory study measuring whether this environment's credit structure is actually interesting (see `research/findings_phase1.md`).
- Frozen, deduplicated, eligibility-checked train/validation/test splits (`data/v1/`).

Nothing here trains a model. That starts in Phase 2 — and there's a hard constraint on how that must work (see the warning below).

---

## ⚠️ Phase 2 warning — read before adding an LLM policy

The exact oracle in this repository (`oracle/evaluation.py`) rests on one assumption: **a policy's action probabilities depend only on the current state, never on trajectory history.** This is what allows two different paths that reach "the same" bag of numbers to be merged into one DAG node — merging is only mathematically valid if the policy would behave identically no matter which path got it there.

When you build the Phase 2 LLM policy:

- **The prompt must be rendered from the current state only** — no trajectory history, no carried-over chain-of-thought. If history is included, history becomes part of the policy's actual state, the merged-state DAG is no longer valid for that policy, and `Vπ`/`Qπ`/`Aπ` must **not** be computed over merged nodes for it.
- **The randomized action-menu ordering** (needed so the model can't learn "option B is usually right") must be seeded from a **stable hash of the canonical state** — e.g. `hashlib.blake2b` of its canonical representation — **never** Python's built-in `hash()`, which is randomized per process and would make the same state produce a different menu on every run.
- **Exact evaluation requires forced-choice, single-token logprobs.** If Phase 2 instead samples free-form chain-of-thought before the model commits to a choice, `π(a|s)` becomes the *marginal* over CoT samples — still a valid stationary policy in principle, but now something you must *estimate* (Monte Carlo), not read exactly. Keep the two evaluation modes clearly labeled in any downstream analysis.

---

## Two LLM evaluation modes

`experiments.evaluate_allowed_action_llm_policy` is the exact-credit mode. It
shows only the current state, requires one allowed-action label, and keeps
thinking disabled so the merged-state DAG remains valid.

`experiments.evaluate_countdown_capability` is a separate literature-style
model check. It asks for a complete expression using every number once,
samples 32 responses per problem at temperature `0.6`, and reports accuracy,
Pass@32, All-correct@32, format rate, and valid-equation rate. Its free-form
outputs must never be passed to the exact credit oracle.

Example:

```bash
uv run python -m experiments.evaluate_countdown_capability \
  --model Qwen/Qwen3.5-4B \
  --output artifacts/phase2/capability/qwen3.5-4b \
  --samples-per-problem 32 \
  --temperature 0.6 \
  --max-new-tokens 1024 \
  --generation-batch-size 1 \
  --device mps
```

For the complete 12-problem, 32-sample evaluation, use the prepared Cloud Run
Job with one NVIDIA L4:

```bash
brew install --cask google-cloud-sdk
gcloud init
gcloud auth login

PROJECT_ID=your-billed-project \
  scripts/cloud_run_countdown_capability.sh deploy

PROJECT_ID=your-billed-project \
  scripts/cloud_run_countdown_capability.sh execute

PROJECT_ID=your-billed-project \
  scripts/cloud_run_countdown_capability.sh download
```

The job mounts Cloud Storage at `/results`, checkpoints after each completed
problem, and can resume across Cloud Run retries.
The configured non-zonal L4, 8-vCPU, 32-GiB instance costs approximately
`$1.42` per active hour in `us-central1`. The project needs quota for one
non-zonal Cloud Run L4 GPU.

## Generic RunPod experiments

`scripts/runpod_experiment.sh` builds the CUDA image, launches an arbitrary
repository experiment, follows its completion, downloads the complete output
directory, and terminates successful Pods. Failed runs download partial
artifacts and leave the Pod running for inspection.

Configure `runpodctl`, then build the CUDA image. The build pushes to this
project's Artifact Registry repository by default; override `IMAGE` only to
publish somewhere else.

```bash
runpodctl doctor

scripts/runpod_experiment.sh build
```

The general form is:

```bash
scripts/runpod_experiment.sh run RUN_NAME LOCAL_OUTPUT -- \
  python -m experiments.EXPERIMENT [arguments writing to /workspace/results]
```

Named runs can also be managed separately:

```bash
scripts/runpod_experiment.sh submit RUN_NAME LOCAL_OUTPUT -- COMMAND [ARG...]
scripts/runpod_experiment.sh status RUN_NAME
scripts/runpod_experiment.sh logs RUN_NAME
scripts/runpod_experiment.sh wait RUN_NAME
scripts/runpod_experiment.sh download RUN_NAME
scripts/runpod_experiment.sh terminate RUN_NAME
```

For staged experiments, set `NETWORK_VOLUME_ID` on every command so
`/workspace/results` persists between Pods. For example, run policy-bridge
Stage A:

```bash
export NETWORK_VOLUME_ID=YOUR_RUNPOD_NETWORK_VOLUME_ID

scripts/runpod_experiment.sh run policy-bridge-stage-a \
  artifacts/phase2/policy-bridge/qwen3.5-4b -- \
  python -m experiments.evaluate_policy_bridge stage-a \
  --output /workspace/results \
  --model Qwen/Qwen3.5-4B \
  --device cuda
```

Inspect the downloaded Stage A gate:

```bash
python -c '
import json
from pathlib import Path
p = Path("artifacts/phase2/policy-bridge/qwen3.5-4b/policy_bridge_stage_a.json")
print(json.loads(p.read_text())["aggregate"]["gate"])
'
```

Only after that gate passes, reuse the same network volume for Stage B:

```bash
scripts/runpod_experiment.sh run policy-bridge-stage-b \
  artifacts/phase2/policy-bridge/qwen3.5-4b -- \
  python -m experiments.evaluate_policy_bridge stage-b \
  --output /workspace/results \
  --model Qwen/Qwen3.5-4B \
  --device cuda
```

The previous capability-specific launcher remains as a compatibility wrapper
around this generic helper.

---

## Repository layout

```
env/            Problem/State/Action/RuleSet, legal-move rules, transitions, CountdownEnv,
                candidate generation (env/countdown.py, env/generator.py)
oracle/         DAG enumeration, exact V*/Q*/min_steps_to_success, the Policy protocol,
                ExactPolicyEvaluator (Vπ/Qπ/Aπ), visit-probability computation
                (oracle/evaluation.py)
policies/       Uniform, greedy-distance, epsilon-greedy-distance baselines
                (policies/baselines.py)
analysis/       Credit-structure profiling (analysis/profiles.py) and canonical
                JSON/DOT/CSV artifact writers (analysis/artifacts.py)
experiments/    CLI entry points: render_dag, profile_environment, freeze_dataset
tests/          Unit, property-based (Hypothesis), and independent brute-force
                cross-check tests
data/v1/        Frozen train/validation/test splits + manifest
artifacts/      Generated DAG visualizations and the exploratory study's output
research/       experiment_log.md, findings_phase1.md
```

---

## The V1 game rules

- **Operators:** `+`, `−`, `×` only. No division.
- **All intermediate values are strictly positive integers.** `a − b` is legal only when `a > b`.
- **Each number is consumed at most once** per move — an action removes two values from the bag and inserts one result.
- **Immediate success:** the episode ends the moment the target value appears anywhere in the bag. You do not have to use up the remaining numbers.

A **state** is `(target, sorted tuple of values)` — two bags that are permutations of each other are the *same* state. An **action** is identified by `(operator, operand values)`, not by list position — this is why `(3, 3, 7)` offers `3 + 7` exactly once (not twice), while `(2, 2)` offers **both** `2 + 2 = 4` and `2 × 2 = 4` as genuinely separate moves that happen to land on the same next state.

---

## Quickstart

```bash
uv sync --locked
uv run pytest -q                 # full suite (fast; ~230 tests, ~1.3s)
uv run pytest -q -m slow         # + the 20k-rollout Monte-Carlo-vs-exact sanity check
```

Render the DAG for any problem:

```bash
uv run python -m experiments.render_dag \
    --numbers 3 7 10 25 --target 245 \
    --output artifacts/phase1/example --max-nodes 100000
dot -Tsvg artifacts/phase1/example.dot -o artifacts/phase1/example.svg   # optional, needs Graphviz
```

Re-run the exploratory study (Step 7 — see `research/findings_phase1.md` for what it found):

```bash
uv run python -m experiments.profile_environment --output artifacts/phase1/exploratory --per-stratum 250
```

Regenerate the frozen dataset splits (only after reading the go/no-go decision in `research/findings_phase1.md`):

```bash
uv run python -m experiments.freeze_dataset --output data/v1
```

Determinism check — everything above should regenerate byte-identically:

```bash
git status --porcelain    # should be empty after re-running any of the above
```

---

## Public API by module

### `env.countdown` — the game itself

```python
from env.countdown import Problem, allowed_actions, transition, is_terminal, reward

problem = Problem(values=(3, 7, 10, 25), target=245)
state = problem.to_state()

for action in allowed_actions(state):
    print(action.render())          # "10 + 25 = 35"

next_state = transition(state, allowed_actions(state)[0])
```

### `oracle.evaluation` — the exact oracle

```python
from oracle.evaluation import (
    build_reasoning_dag, compute_v_star, compute_q_star,
    compute_min_steps_to_success, ExactPolicyEvaluator, compute_visit_probabilities,
)
from policies.baselines import UniformPolicy

dag = build_reasoning_dag(state)
v_star = compute_v_star(dag)                      # {state: 0 or 1}
q_star = compute_q_star(dag, v_star)               # {state: {action: 0 or 1}}

evaluator = ExactPolicyEvaluator(UniformPolicy(), dag)
evaluator.v_pi(state)                              # exact Fraction, e.g. Fraction(1, 3)
evaluator.a_pi(state, allowed_actions(state)[0])   # exact Aπ(s, a)

visits = compute_visit_probabilities(dag, evaluator)
visits.probability_of_reaching_state[state]        # how often the policy is even here
```

### `analysis.profiles` — credit-structure summaries

```python
from analysis.profiles import compute_optimal_action_profile, compute_policy_credit_profile

# Policy-INDEPENDENT: was this state already a minefield under perfect play?
compute_optimal_action_profile(state, v_star, q_star, min_steps).fraction_of_moves_that_ruin_solution

# Policy-DEPENDENT: how does this specific policy's advantage look here?
compute_policy_credit_profile(state, evaluator).mean_abs_advantage
```

These two are **structurally separate types on purpose** — never conflate "this move was optimal-play-fatal" with "this move looked bad to a specific weak policy." See the module docstring in `analysis/profiles.py` for why.

### `analysis.artifacts` — canonical JSON/DOT/CSV output

```python
from analysis.artifacts import build_dag_artifact, to_canonical_json, to_dot

artifact = build_dag_artifact(state, policies={"uniform": UniformPolicy()})
to_canonical_json(artifact)   # byte-reproducible JSON
to_dot(artifact)              # Graphviz DOT text
```

`build_dag_artifact` is for **one-off, illustrative DAGs only** — nodes carry aggregate stats, edges carry the exact per-action `Q*`/probability/advantage (never both, see `docs/project_plan.md` §31.1 for why). Bulk analysis across many problems goes through `analysis.profiles.compute_problem_profile` instead, which is what `experiments/profile_environment.py` uses.

---

## Testing philosophy

The single highest-value test in this repository is `tests/reference_solver.py` — a deliberately naive, unmemoized, brute-force solver sharing no code with `oracle/evaluation.py`. Every oracle computation is cross-checked against it on small state spaces. Beyond that:

- Hand-computed fixtures with values you can verify on paper (e.g. `(1,2) → 3` under uniform policy gives exactly `Vπ = 1/3`).
- Hypothesis property-based tests: `Σ_a π(a|s)·Aπ(s,a) == 0` exactly, `Vπ(s) ≤ V*(s)`, `Qπ(s,a) == Vπ(child(s,a))`.
- A 20k-rollout Monte-Carlo-vs-exact sanity check (marked `slow`), pulled forward from what the project plan calls Phase 2's "critical sanity test."
- Committed JSON/DOT snapshot fixtures (`tests/fixtures/`) to catch any accidental format drift.

---

## Status

Phase 1 is complete. See `research/findings_phase1.md` for the go/no-go interpretation and `docs/project_plan.md` for what comes next (Phase 2: putting an LLM inside the environment).
