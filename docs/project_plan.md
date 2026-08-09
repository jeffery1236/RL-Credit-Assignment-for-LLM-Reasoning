# Project Plan: Causal Credit Assignment in LLM Reinforcement Learning

## Working title

**Does LLM RL Put Credit Where It Belongs? Measuring Causal Credit Assignment in a Controlled Reasoning Environment**

### One-sentence research question

> When an LLM succeeds or fails on a multi-step reasoning task, how accurately do practical RL credit-assignment methods identify the intermediate decisions that actually caused the outcome—and does more accurate credit assignment lead to better learning?

---

# 1. Why this project exists

Outcome-based RL methods such as GRPO assign a trajectory-level signal despite the fact that a multi-step trajectory can contain a mixture of good and bad decisions.

For example:

```text
State 0
  ↓
good decision
  ↓
State 1
  ↓
good decision
  ↓
State 2
  ↓
BAD DECISION
  ↓
failure
```

Outcome-only training sees:

```text
decision 1   negative
decision 2   negative
decision 3   negative
```

But intuitively the causal credit should look more like:

```text
decision 1    +
decision 2    +
decision 3   ---
```

Modern work attempts to solve this with resets, counterfactual interventions, learned value estimates, turn-level rewards, and other methods. Reset-based methods, for example, explicitly resample continuations from intermediate states to improve localization of credit. Counterfactual token-weighting approaches likewise attempt to identify causally important portions of reasoning traces, although existing work notes that interventions such as span masking are only approximations to true counterfactuals.

The measurement problem is:

> **How do we know whether these estimated credit signals correspond to the actual causal importance of decisions?**

A recent survey of LLM RL credit assignment explicitly proposes controlled bifurcation tasks and benchmark protocols as a way to study this problem.

This project builds such a controlled environment.

---

# 2. Core idea

Instead of starting with free-form mathematical Chain-of-Thought, use an environment where:

1. the model still performs multi-step symbolic reasoning;
2. actions are discrete and interpretable;
3. transitions are deterministic;
4. the entire future state graph can be enumerated;
5. success is objectively verifiable;
6. the true value of taking an action can be computed.

That gives us something normally unavailable in LLM RL:

# A ground-truth credit oracle.

The experiment becomes:

```text
                       trajectory
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      estimated credit             exact credit
      from RL method                from environment
             │                           │
             └─────────────┬─────────────┘
                           ▼
                       compare
                           │
                           ▼
             Does better credit produce
                  better RL training?
```

---

# 3. Environment: Countdown-style arithmetic reasoning

## State

The model receives:

* a multiset of numbers;
* a target;
* the currently legal arithmetic operations.

Example:

```text
Numbers: [3, 7, 10, 25]
Target: 245
```

## Action

At every step the model chooses two available numbers and applies an operation.

For example:

```text
A. 25 - 10 = 15
B. 25 × 10 = 250
C. 7 - 3 = 4
D. 7 × 3 = 21
...
```

Suppose it chooses:

```text
B. 25 × 10 = 250
```

The new state becomes:

```text
Numbers: [3, 7, 250]
Target: 245
```

Then another decision is made.

Each operation reduces the number of available values by one, so the trajectory forms a finite DAG rather than an unbounded reasoning process.

Initial operator set:

```text
+
-
×
```

Stretch:

```text
÷  only when exact integer division is valid
```

---

# 4. Why this environment is useful

Consider:

```text
                 State s
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
       a₁          a₂          a₃
        │           │           │
        ▼           ▼           ▼
    solvable     dead end    solvable
```

If the model takes `a₂`, we objectively know that decision destroyed the possibility of success.

No LLM judge is necessary.

More subtly, we can measure how good an action is under the model's actual future policy rather than merely whether some perfect continuation exists.

---

# 5. The ground-truth credit oracle

For policy (\pi), define terminal reward:

[
R(s)=
\begin{cases}
1 & \text{target reached}\
0 & \text{otherwise}
\end{cases}
]

Because the environment is finite and deterministic:

[
Q^\pi(s,a)=V^\pi(T(s,a))
]

where (T(s,a)) is the state reached after action (a).

And:

[
V^\pi(s)
========

\sum_a \pi(a|s)Q^\pi(s,a)
]

Therefore the exact policy advantage is:

[
A^\pi(s,a)
==========

Q^\pi(s,a)-V^\pi(s)
]

This is the quantity we care about.

Intuitively:

```text
A > 0

"Taking this action made eventual success
more likely than what the model would
normally have done here."


A < 0

"Taking this action made eventual success
less likely."


A ≈ 0

"This action didn't matter much."
```

Because the state graph is small, recursively evaluating the model's action probabilities allows us to calculate this quantity exactly or arbitrarily closely.

---

# 6. Two different notions of ground truth

Track both.

## Optimal credit

[
Q^*(s,a)
]

asks:

> Did this action preserve the possibility of solving the problem under an optimal future policy?

Useful for detecting catastrophic decisions.

Example:

```text
before action: solution exists
after action:  no solution exists

→ objectively destructive action
```

## Policy-relative credit

[
A^\pi(s,a)
]

asks:

> Given how this particular model behaves afterward, how much did taking this action change its chance of succeeding?

This should be the main quantity for studying RL.

---

# 7. Research questions

## RQ1 — Measurement

**How accurately do practical credit-assignment estimators recover true (A^\pi(s,a))?**

---

## RQ2 — Sample complexity

**How many counterfactual rollouts are needed to estimate useful credit?**

Compare:

```text
K = 1
K = 2
K = 4
K = 8
K = 16
```

counterfactual suffix samples.

---

## RQ3 — Horizon

**Does credit-assignment accuracy deteriorate as reasoning horizon increases?**

Compare problems beginning with:

```text
4 numbers → ≤3 decisions
5 numbers → ≤4 decisions
6 numbers → ≤5 decisions
```

---

## RQ4 — The important question

**Does more causally accurate credit actually lead to better RL training?**

This is the central research question.

The surprising result could easily be:

```text
credit fidelity

Method A > Method B

but

RL performance

Method B > Method A
```

If that happens, the obvious next question is:

> Why is the worse credit estimator producing the better policy?

That could become the most interesting part of the project.

---

# 8. Credit methods to compare

Do not implement ten methods.

Start with three.

## Method 0 — Uniform outcome credit

The baseline.

Successful trajectory:

```text
a₁   +
a₂   +
a₃   +
a₄   +
```

Failed trajectory:

```text
a₁   -
a₂   -
a₃   -
a₄   -
```

This approximates the coarse credit structure underlying trajectory-level RL.

---

## Method 1 — Monte Carlo reset credit

At intermediate state (s_t):

```text
                  s_t
                   │
        chosen action a_t
                   │
                   ▼
                s_t+1
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
     rollout     rollout    rollout
        │          │          │
        ▼          ▼          ▼
        1          0          1
```

Estimate:

[
\hat Q(s_t,a_t)
===============

\frac{1}{K}\sum_{k=1}^{K}R_k
]

Separately sample from (s_t) to estimate:

[
\hat V(s_t)
]

Then:

[
\hat A_t
========

\hat Q(s_t,a_t)-\hat V(s_t)
]

This is closely related to the reset/counterfactual intuition appearing in recent reasoning-RL credit-assignment work.

---

## Method 2 — Oracle credit

Use exact:

[
A^\pi(s,a)
]

This method is deliberately impossible in general real-world tasks.

It provides the **upper bound**:

> If we had perfect credit assignment, how much would it actually help?

This distinction is crucial.

If oracle credit barely improves training over uniform credit, then developing increasingly fancy credit estimators may simply not matter much in this environment.

---

# 9. Metrics

## Credit fidelity

### Rank correlation

Spearman correlation between:

[
\hat A(s,a)
]

and:

[
A^\pi(s,a)
]

---

### Sign accuracy

How often do we correctly determine whether an action was:

```text
helpful      A > 0
neutral      A ≈ 0
harmful      A < 0
```

---

### Pivotal-action recall

Define a catastrophic decision:

```text
solution exists before action
             +
no solution exists afterward
```

Measure whether the credit estimator assigns these actions strongly negative credit.

---

### Top-k attribution

Among actions in a trajectory, does the estimator correctly identify the most consequential action?

---

## RL performance

Track:

* test success rate;
* sample efficiency;
* environment interactions;
* training stability;
* entropy;
* KL from reference policy;
* performance by horizon;
* performance on unseen targets/numbers.

The main plot eventually becomes something like:

```text
           downstream RL improvement
                    ▲
                    │
                    │       ●
                    │   ●
                    │
                    │ ●
                    └────────────────────►
                         credit fidelity
```

Then ask:

> Is this relationship actually strong?

---

# 10. Model

Start with:

**Qwen3-0.6B**

It is small enough for inexpensive experimentation while retaining reasoning capability, and an official 0.6B checkpoint is publicly available.

Do **not** begin with a 7B model.

The purpose of the first month is understanding the phenomenon, not maximizing benchmark accuracy.

Only scale to roughly a 1–3B model if V1 produces a result worth testing for robustness.

---

# 11. Experimental dataset

Programmatically generate tasks.

Initial split:

```text
Train          5,000
Validation     500
Test           1,000
```

But compute oracle statistics on a smaller fixed diagnostic subset initially:

```text
Oracle diagnostic set: 200–300 problems
```

Stratify by:

```text
horizon
branching factor
number magnitude
number of valid solution paths
solution uniqueness
```

Avoid datasets where nearly every problem is either trivially solvable or essentially impossible.

Target initial model success should ideally sit somewhere roughly in the middle rather than at 0% or 100%.

---

# 12. Important environment controls

## Action ordering

Randomize action-menu order.

Otherwise the policy could learn:

```text
"option B tends to be correct"
```

rather than reasoning.

---

## Train/test generation

Generate held-out test instances using different random seeds and targets.

---

## Memorization

The state space should be sufficiently large that memorizing entire task trajectories is impractical.

---

## Difficulty

Create difficulty primarily through:

```text
longer horizon
+
higher branching
+
fewer viable paths
```

rather than obscure arithmetic.

The project is about **credit assignment**, not arithmetic capability.

---

# 13. Phase 0 — Literature + research spec

## Week 1

Time: **4–6 hours**

No GPU spending.

Read deeply:

1. one GRPO/DAPO reference;
2. the recent credit-assignment survey;
3. reset-based credit assignment;
4. one counterfactual credit paper;
5. one agentic credit-assignment paper.

The current literature already spans reset-based reasoning credit, counterfactual interventions, and turn-level agentic credit estimators such as TRACE, so this week is explicitly intended to prevent re-solving an already-answered question.

Produce:

```text
research_spec.md
```

containing:

* exact RQs;
* closest prior work;
* what those papers measure;
* what they do not measure;
* your novelty claim;
* conditions that would falsify your hypotheses.

### Gate

Do **not** write substantial training infrastructure until this document exists.

---

# 14. Phase 1 — Build the environment

## Weeks 2–3

Time: **8–12 hours**

Budget:

**$0–20**

Build:

```text
env/
├── state.py
├── actions.py
├── transition.py
├── generator.py
├── solver.py
└── oracle.py
```

Requirements:

```text
reset(problem)
legal_actions(state)
step(action)
is_terminal(state)
reward(state)
enumerate_descendants(state)
```

Then implement dynamic programming for:

```text
V*
Q*
```

Initially with a hand-coded/random policy.

### Unit tests

Create small states where values can be calculated manually.

Example:

```text
State
  ├── A → guaranteed success
  ├── B → 50% future success
  └── C → guaranteed failure
```

Verify the oracle exactly.

### Deliverable

A visualization of one complete reasoning DAG:

```text
             s₀
        /     |      \
      a₁     a₂      a₃
      /       |       \
    ...      ...      ...
                ↓
          success/failure
```

---

# 15. Phase 2 — Put an LLM inside the environment

## Week 4

Time: **5–8 hours**

Use the LLM as:

[
\pi(a|s)
]

Present valid actions as discrete options.

Obtain normalized probabilities over legal actions.

Now compute exact:

[
V^\pi,\ Q^\pi,\ A^\pi
]

through the state DAG.

### Critical sanity test

Compare the exact dynamic-programming value against a very large Monte Carlo estimate.

For example:

```text
exact Vπ(s)      = 0.437

MC estimate
with many runs   ≈ 0.44
```

They should agree closely.

If they do not, stop and fix the oracle.

### First meaningful artifact

Produce:

```text
trajectory
observed outcome
true credit at each step
```

Example:

```text
Step              Outcome credit    True Aπ

25 × 10 = 250          -1            +0.24
250 - 7 = 243          -1            +0.31
243 + 3 = 246          -1            -0.81
```

The trajectory failed, but only the final decision deserved strong blame.

This is the first “aha” visualization of the project.

---

# 16. Phase 3 — Benchmark credit estimators

## Weeks 5–6

Time: **10–15 hours**

Cumulative budget cap:

**$50**

No RL training yet.

For approximately 200–500 diagnostic problems, collect trajectories.

Compare:

```text
Uniform outcome
MC-reset K=1
MC-reset K=2
MC-reset K=4
MC-reset K=8
MC-reset K=16
Oracle
```

Run across:

```text
4-number tasks
5-number tasks
6-number tasks
```

Produce plots:

### Plot 1

```text
credit correlation
      ▲
      │          ______
      │       __/
      │     _/
      │ ___/
      └────────────────►
             K
```

### Plot 2

```text
credit fidelity
      ▲
      │
      │ ● horizon 3
      │
      │   ● horizon 4
      │
      │       ● horizon 5
      └────────────────►
             horizon
```

### Plot 3

Pivotal-action detection accuracy.

### Plot 4

Estimator quality versus number of additional model rollouts.

This introduces the practical frontier:

```text
accuracy of credit
        versus
cost of obtaining credit
```

---

# 17. First go/no-go decision

At the end of Week 6, write:

# `findings_v1.md`

Answer:

1. Do the credit estimators meaningfully differ?
2. Does reset sample count matter?
3. Does horizon make credit harder?
4. What kinds of actions are misattributed?
5. Did anything happen that you did not expect?

### Continue to RL training if:

There is a meaningful spread between estimators and the environment produces nontrivial credit structure.

### Rework the environment if:

```text
almost every action has A ≈ 0

or

nearly all trajectories have one obvious good action

or

the base policy almost always succeeds/fails
```

### Stop the project if:

The benchmark reveals essentially nothing beyond trivial Monte Carlo variance and you cannot identify a sharper question.

Stopping here is acceptable.

You still have a clean credit-assignment benchmark implementation and learned a substantial amount for <$50.

---

# 18. Phase 4 — RL baseline

## Weeks 7–8

Time: **10–15 hours**

Cumulative compute budget:

**≤$100**

Implement standard outcome-based training first.

Use GRPO or a GRPO-like grouped policy-gradient setup.

Current Hugging Face TRL tooling includes `GRPOTrainer` and support for environment-owned rewards/multi-environment agentic RL, so use existing infrastructure wherever possible instead of turning this project into a training-framework rewrite.

Goal:

```text
initial success      25–40%
        ↓ RL
final success        meaningfully higher
```

Exact number does not matter.

You only need a baseline where learning reliably occurs.

Run at least:

```text
3 seeds
```

before comparing novel credit methods.

---

# 19. Phase 5 — Does better credit produce better learning?

## Weeks 9–10

Time: **10–15 hours**

Cumulative budget cap:

**$200**

Train three conditions.

## A. Uniform

Trajectory outcome credit.

---

## B. MC-reset

Use the cheapest value of `K` that had reasonably good credit fidelity during Phase 3.

Probably something like:

```text
K = 4 or 8
```

depending on results.

---

## C. Oracle

Train using exact per-decision advantage:

[
A^\pi(s_t,a_t)
]

This is the crucial experimental condition.

The comparison becomes:

```text
Uniform
    │
    │ increasingly accurate credit
    ▼
MC reset
    │
    ▼
Oracle
```

Then measure:

```text
                    Credit accuracy

Uniform        low
MC reset       medium
Oracle         perfect

                    ↓

                 RL result

???
```

---

# 20. The four possible results

## Result A

```text
Oracle > MC > Uniform
```

and RL performance tracks credit fidelity.

Interpretation:

> Better causal credit directly improves learning in this environment.

Then investigate how cheaply oracle-like credit can be approximated.

---

## Result B

```text
Oracle ≈ MC > Uniform
```

Interpretation:

> Approximate credit is sufficient; highly precise attribution has diminishing returns.

Very practical result.

---

## Result C

```text
Oracle ≈ Uniform
```

This is interesting.

It suggests:

> Credit assignment wasn't the actual bottleneck despite the apparent theoretical problem.

Then ask what dominates:

* exploration?
* model capacity?
* policy optimization?
* trajectory diversity?

---

## Result D

```text
MC > Oracle
```

This is potentially the most interesting.

Then:

> A noisier causal-credit estimator improves learning more than perfect causal credit.

Possible mechanisms to test:

```text
regularization
exploration
gradient smoothing
optimization variance
entropy preservation
```

This could become the flagship result.

---

# 21. Phase 6 — Mechanism experiment

## Weeks 11–12

Do **not** decide this experiment now.

Select it based on the first ten weeks.

Examples:

### If horizon matters

Sweep:

```text
3 → 4 → 5 → 6 steps
```

and study where each estimator breaks.

### If MC beats oracle

Inject controlled noise into oracle credit:

```text
Aoracle + ε
```

and ask whether some noise level improves training.

### If only pivotal actions matter

Compare:

```text
dense exact credit

vs

only update top causal decisions
```

### If oracle strongly wins

Ask how sparse the oracle can become before losing its advantage.

This is where the actual research question should sharpen.

---

# 22. Weekly time commitment

Target:

**5 hours/week average**

Example:

```text
Saturday
2.5h experiment / implementation

Sunday
1.5h analysis

Weeknight
1h reading / research log
```

For twelve weeks:

[
\approx 60 \text{ hours}
]

Do not turn this into a second full-time job.

---

# 23. Compute strategy

## First gate

Spend:

**<$50**

before RL training.

The majority of early work should be:

```text
environment
oracle
analysis
literature
```

not GPU burn.

---

## Second gate

Spend up to:

**$100 total**

only once ordinary RL reliably learns.

---

## Third gate

Spend up to:

**$200 total**

only after you have evidence that the credit estimators differ meaningfully.

---

## Expansion

Only consider:

**$200–1,000**

if you already have a specific finding worth validating on:

* larger models;
* longer horizons;
* a second environment;
* agentic tasks.

Never spend $800 hoping an interesting result appears.

---

# 24. Research log

Create:

```text
research/
├── questions.md
├── literature.md
├── hypotheses.md
├── experiment_log.md
├── findings_v1.md
└── failures.md
```

Every experiment gets:

```text
Experiment ID:
Question:
Hypothesis:
Independent variable:
Dependent variable:
Controls:
Expected result:
Actual result:
Interpretation:
Next experiment:
```

Example:

```text
E017

Question:
Does MC credit fidelity deteriorate with horizon?

Hypothesis:
K=4 becomes increasingly biased/noisy as horizon grows.

Independent:
horizon ∈ {3,4,5}

Controlled:
same model
same task generation distribution
same K
same sampling temperature

Metric:
Spearman(Â, A*)

Result:
...

Interpretation:
...
```

This discipline is arguably more important than the exact RL framework.

---

# 25. Repository structure

```text
causal-credit-rl/
│
├── README.md
│
├── pyproject.toml
│
│
├── env/
│   ├── state.py
│   ├── actions.py
│   ├── transition.py
│   ├── generator.py
│   └── solver.py
│
├── oracle/
│   ├── optimal_value.py
│   └── policy_value.py
│
├── policies/
│   ├── llm_policy.py
│   └── random_policy.py
│
├── credit/
│   ├── uniform.py
│   ├── monte_carlo.py
│   └── oracle.py
│
├── training/
│   ├── grpo.py
│   └── credit_pg.py
│
├── experiments/
│   ├── configs/
│   ├── run_credit_eval.py
│   └── run_training.py
│
├── analysis/
│   ├── credit_fidelity.ipynb
│   ├── horizon.ipynb
│   └── training.ipynb
│
├── tests/
│
└── research/
    ├── literature.md
    ├── hypotheses.md
    ├── experiment_log.md
    └── failures.md
```

---

# 26. What NOT to build

Do not spend time building:

* custom distributed RL infrastructure;
* fancy dashboards;
* web UI;
* generic agent framework;
* huge environment abstraction layers;
* sophisticated experiment management;
* your own inference server.

Use existing infrastructure.

The intellectual artifact should be:

```text
Question
   ↓
Measurement
   ↓
Experiment
   ↓
Finding
```

not:

```text
Look how much software I wrote.
```

---

# 27. What would make the project successful?

There are three success levels.

## Level 1 — Learning artifact

You accurately reproduce credit-assignment concepts and build the oracle benchmark.

Result:

```text
good GitHub project
+
deep personal understanding of GRPO/credit assignment
```

Still useful.

---

## Level 2 — Strong portfolio artifact

You establish something like:

> Reset-based credit estimates become substantially less reliable with horizon, but only modest credit fidelity is required to capture most downstream training gains.

Now you have:

```text
clear question
+
controlled environment
+
empirical result
+
RL training
+
analysis
```

Excellent Research Engineer interview material.

---

## Level 3 — Research result

You discover something counterintuitive such as:

> Perfect causal credit underperforms deliberately noisy credit because the noise preserves exploration.

or:

> Training performance is largely insensitive to credit fidelity until attribution errors affect a small class of irreversible bifurcation decisions.

Now there may be enough for:

* serious technical report;
* workshop submission;
* collaboration;
* follow-up research.

Do not expect Level 3.

Design the project so Level 2 is already a win.

---

# 28. Extension: long-horizon tool agents

Only pursue this after V1.

Replace arithmetic reasoning with:

```text
state
 ↓
tool action
 ↓
environment transition
 ↓
observation
 ↓
tool action
 ↓
...
 ↓
task result
```

Construct controlled tasks containing:

* useful information-gathering actions;
* distractor actions;
* irreversible mistakes;
* redundant actions;
* delayed consequences.

Example:

```text
                     START
                       │
              ┌────────┴────────┐
              ▼                 ▼
         inspect DB         modify state
              │                 │
        obtain clue         premature change
              │                 │
              ▼                 ▼
        inspect logs          dead end
              │
              ▼
          correct fix
```

Now study:

> Can credit-assignment methods correctly reward an early information-gathering action whose value only becomes apparent ten steps later?

This directly connects V1 to long-horizon agent RL.

Modern agentic RL is already pushing credit assignment toward turn-level and counterfactual approaches because long trajectories make a single outcome reward increasingly uninformative.

---

# 29. Final public artifacts

If results are meaningful, produce exactly three things.

## 1. GitHub repository

README starts with the central figure, not installation instructions.

Example:

```text
Does better causal credit make RL better?

               Credit fidelity    Final reward

Uniform             .12               .47
MC Reset             .71               .58
Oracle              1.00               .59
```

Then explain the question.

---

## 2. Technical report

Target:

**6–10 pages**

Structure:

```text
1. Problem
2. Related work
3. Controlled environment
4. Exact credit oracle
5. Credit estimators
6. Experiments
7. Results
8. Mechanistic analysis
9. Limitations
10. Agentic extension
```

---

## 3. Interview-ready whiteboard story

You should eventually be able to explain:

```text
Outcome reward has a causal attribution problem.

                ↓

Existing work proposes finer credit.

                ↓

But measuring whether that credit is correct is hard.

                ↓

I built an environment where Qπ is computable.

                ↓

I benchmarked credit estimators against true advantage.

                ↓

Then I tested whether causal fidelity predicts
actual RL learning.

                ↓

Here's the surprising thing I found...
```

That final sentence is the whole point of the project.

---

# 30. First 7 days

Do only these things:

### Task 1

Read the recent credit-assignment survey and create a table of the 8–10 methods most relevant to reasoning/agentic RL.

### Task 2

Write a one-page research specification answering:

> What exactly has not already been measured?

### Task 3

Implement the Countdown environment with a non-LLM random policy.

### Task 4

Implement exhaustive search for:

[
V^*(s)
]

### Task 5

Generate one DAG and manually inspect every state.

Do **not** touch GRPO yet.

The first milestone is:

> **“I can point at any action in my toy reasoning environment and mathematically say how much that action changed the probability of eventual success.”**

Once that is true, the research apparatus exists.

Everything after that is an experiment.
