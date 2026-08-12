# Findings — Phase 1 (E002 exploratory study go/no-go)

## E002 setup (as actually run)

- 750 candidates: 250 each of 4, 5, and 6 starting numbers.
- Value pool: `1–10` (small) plus `25, 50, 75, 100` (large) — the classic Countdown pool.
- Target range: `100–999` (same for all three strata).
- `generator_version = "v1.0-exploratory"`, one seed namespace per stratum (`exploratory-4`, `exploratory-5`, `exploratory-6`).
- Three baseline policies evaluated through the identical `ExactPolicyEvaluator`: uniform, greedy-distance, epsilon-greedy-distance (`ε = 1/10`).
- `max_nodes = 100,000` exactness guard.
- Runtime: 24 minutes (dominated by the n=6 stratum — full DAG + 3-policy evaluation averages ~5s/problem at that size).

The six diagnostic triggers below were declared **before** this run, in `experiments/profile_environment.py`'s docstring and the Phase 1 implementation plan, specifically so they couldn't be adjusted after seeing the numbers. They are triggers for inspection, not pass/fail gates — the actual go/no-go judgment is in the Interpretation and Decision sections.

## Predeclared diagnostic triggers — results

### Scientific triggers

| # | Condition | Actual (pooled) | Triggered? |
|---|---|---|---|
| 1 | Fewer than 10% or more than 90% of solvable non-terminal states are decision-sensitive | **99.99%** | **Yes** (above 90%) |
| 2 | Irreversibly-destructive actions below 1% or above 90% of legal actions from solvable states | **84.1%** | No (inside band, near its top) |
| 3 | Fewer than 10% of profiled actions have non-zero Aπ under any baseline (unweighted, all reachable pairs) | uniform 1.71%, greedy 0.56%, ε-greedy 1.71% | **Yes**, all three |
| 4 | All three policies' advantage distributions effectively identical (mean\|Aπ\| diff < 0.02 AND negative-frequency diff < 5pp), unweighted | all three pairs: mean\|Aπ\| diff ≤ 0.0046, neg-freq diff ≤ 0.98pp | **Yes** |

### Computational-degeneracy triggers

| # | Condition | Actual | Triggered? |
|---|---|---|---|
| 5 | More than 5% of candidates hit the 100,000-node exactness guard | 0.0% (0/750) | No |
| 6 | An entire number-count stratum shows only one observed minimum solution depth | n=4: {2,3}, n=5: {2,3,4}, n=6: {0,1,2,3,4,5} | No |

Four of six triggers fired or sat at a notable extreme. Each is interpreted below rather than treated as a stop signal on its own — per the plan, the point of these bands is to force a look, not to be satisfied.

## Interpretation

### Triggers 1 & 2 — fragility is real and close to universal (expected, not alarming)

Among solvable non-terminal states, **99.99%** have at least one move that would irreversibly destroy the win, and on average **84.1%** of a solvable state's legal moves are exactly that kind of landmine. This matches E001's manual finding on the single worked example — 17 of 18 root moves in `(3,7,10,25)→245` were instantly fatal — and confirms it's not a one-off: this environment is *pervasively* unforgiving once you're in a position that can still be won.

One nuance worth recording: because decision-sensitivity (the binary "has ≥1 landmine" flag) saturates at ~100%, it stops being a useful way to distinguish *which* solvable states are riskier than others — nearly all of them qualify. The **continuous** fragility measure (fraction of moves that ruin the solution) is the more informative quantity going forward, and it does vary meaningfully by problem even though the binary flag doesn't.

### Triggers 3 & 4 — the raw numbers are misleading; occupancy weighting tells the real story

Taken at face value, triggers 3 and 4 describe almost exactly the failure mode `docs/project_plan.md` §17 warns about ("almost every action has A ≈ 0" / "rework the environment"). But the raw, unweighted statistic pools **all 15.3 million reachable state-action pairs across all 750 problems**, and the overwhelming majority of that space lives inside subtrees that are *already unsolvable*. Once a trajectory enters a dead subtree, `Vπ(s) = Qπ(s,a) = 0` for literally every remaining action under every baseline, by simple arithmetic — none of the three baselines consult the oracle, so none of them have any way to "notice" they're doomed and behave differently there. That single mechanical fact mass-produces exact zeros across a huge fraction of the graph, and it produces them **identically** for all three policies, which is exactly what would make trigger 4 fire regardless of whether the policies are actually different in any way that matters.

This is precisely why Step 7 asked for an occupancy-weighted companion view — weighting by how often a policy actually *visits* each state-action pair, rather than counting every corner of the graph equally. Restricted to solvable problems (where credit assignment is even a meaningful question) and weighted by visitation:

| Policy | mean negative mass | mean positive mass | mean \|Aπ\| |
|---|---|---|---|
| uniform | 24.7% | 5.9% | 0.0023 |
| greedy-distance | 0.4% | 0.2% | 0.0018 |
| epsilon-greedy-distance | 25.8% | 11.4% | 0.0170 |

Uniform and epsilon-greedy show **30–37% of their visited probability mass sitting on actions with genuinely nonzero advantage** — a rich, usable signal, not a degenerate one. Greedy-distance remains close to zero even under occupancy weighting, but for a specific and well-understood reason: it is *near-deterministic* (it puts ~all its mass on whichever single action currently looks best), so by construction `Vπ(s) ≈ Qπ(s, chosen action)` almost everywhere it actually goes — a near-deterministic policy has very little on-policy variance left to assign credit to. That's a property of the baseline, not a flaw in the environment, but it's a useful methodological note for Phase 3: a purely-greedy baseline is a poor subject for testing per-action credit estimators, because it rarely deviates from its own single path.

Trigger 4's "the three policies look identical" conclusion does not survive the occupancy-weighted view either: greedy-distance's weighted mean\|Aπ\| (0.0018) is roughly an order of magnitude smaller than epsilon-greedy's (0.0170), and their negative-mass shares differ by two orders of magnitude (0.4% vs 25.8%). The policies are clearly, substantially different once you look at where they actually spend their probability mass — the "identical" result was an artifact of the same dead-subtree zero-inflation as trigger 3.

### An unanticipated finding: solvability scales strongly (and counter-intuitively) with horizon

Not one of the six predeclared triggers, but directly relevant to the project's "did anything happen you didn't expect?" question (`docs/project_plan.md` §17): under the current fixed target range (`100–999`), raw solvability rate is **6.0% at n=4, 37.6% at n=5, and 82.4% at n=6**. More starting numbers makes the puzzle *easier*, not harder — the opposite of the intended "difficulty primarily through longer horizon" design goal in §12. This makes sense mechanically (more numbers means more ways to combine toward a fixed 3-digit target) but is still worth flagging explicitly.

Reassuringly, this looks like a **base-rate effect, not a signal-quality effect**: conditioned on being solvable, both the pooled fragility (~83–85% ruining-action rate at every horizon) and the occupancy-weighted credit richness under epsilon-greedy (mean\|Aπ\| 0.015–0.020 at every horizon) are fairly stable across n=4/5/6. The credit structure *within* a solvable problem doesn't obviously get easier or harder with horizon in this sample — it's specifically the *probability of landing on a solvable problem at all* that swings hugely.

**Practical consequence:** generating a large n=4 slice is expensive (roughly 1 eligible problem per 15–16 attempts, versus roughly 1-in-1.2 at n=6), which is why `experiments/freeze_dataset.py`'s n=4 stratum takes noticeably longer per accepted row than n=6. It does **not** block Step 8's actual "matched profile strata" requirement, which is about train/validation/test being comparable *to each other* (satisfied by construction — all three splits are drawn from the identical per-stratum generation config, varying only the seed namespace). It **does** matter for any future Phase 3/6 experiment that specifically wants to isolate horizon as an independent variable (RQ3): solvability rate and horizon are confounded in this generation config, so a future horizon-vs-credit-difficulty study should either condition on solvability explicitly or use a horizon-adjusted target range designed for that specific comparison. That is a Phase 3/6 design note, not a Phase 1 blocker.

## Go / no-go decision

**Proceed to Step 8 (freeze the dataset) using the current `v1.0` generation config.**

Justification:

- Both computational-degeneracy triggers are clean (0% node-limit hits; every stratum shows a genuine spread of solution depths) — the environment is tractable at the sizes actually used.
- The two triggers that looked most alarming at face value (3 and 4) are explained by a specific, well-understood mechanism (dead-subtree zero-inflation in the *unweighted* statistic) rather than a real absence of credit structure. The occupancy-weighted view — which is the theoretically appropriate lens, since it reflects what an RL rollout would actually experience — shows a rich, clearly policy-differentiated signal for two of the three baselines.
- Fragility is real, pervasive, and consistent across horizons, which is exactly the kind of environment this project needs: most decisions from a winnable state are landmines, and only a narrow "correct doorway" preserves the win (matching E001's hand-verified example exactly).
- The one genuinely unanticipated finding (horizon/solvability confound) doesn't block Phase 1's own deliverable and is documented above as an explicit design note for whoever runs the Phase 3/6 horizon experiment.

No revision to the generation config is being made at this time. Revising specifically to decouple horizon from solvability (e.g. a per-stratum target range) would be a reasonable **future** change if and when a horizon-specific experiment is actually being designed — but making that change now, without a concrete experiment driving it, would just be speculative tuning.
