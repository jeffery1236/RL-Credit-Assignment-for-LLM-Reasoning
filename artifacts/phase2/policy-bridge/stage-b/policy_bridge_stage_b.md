# Policy Bridge — Stage B (conclusion robustness)

Asks whether the ranking of credit estimators survives replacing the constrained next-token policy with full-solution Qwen completions.

- Benchmark state-action pairs: `24`
- Reference split-half stability: `0.9870933188046012`
- Method-level rank agreement across setups: `0.9411239481143202`
- Decision: `GO`

## Median fidelity by method

| Method | Constrained | Deliberative |
| --- | --- | --- |
| `uniform` | `0.5956198637179363` | `0.5956198637179363` |
| `mc_reset_k1` | `0.7726286319233321` | `0.8815555431386801` |
| `mc_reset_k2` | `0.7786658025072327` | `0.900509962061637` |
| `mc_reset_k4` | `0.8282893225264996` | `0.912462039710974` |
| `mc_reset_k8` | `0.8282893225264996` | `0.9228828803737275` |
| `mc_reset_k16` | `0.8282893225264996` | `0.93027002443386` |

## Directional bootstrap support

- constrained: K=16 beats Uniform in `100.0%` of replicates; fidelity rises with K in `100.0%`
- deliberative: K=16 beats Uniform in `100.0%` of replicates; fidelity rises with K in `100.0%`

Constrained-side references are exact A-pi; deliberative-side references are sampled from a held-out rollout pool and carry uncertainty. An unstable sampled reference yields `INCONCLUSIVE` rather than `NO-GO`, because that is evidence about sample size and not about disagreement.
