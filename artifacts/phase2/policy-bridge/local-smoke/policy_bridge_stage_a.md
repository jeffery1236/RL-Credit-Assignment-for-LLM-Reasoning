# Policy Bridge — Stage A

Compares the menu-order-averaged constrained next-token policy with the sampled state-only deliberative policy at selected roots.

- Roots: `1`
- Valid legal deliberative response rate: `0.000%`
- Median Spearman: `None`
- Median TV: `None`
- Gate passed: `False`

Constrained-policy credit is exact conditional on cached/recorded next-token probabilities. Deliberative policy measurements are sampled; invalid responses are reported separately and are excluded only from this Stage A action-distribution comparison.
