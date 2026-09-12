# Policy Bridge — completion calibration

Measures how many tokens project-rule full-solution completions actually need, so the paid benchmark can pick the smallest non-truncating cap.

| Token cap | Responses | Truncated | Answer tag | Solved | Mean tokens |
| --- | --- | --- | --- | --- | --- |
| `1024` | `12` | `58.3%` | `50.0%` | `50.0%` | `849` |
| `2048` | `12` | `8.3%` | `91.7%` | `66.7%` | `954` |
| `3072` | `12` | `16.7%` | `100.0%` | `100.0%` | `1321` |

- Maximum acceptable truncation rate: `25%`
- Selected token cap: `2048`
- Local throughput (tokens/second): `33.10`
