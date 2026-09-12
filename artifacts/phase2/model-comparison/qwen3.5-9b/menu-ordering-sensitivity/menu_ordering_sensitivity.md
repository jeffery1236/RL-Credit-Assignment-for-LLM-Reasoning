# Menu-Order Sensitivity

Root: numbers `[3, 7, 10, 25]`, target `245`. Compared `5` complete-DAG policies over `103` nonterminal states.

Winning root action: `10 + 25 = 35`.

## Root policy by menu ordering

| Menu ordering | Vπ(root) | P(winning action) | Top root action | Label | P(top) |
|---|---:|---:|---|---|---:|
| `phase2-sensitivity-1` | 0.017949062 | 0.01801634 | `10 * 25 = 250` | `d` | 0.70850293 |
| `phase2-sensitivity-2` | 0.010910721 | 0.01091428 | `10 * 25 = 250` | `o` | 0.69667718 |
| `phase2-sensitivity-3` | 0.021393668 | 0.02140606 | `10 * 25 = 250` | `r` | 0.75458976 |
| `phase2-sensitivity-4` | 0.052959287 | 0.05299756 | `10 * 25 = 250` | `i` | 0.68728283 |
| `phase2-v1` | 0.01055716 | 0.01056145 | `10 * 25 = 250` | `r` | 0.55889432 |

## Pairwise root total-variation distance

| First ordering | Second ordering | TV distance |
|---|---|---:|
| `phase2-sensitivity-1` | `phase2-sensitivity-2` | 0.12286342 |
| `phase2-sensitivity-1` | `phase2-sensitivity-3` | 0.09357160 |
| `phase2-sensitivity-1` | `phase2-sensitivity-4` | 0.20195783 |
| `phase2-sensitivity-1` | `phase2-v1` | 0.28837812 |
| `phase2-sensitivity-2` | `phase2-sensitivity-3` | 0.14374350 |
| `phase2-sensitivity-2` | `phase2-sensitivity-4` | 0.12789182 |
| `phase2-sensitivity-2` | `phase2-v1` | 0.22400786 |
| `phase2-sensitivity-3` | `phase2-sensitivity-4` | 0.22116853 |
| `phase2-sensitivity-3` | `phase2-v1` | 0.29666338 |
| `phase2-sensitivity-4` | `phase2-v1` | 0.20562339 |

## Average probability by option label

### `phase2-sensitivity-1`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.28284242 |
| `b` | 103 | 0.29229366 |
| `c` | 103 | 0.31310081 |
| `d` | 19 | 0.14040959 |
| `e` | 19 | 0.07125925 |
| `f` | 16 | 0.07352406 |
| `g` | 16 | 0.14915710 |
| `h` | 16 | 0.16458982 |
| `i` | 16 | 0.06741063 |
| `j` | 1 | 0.00178386 |
| `k` | 1 | 0.00046173 |
| `l` | 1 | 0.00211839 |
| `m` | 1 | 0.01801634 |
| `n` | 1 | 0.00079158 |
| `o` | 1 | 0.00274139 |
| `p` | 1 | 0.00269889 |
| `q` | 1 | 0.18195860 |
| `r` | 1 | 0.00441509 |

### `phase2-sensitivity-2`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.27388403 |
| `b` | 103 | 0.25243499 |
| `c` | 103 | 0.34550446 |
| `d` | 19 | 0.22341426 |
| `e` | 19 | 0.15472485 |
| `f` | 16 | 0.07416173 |
| `g` | 16 | 0.05545193 |
| `h` | 16 | 0.07249280 |
| `i` | 16 | 0.11812139 |
| `j` | 1 | 0.07458558 |
| `k` | 1 | 0.10518252 |
| `l` | 1 | 0.00175408 |
| `m` | 1 | 0.00102315 |
| `n` | 1 | 0.00142050 |
| `o` | 1 | 0.69667718 |
| `p` | 1 | 0.00179568 |
| `q` | 1 | 0.01057848 |
| `r` | 1 | 0.00087515 |

### `phase2-sensitivity-3`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.26744489 |
| `b` | 103 | 0.27342039 |
| `c` | 103 | 0.33761234 |
| `d` | 19 | 0.17545489 |
| `e` | 19 | 0.06993237 |
| `f` | 16 | 0.12652439 |
| `g` | 16 | 0.05423868 |
| `h` | 16 | 0.14310548 |
| `i` | 16 | 0.11743723 |
| `j` | 1 | 0.00074985 |
| `k` | 1 | 0.00030296 |
| `l` | 1 | 0.00034063 |
| `m` | 1 | 0.00036545 |
| `n` | 1 | 0.02140606 |
| `o` | 1 | 0.00051536 |
| `p` | 1 | 0.00485155 |
| `q` | 1 | 0.01043248 |
| `r` | 1 | 0.75458976 |

### `phase2-sensitivity-4`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.25186813 |
| `b` | 103 | 0.27238181 |
| `c` | 103 | 0.36079548 |
| `d` | 19 | 0.21199122 |
| `e` | 19 | 0.18871838 |
| `f` | 16 | 0.05652616 |
| `g` | 16 | 0.02848498 |
| `h` | 16 | 0.04590252 |
| `i` | 16 | 0.13023918 |
| `j` | 1 | 0.00623152 |
| `k` | 1 | 0.03115566 |
| `l` | 1 | 0.00141234 |
| `m` | 1 | 0.00338802 |
| `n` | 1 | 0.00048809 |
| `o` | 1 | 0.00238377 |
| `p` | 1 | 0.00084996 |
| `q` | 1 | 0.00200734 |
| `r` | 1 | 0.00047678 |

### `phase2-v1`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.26338230 |
| `b` | 103 | 0.36745218 |
| `c` | 103 | 0.26300429 |
| `d` | 19 | 0.13209679 |
| `e` | 19 | 0.11126274 |
| `f` | 16 | 0.06489627 |
| `g` | 16 | 0.03501332 |
| `h` | 16 | 0.16214032 |
| `i` | 16 | 0.08772435 |
| `j` | 1 | 0.01089670 |
| `k` | 1 | 0.00433441 |
| `l` | 1 | 0.07217440 |
| `m` | 1 | 0.00835479 |
| `n` | 1 | 0.02067831 |
| `o` | 1 | 0.00348280 |
| `p` | 1 | 0.01023651 |
| `q` | 1 | 0.02533560 |
| `r` | 1 | 0.55889432 |

Root distributions are matched by arithmetic `Action`, not by option label. Large variation is reported as a result rather than treated as a failed run.
