# Menu-Order Sensitivity

Root: numbers `[3, 7, 10, 25]`, target `245`. Compared `5` complete-DAG policies over `103` nonterminal states.

Winning root action: `10 + 25 = 35`.

## Root policy by menu ordering

| Menu ordering | Vπ(root) | P(winning action) | Top root action | Label | P(top) |
|---|---:|---:|---|---|---:|
| `phase2-sensitivity-1` | 0.02910034 | 0.02930101 | `10 * 25 = 250` | `d` | 0.58852661 |
| `phase2-sensitivity-2` | 0.0089117111 | 0.00892817 | `10 * 25 = 250` | `o` | 0.35110534 |
| `phase2-sensitivity-3` | 0.051199001 | 0.05148336 | `10 * 25 = 250` | `r` | 0.50396588 |
| `phase2-sensitivity-4` | 0.11060559 | 0.11093419 | `10 * 25 = 250` | `i` | 0.24230254 |
| `phase2-v1` | 0.010065625 | 0.01012987 | `10 * 25 = 250` | `r` | 0.42405501 |

## Pairwise root total-variation distance

| First ordering | Second ordering | TV distance |
|---|---|---:|
| `phase2-sensitivity-1` | `phase2-sensitivity-2` | 0.41863066 |
| `phase2-sensitivity-1` | `phase2-sensitivity-3` | 0.24407988 |
| `phase2-sensitivity-1` | `phase2-sensitivity-4` | 0.47204445 |
| `phase2-sensitivity-1` | `phase2-v1` | 0.28290584 |
| `phase2-sensitivity-2` | `phase2-sensitivity-3` | 0.37963227 |
| `phase2-sensitivity-2` | `phase2-sensitivity-4` | 0.37416791 |
| `phase2-sensitivity-2` | `phase2-v1` | 0.33401762 |
| `phase2-sensitivity-3` | `phase2-sensitivity-4` | 0.44783573 |
| `phase2-sensitivity-3` | `phase2-v1` | 0.35332248 |
| `phase2-sensitivity-4` | `phase2-v1` | 0.45872951 |

## Average probability by option label

### `phase2-sensitivity-1`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.37760175 |
| `b` | 103 | 0.27153764 |
| `c` | 103 | 0.24993322 |
| `d` | 19 | 0.18277455 |
| `e` | 19 | 0.03945572 |
| `f` | 16 | 0.08598740 |
| `g` | 16 | 0.10945774 |
| `h` | 16 | 0.12009885 |
| `i` | 16 | 0.04982665 |
| `j` | 1 | 0.00225945 |
| `k` | 1 | 0.00162742 |
| `l` | 1 | 0.07719789 |
| `m` | 1 | 0.02930101 |
| `n` | 1 | 0.00996917 |
| `o` | 1 | 0.01982606 |
| `p` | 1 | 0.07139638 |
| `q` | 1 | 0.09758728 |
| `r` | 1 | 0.01805183 |

### `phase2-sensitivity-2`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.38213745 |
| `b` | 103 | 0.22766806 |
| `c` | 103 | 0.29607074 |
| `d` | 19 | 0.16848512 |
| `e` | 19 | 0.05253362 |
| `f` | 16 | 0.10547320 |
| `g` | 16 | 0.05163138 |
| `h` | 16 | 0.08375146 |
| `i` | 16 | 0.05452425 |
| `j` | 1 | 0.09159130 |
| `k` | 1 | 0.16846218 |
| `l` | 1 | 0.02108551 |
| `m` | 1 | 0.01980801 |
| `n` | 1 | 0.00812919 |
| `o` | 1 | 0.35110534 |
| `p` | 1 | 0.05821897 |
| `q` | 1 | 0.03369456 |
| `r` | 1 | 0.01720950 |

### `phase2-sensitivity-3`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.39666820 |
| `b` | 103 | 0.22159668 |
| `c` | 103 | 0.28535053 |
| `d` | 19 | 0.13784036 |
| `e` | 19 | 0.05159941 |
| `f` | 16 | 0.07816283 |
| `g` | 16 | 0.03869788 |
| `h` | 16 | 0.12423850 |
| `i` | 16 | 0.10744633 |
| `j` | 1 | 0.00471446 |
| `k` | 1 | 0.00867121 |
| `l` | 1 | 0.01130939 |
| `m` | 1 | 0.01893967 |
| `n` | 1 | 0.05148336 |
| `o` | 1 | 0.01013767 |
| `p` | 1 | 0.09469238 |
| `q` | 1 | 0.04761432 |
| `r` | 1 | 0.50396588 |

### `phase2-sensitivity-4`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.37827223 |
| `b` | 103 | 0.24717423 |
| `c` | 103 | 0.28644113 |
| `d` | 19 | 0.14733498 |
| `e` | 19 | 0.11162059 |
| `f` | 16 | 0.05906851 |
| `g` | 16 | 0.03819717 |
| `h` | 16 | 0.06109660 |
| `i` | 16 | 0.08398323 |
| `j` | 1 | 0.01927747 |
| `k` | 1 | 0.07389810 |
| `l` | 1 | 0.01648889 |
| `m` | 1 | 0.03228369 |
| `n` | 1 | 0.01410370 |
| `o` | 1 | 0.02718559 |
| `p` | 1 | 0.06521484 |
| `q` | 1 | 0.01133264 |
| `r` | 1 | 0.01810951 |

### `phase2-v1`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.39118617 |
| `b` | 103 | 0.28858935 |
| `c` | 103 | 0.23385198 |
| `d` | 19 | 0.12130782 |
| `e` | 19 | 0.03912196 |
| `f` | 16 | 0.08537006 |
| `g` | 16 | 0.04082092 |
| `h` | 16 | 0.15172269 |
| `i` | 16 | 0.03879790 |
| `j` | 1 | 0.02318750 |
| `k` | 1 | 0.00384486 |
| `l` | 1 | 0.08350138 |
| `m` | 1 | 0.01363123 |
| `n` | 1 | 0.01184301 |
| `o` | 1 | 0.04069532 |
| `p` | 1 | 0.15120147 |
| `q` | 1 | 0.02885731 |
| `r` | 1 | 0.42405501 |

Root distributions are matched by arithmetic `Action`, not by option label. Large variation is reported as a result rather than treated as a failed run.
