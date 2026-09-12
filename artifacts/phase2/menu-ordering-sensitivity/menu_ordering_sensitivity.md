# Menu-Order Sensitivity

Root: numbers `[3, 7, 10, 25]`, target `245`. Compared `5` complete-DAG policies over `103` nonterminal states.

Winning root action: `10 + 25 = 35`.

## Root policy by menu ordering

| Menu ordering | Vπ(root) | P(winning action) | Top root action | Label | P(top) |
|---|---:|---:|---|---|---:|
| `phase2-sensitivity-1` | 0.17704055 | 0.18137166 | `7 - 3 = 4` | `a` | 0.30852359 |
| `phase2-sensitivity-2` | 1.3596431e-05 | 0.00029682 | `25 - 10 = 15` | `g` | 0.68907851 |
| `phase2-sensitivity-3` | 0.0040974912 | 0.00429259 | `7 - 3 = 4` | `g` | 0.33051147 |
| `phase2-sensitivity-4` | 3.0745919e-05 | 0.00174118 | `3 * 25 = 75` | `g` | 0.29743306 |
| `phase2-v1` | 0.027104837 | 0.03152394 | `10 - 7 = 3` | `a` | 0.29909086 |

## Pairwise root total-variation distance

| First ordering | Second ordering | TV distance |
|---|---|---:|
| `phase2-sensitivity-1` | `phase2-sensitivity-2` | 0.80430614 |
| `phase2-sensitivity-1` | `phase2-sensitivity-3` | 0.42436979 |
| `phase2-sensitivity-1` | `phase2-sensitivity-4` | 0.75136452 |
| `phase2-sensitivity-1` | `phase2-v1` | 0.64835891 |
| `phase2-sensitivity-2` | `phase2-sensitivity-3` | 0.79653943 |
| `phase2-sensitivity-2` | `phase2-sensitivity-4` | 0.76847161 |
| `phase2-sensitivity-2` | `phase2-v1` | 0.69962054 |
| `phase2-sensitivity-3` | `phase2-sensitivity-4` | 0.74613249 |
| `phase2-sensitivity-3` | `phase2-v1` | 0.71733334 |
| `phase2-sensitivity-4` | `phase2-v1` | 0.68012927 |

## Average probability by option label

### `phase2-sensitivity-1`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.58216306 |
| `b` | 103 | 0.21124147 |
| `c` | 103 | 0.08563043 |
| `d` | 19 | 0.07802236 |
| `e` | 19 | 0.05351164 |
| `f` | 16 | 0.02425008 |
| `g` | 16 | 0.34884529 |
| `h` | 16 | 0.20524483 |
| `i` | 16 | 0.01127703 |
| `j` | 1 | 0.02532516 |
| `k` | 1 | 0.01795823 |
| `l` | 1 | 0.07800698 |
| `m` | 1 | 0.18137166 |
| `n` | 1 | 0.00772374 |
| `o` | 1 | 0.00211155 |
| `p` | 1 | 0.14347679 |
| `q` | 1 | 0.04585794 |
| `r` | 1 | 0.02454599 |

### `phase2-sensitivity-2`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.64081073 |
| `b` | 103 | 0.18747101 |
| `c` | 103 | 0.08487039 |
| `d` | 19 | 0.03884984 |
| `e` | 19 | 0.03416142 |
| `f` | 16 | 0.02749118 |
| `g` | 16 | 0.25227439 |
| `h` | 16 | 0.17446047 |
| `i` | 16 | 0.00548104 |
| `j` | 1 | 0.04838091 |
| `k` | 1 | 0.04544965 |
| `l` | 1 | 0.02048576 |
| `m` | 1 | 0.01725074 |
| `n` | 1 | 0.00124964 |
| `o` | 1 | 0.00044557 |
| `p` | 1 | 0.01522373 |
| `q` | 1 | 0.05150121 |
| `r` | 1 | 0.00281611 |

### `phase2-sensitivity-3`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.64802315 |
| `b` | 103 | 0.14264562 |
| `c` | 103 | 0.10321957 |
| `d` | 19 | 0.06709744 |
| `e` | 19 | 0.02856561 |
| `f` | 16 | 0.02314029 |
| `g` | 16 | 0.25230264 |
| `h` | 16 | 0.20642735 |
| `i` | 16 | 0.07534209 |
| `j` | 1 | 0.04543426 |
| `k` | 1 | 0.01954103 |
| `l` | 1 | 0.01203888 |
| `m` | 1 | 0.08099532 |
| `n` | 1 | 0.00429259 |
| `o` | 1 | 0.00456944 |
| `p` | 1 | 0.00409602 |
| `q` | 1 | 0.02214289 |
| `r` | 1 | 0.00339572 |

### `phase2-sensitivity-4`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.62321319 |
| `b` | 103 | 0.17026082 |
| `c` | 103 | 0.11316006 |
| `d` | 19 | 0.04832501 |
| `e` | 19 | 0.02079027 |
| `f` | 16 | 0.04123580 |
| `g` | 16 | 0.25849071 |
| `h` | 16 | 0.19235496 |
| `i` | 16 | 0.01203687 |
| `j` | 1 | 0.08005302 |
| `k` | 1 | 0.03038470 |
| `l` | 1 | 0.06332715 |
| `m` | 1 | 0.04088708 |
| `n` | 1 | 0.00315287 |
| `o` | 1 | 0.00203565 |
| `p` | 1 | 0.00553347 |
| `q` | 1 | 0.01083400 |
| `r` | 1 | 0.00139908 |

### `phase2-v1`

| Label | Appearances | Mean probability |
|---|---:|---:|
| `a` | 103 | 0.68559709 |
| `b` | 103 | 0.15665903 |
| `c` | 103 | 0.06497094 |
| `d` | 19 | 0.04097590 |
| `e` | 19 | 0.01451223 |
| `f` | 16 | 0.00692601 |
| `g` | 16 | 0.24890174 |
| `h` | 16 | 0.23043672 |
| `i` | 16 | 0.01056388 |
| `j` | 1 | 0.04586707 |
| `k` | 1 | 0.19011372 |
| `l` | 1 | 0.09559518 |
| `m` | 1 | 0.11176209 |
| `n` | 1 | 0.02003784 |
| `o` | 1 | 0.00327109 |
| `p` | 1 | 0.02342661 |
| `q` | 1 | 0.03462228 |
| `r` | 1 | 0.02738848 |

Root distributions are matched by arithmetic `Action`, not by option label. Large variation is reported as a result rather than treated as a failed run.
