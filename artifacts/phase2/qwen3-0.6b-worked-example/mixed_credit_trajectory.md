# Mixed-Credit Trajectory

Final environment reward: `0`. Outcome-only display credit: `-1` at every step.

The environment reward remains 0/1. “Outcome-only display credit” maps the final outcome to +1/-1 only for comparison with per-action Aπ. This trajectory was selected by the predeclared seed-and-first-match rule and does not estimate population frequency.

Selection: sample index `49` under seed `20260815`.

## Step 1

State: numbers `[3, 7, 10, 25]`, target `245`

Menu:
- `a`: `10 - 7 = 3` — π=0.29909086
- `b`: `25 - 7 = 18` — π=0.00530950
- `c`: `7 + 25 = 32` — π=0.00107869
- `d`: `25 - 3 = 22` — π=0.00737151
- `e`: `10 - 3 = 7` — π=0.00029490
- `f`: `3 + 10 = 13` — π=0.00019954
- `g`: `7 + 10 = 17` — π=0.02379552
- `h`: `3 * 25 = 75` — π=0.07925119
- `i`: `10 + 25 = 35` — π=0.03152394 **← selected**
- `j`: `7 * 10 = 70` — π=0.04586707
- `k`: `3 * 7 = 21` — π=0.19011372
- `l`: `7 * 25 = 175` — π=0.09559518
- `m`: `3 * 10 = 30` — π=0.11176209
- `n`: `3 + 25 = 28` — π=0.02003784
- `o`: `7 - 3 = 4` — π=0.00327109
- `p`: `25 - 10 = 15` — π=0.02342661
- `q`: `3 + 7 = 10` — π=0.03462228
- `r`: `10 * 25 = 250` — π=0.02738848

Selected action: `i` (`10 + 25 = 35`)

- `Vπ(s) = 0.02710484`
- `Qπ(s,a) = 0.85981741`
- `Aπ(s,a) = +0.83271257`
- Outcome-only display credit: `-1`

## Step 2

State: numbers `[3, 7, 35]`, target `245`

Menu:
- `a`: `35 - 3 = 32` — π=0.04021420 **← selected**
- `b`: `3 + 7 = 10` — π=0.00144209
- `c`: `3 + 35 = 38` — π=0.00036462
- `d`: `35 - 7 = 28` — π=0.02678843
- `e`: `7 - 3 = 4` — π=0.00041317
- `f`: `3 * 7 = 21` — π=0.00241505
- `g`: `7 + 35 = 42` — π=0.06840665
- `h`: `7 * 35 = 245` — π=0.85981741
- `i`: `3 * 35 = 105` — π=0.00013839

Selected action: `a` (`35 - 3 = 32`)

- `Vπ(s) = 0.85981741`
- `Qπ(s,a) = 0.00000000`
- `Aπ(s,a) = -0.85981741`
- Outcome-only display credit: `-1`

## Step 3

State: numbers `[7, 32]`, target `245`

Menu:
- `a`: `7 * 32 = 224` — π=0.97123757 **← selected**
- `b`: `7 + 32 = 39` — π=0.00121055
- `c`: `32 - 7 = 25` — π=0.02755189

Selected action: `a` (`7 * 32 = 224`)

- `Vπ(s) = 0.00000000`
- `Qπ(s,a) = 0.00000000`
- `Aπ(s,a) = +0.00000000`
- Outcome-only display credit: `-1`
