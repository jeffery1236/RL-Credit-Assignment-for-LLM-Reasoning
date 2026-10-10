# Phase 3: Can more simulations rank moves better?

Status: **completed**<br>
Official result: **NOT-DEMONSTRATED**

## How to read this report

- **One rollout** scores a move using one simulated game after taking it.
- **Before/after ×K** compares K simulated games before the move with K
  simulated games after it. K means the number of games simulated both
  before and after a move.
- The accuracy score measures how well a method ranks the available moves:
  `1.0` is perfect and `0.0` means no useful ranking agreement.

## How accurately each method ranked the moves
- One rollout: 0.490
- Before/after ×1: 0.320
- Before/after ×2: 0.429
- Before/after ×4: 0.464
- Before/after ×8: 0.503
- Before/after ×16: 0.494

## Official decision check
- Typical K=16 improvement over one rollout: -0.023
- K=16 won in 38.1% of resampled trials
- Required to pass: improvement of at least 0.100 and wins in at least
  80.0% of resampled trials

## How often each method picked the truly best move

This counts positions with one unambiguously best move and asks whether
the scoring method chose that same move.

- One rollout: 109/140 (77.9%)
- Before/after ×1: 96/140 (68.6%)
- Before/after ×2: 110/140 (78.6%)
- Before/after ×4: 117/140 (83.6%)
- Before/after ×8: 125/140 (89.3%)
- Before/after ×16: 125/140 (89.3%)

## Extra checks run after seeing the result

These checks explain the result; they do not replace the official decision.
They reuse the same resampled trials, so they are not independent evidence.

- Use one starting-position estimate for every move: CAPABLE, typical improvement 0.348, K=16 won 100.0% of resamples
- Give each decision step equal weight: NOT-DEMONSTRATED, typical improvement -0.108, K=16 won 0.2% of resamples
- Ignore positions where every move is tied: NOT-DEMONSTRATED, typical improvement -0.010, K=16 won 45.2% of resamples
- Ignore the final decision step: NOT-DEMONSTRATED, typical improvement 0.001, K=16 won 50.1% of resamples
