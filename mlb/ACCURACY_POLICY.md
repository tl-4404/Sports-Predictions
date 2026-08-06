# MLB accuracy policy (`accuracy_v2`)

The prior pipeline mixed three different betting policies:

- the simulator required a 6-point edge and a posted lineup;
- the independent LLM used a 3-point edge and could create a wager;
- the judge blended the two and could override the simulator;
- PrizePicks selected individual legs without a sportsbook anchor or exact
  card-level payout/tie EV.

That made the displayed confidence impossible to validate as one strategy. It
also let stale lines, incomplete workload evidence, repeated reruns, correlated
alternate ladders, and synthetic backtests look like independent evidence.

`accuracy_v2` deliberately starts in paper/market-only mode.

## Automatic sportsbook requirements

All of these must be true:

1. The exact policy version has at least 50 unique, priced, graded observations
   for the market. One main line per pitcher/market/game is counted.
2. Raw-model Brier score beats the contemporaneous de-vigged market and paper
   selections have positive entry-to-close CLV.
3. The opposing batting order has nine distinct hitters.
4. The game is still pre-game on the correct Eastern slate.
5. At least three dated recent starts support a stable starter role and leash;
   short rest, long layoffs, active ramps, opener usage, and unstable pitch
   counts force HOLD.
6. The exact line has current two-way prices from at least two books.
7. The calibrated model/market discrepancy is 6–8 percentage points. Gaps
   below 5 points use the market; gaps above 8 points are frozen for a role,
   line, or data investigation.
8. The independent LLM safety judge confirms the simulator candidate. It may
   weaken or veto the candidate but cannot strengthen, flip, or create it.

Until all eight conditions pass, the app stores the forecast for grading but
sets no executable side or stake.

## BAT X-inspired modeling choices

The implementation borrows the useful *structure* of a sharp projection system,
not proprietary BAT X numbers:

- pitcher and lineup per-PA skill are shrunk to stable league priors;
- pitcher/batter matchups use log5 odds ratios;
- park, umpire, current stuff, and velocity remain separate adjustments;
- a workload mixture separates the manager's intended pitch ceiling from
  performance-driven early hooks;
- later trips through the batting order receive a conservative K penalty;
- probabilities come from the simulated distribution, not a point projection;
- the market is treated as a strong prior and a required validation benchmark.

No coefficient is considered proven merely because it improves a synthetic
self-consistency test.

## PrizePicks

PrizePicks legs require the same exact-line sportsbook anchor and judge
confirmation. A leg is never called a play by itself. The app builds cross-game
two-pick cards, uses the displayed multiplier and tie+win multiplier supplied in
the environment, includes push probability in expected value, requires at
least +8% full-card EV, avoids reusing a leg, and authorizes at most one card by
default.

Goblin/demon lines remain audit-only because their payout adjustment is
lineup-specific. Underdog remains unsupported until there is a reliable,
source-specific live-board and payout connector.

## Honest validation

- `make selftest` checks only that the simulator/scoring machinery is internally
  coherent. It is not evidence of real-world edge.
- `make backtest DATASET=... ODDS=...` requires a real historical dataset and
  timestamped odds archive.
- Outcome-derived synthetic lines are opt-in diagnostics and are labeled
  invalid for accuracy or ROI claims.
- CLV is closing market probability minus entry market probability on the
  tracked side, using the last snapshot before first pitch.
- Configure `SPORTSAID_DB_PATH` on durable storage or validation history will
  disappear on redeploy.
