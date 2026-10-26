# MLB pitcher-props pipeline

Implemented targets are pitcher strikeouts and pitcher outs. The [October audit](MLB_AUDIT_2026-10-26.md) distinguishes current behavior, validation and remaining work. Fantasy score, MLB game markets and hitter models are not implemented in serving.

## Execution and publication

The numerical log5 simulator produces shadow distributions. Independent LLM research examines current context without simulator output or market prices; the bounded judge compares those lanes with the no-vig market. Direct chat runs refresh prices and shadow forecasts first. Full orchestration reuses its just-fetched inputs and skips dependent publication after incomplete refreshes.

Only the final judge can publish an actionable recommendation. Database and UI checks require the current model version, fresh live-source prices, a future game start, nine unique opposing batters, reviewed current role/workload evidence and positive EV at the offered price. Missing evidence produces HOLD. Set `MLB_USAGE_EVIDENCE_PATH` to a JSON array described in the audit; without reviewed input, official recommendations remain held. The adapter validates structure/timing; upstream researchers own factual verification.

Policy `chat-audited-v5.0` retains the existing 70/30 blend, ±5-point judge band, 64% side cap and 6-point base edge hurdle (8 points for outs unders). These settings remain unvalidated policy, not proof of edge. Only one primary market per pitcher/game is selected. A K under also must survive the coded +4 BF numerical scenario.

## Data and evaluation

- Pitcher rates use game logs strictly before the slate date. Recent pitch counts include postseason outings. Current live hitter splits use data known at fetch time; historical hitter platoon queries use completed prior seasons.
- Park/umpire factors and LLM form summaries require dated snapshots. Missing/future snapshots are neutral or unavailable. Hand-set CSW/velocity multipliers are disabled in the default simulator.
- SQLite line, prediction, result and calibration ledgers reject UPDATE/DELETE. Raw/final probabilities, pushes, versions, calibration identity and publication reasons are retained. Fixtures cannot authorize bets.
- Grading waits for final complete official boxes and matches game/player IDs. Failures remain pending. CLV compares entry/closing fair probabilities on the exact contract using a live quote within 15 minutes before scheduled start.
- Calibration uses the first priced raw pregame forecast per game/player/market/source across reruns/thresholds. Only compatible versions are used. Application requires prior creation/cutoff dates, 200 segment observations, two usable knots and 30 observations per knot. Shrinkage/monotone pooling restrain instability; fit metrics are in-sample diagnostics.
- The scorecard selects the first official recommendation per pitcher/game/source. Decided hit rate excludes pushes. Staked paper ROI includes push stakes and excludes void stakes. No actual fill ledger exists.
- PrizePicks stores standard More/Less research probabilities with separate push mass. Without a verified entry ledger it cannot report entry EV, playability or ROI. `entry_ev` evaluates externally supplied joint outcomes and exact settlement terms.

## Reproduction

```bash
pip install -r requirements-dev.txt
make test
make audit-baseline
python run_pipeline.py --stage doctor
```

`make test` isolates seven legacy script suites, then runs the focused pytest cases. Do not collect the executable legacy scripts in one pytest process.

`make audit-baseline` reproduces the frozen annual-rate test, tuned on 2023/2024 and tested on 2025. The challenger remains shadow; this does not establish daily-prop accuracy or betting returns. Synthetic backtests test numerical consistency only. Historical betting evaluation requires archived decision-time features/lineups and timestamped prices; attaching an undated CSV to box scores does not create a valid betting replay.

`SPORTSAID_BATX_MODE` retains `bootstrap_gate`, `advisory` and `off` for separately logged future comparisons. External projections are benchmarks; missing values are never fabricated. Promotion needs frozen daily out-of-sample calibration, coverage and performance evidence.
