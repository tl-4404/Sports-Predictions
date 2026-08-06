# MLB Pipeline — merged v4.2 + v5 audit policy

Policy version: `sportsaid-v5.0-merged-audit`

This version consolidates both August 5, 2026 audits. When the audits overlap, the
stricter data-integrity rule wins; when v5 identified that a blanket veto was
suppressing valid reads, the veto is replaced by a measured stress lane.

## Default operating mode

- Production defaults to **shadow mode** (`MLB_SHADOW_MODE=1`).
- Promotion requires at least 14 clean timestamped slates, 200 canonical priced and
  graded rows for the relevant lane/market, raw-model Brier better than the no-vig
  market, and positive true CLV.
- Pitching-outs recommendations remain shadow-only until
  `MLB_OUTS_OFFICIAL_ENABLED=1` is set after workload-state validation.
- During probation, at most two independent strikeout props and one primary market
  per pitcher may become official.

## Immutable decision ledger

Every prediction insert also writes a SHA-256-hashed `decision_ledger` snapshot. The
snapshot contains the exact line, line timestamp, first-pitch timestamp, model lane,
raw and decision probabilities, candidate/official status, verification fields,
calibration state, external-model mode, stress results, suppression reason, and rule
trace. Later reruns create new snapshots and never overwrite an earlier decision.

`pipeline_runs` records zero-line and failed-closed runs instead of substituting stale
or inferred lines.

## Line and evidence integrity

Official eligibility requires all of the following:

- exact current live line and standard payout tier;
- at least two independent two-way sportsbook anchors;
- confirmed starter role;
- nine distinct confirmed hitters;
- dated workload evidence;
- pregame timestamps;
- no synthetic, reconstructed, estimated, manual, or user-reported inputs;
- current policy-version validation.

Projected lineups remain useful in a shadow lane. The pipeline stores the early
forecast, reruns after confirmation, and permits only the confirmed-lineup row to
become official.

## Forecast engine

The numerical engine now models explicit short, normal, and extended workload states.
The planned ceiling is estimated from verified limits or recent upper usage rather
than a raw mean contaminated by bad-outing hooks. Short states activate conditionally
on traffic/score pressure. The simulation also includes:

- plate-appearance outcome simulation;
- pitcher/opponent log5 blending;
- park, umpire, and current stuff factors;
- true times-through-order strikeout decay;
- pitcher-specific pitches-per-PA hooks;
- score/traffic-based early removal;
- batters faced, pitches, outs, and strikeouts from the same simulated game.

Strikeout unders are reported under base, +2 BF, and +4 BF lanes and labeled
`skill_under` or `workload_under`. The former blanket +4 BF veto is removed; stress
survival now raises or lowers uncertainty mechanically.

## Calibration

Calibration uses one canonical row per pitcher/game/market/source from the **first
eligible decision snapshot**. A later lineup rerun cannot replace history. Eligible
rows must be exact, priced, standard-tier, pregame, confirmed-lineup, verified-role,
verified-workload observations. Pushes, voids, refunds, synthetic rows, and postgame
reconstructions are excluded.

Each model lane is calibrated separately. Platt scaling is used while samples are
limited, with fallback order:

1. exact market + source;
2. market + all sources;
3. all markets + source;
4. global model.

Both forecast-over and chosen-side calibration are persisted. Reports include Brier,
log loss, ECE, true probability CLV, and the number of clean slates. Segment-specific
promotion is deferred until 500 observations.

## True CLV

CLV is computed only at the exact same line using the final two-way no-vig snapshot
**strictly before first pitch**:

`closing probability of our side - entry probability of our side`

When an exact close is unavailable, CLV remains null. It is never fabricated from a
moved threshold or post-start price.

## Deterministic decision policy

Bootstrap gates:

| Market | Minimum model-market edge | Minimum side probability |
|---|---:|---:|
| Strikeouts, either side | 8 points | 60% |
| Outs under | 10 points | 60% |
| Outs over | 6 points | 57% |

Uncertainty can add up to three additional edge points. Sim/LLM disagreement above 12
points, opposite preferred sides, stale/moved lines, external near-line disagreement
without a unique verified counter-signal, or integrity failures produce review/pass.

The independent LLM never sees market prices or simulator output. The final judge uses
75% simulator / 25% independent LLM, may move no more than three points, may only erode
the simulator edge, and cannot create or flip a play.

## External projections

`SPORTSAID_BATX_MODE` is immutable per run:

- `bootstrap_gate`: near-line or large disagreement triggers review;
- `advisory`: logged diagnostic only;
- `off`: standalone ablation.

External projections never supply the base forecast or automatically reverse a side.

## Pushes, voids, and refunds

The result ledger separates statistical outcome from settlement:

- ordinary win/loss: ROI-eligible only for official rows;
- push or pitcher did not start: `settlement_status=refund`, zero P&L, excluded from
  ROI and calibration;
- shadow/candidate rows: graded for forecast accuracy, never included in ROI.

PrizePicks whole-number ties/voids are card-level events. A tie plus win uses the
configured smaller-card multiplier; two ties/voids refund the card. The pipeline
requires the displayed full-card and tie-regrade multipliers because platform terms
can change, and it does not count a returned leg twice.

## Required environment defaults

```bash
MLB_SHADOW_MODE=1
MLB_MIN_DIAGNOSTIC_ROWS=50
MLB_MIN_VALIDATION_ROWS=200
MLB_FULL_SEGMENT_ROWS=500
MLB_OUTS_OFFICIAL_ENABLED=0
MLB_MAX_OFFICIAL_KS=2
SPORTSAID_BATX_MODE=bootstrap_gate
```
