# MLB Live vs Shadow Lane Policy

## Status

Effective after the Aug. 16-18, 2026 three-day grading review.

This document defines which lane may influence the user-facing MLB pitcher-prop decision and how the remaining lanes must be recorded for evaluation.

## Lane authority

### LLM lane — live

The independent LLM handicapper is the live decision input. It may use verified current research, the exact posted line, pitcher and lineup data supplied by the application, and market probability for the final edge gate. It must not receive or blend the Monte Carlo probability or the numerical model probability when forming its live probability or ranking.

The current web UI still prefers `source='judge'`. For backwards compatibility, `mlb/live_decision.py` promotes an already-produced LLM probability into a `judge` row after deterministic robustness and edge gates. The compatibility label does not mean a model blend occurred. The row metadata must say `decision_basis=llm_only`, `influenced_by_monte_carlo=false`, and `influenced_by_model=false`.

### Monte Carlo lane — shadow

The PA-level simulator must continue to run, persist predictions, and be graded. Its predictions are research/calibration evidence only and have zero authority over the live LLM probability, side, ranking, or final ledger while this policy is active.

### Numerical model lane — shadow

The numerical model is also shadow/testing-only. It must run, persist, and be graded once its serving path is wired. Until then, every live handoff must record the lane as `not_wired` rather than silently treating the missing output as a successful run.

## Robustness gates for the live LLM lane

The Aug. 16-18 review showed two distinct failure modes that must not be collapsed into one generic matchup penalty: workload-tail error and strikeout-conversion error.

For every strikeout prop, the rationale must include `WORKLOAD-STATES`, `K-CONVERSION-STRESS`, `COUNTERARGUMENT`, and an explicit `VERDICT: BET` or `VERDICT: PASS`. The workload analysis must cover short, normal, and extended opportunity. The conversion stress must test a materially lower and higher plausible strikeout-conversion assumption. If a modest adverse conversion or workload assumption flips the recommended side, the prop is fragile and cannot be promoted as a primary play.

For every pitching-outs prop, the rationale must include `WORKLOAD-STATES`, `OUTS-TAIL-STRESS`, `COUNTERARGUMENT`, and an explicit verdict. An outs under that works only in the short-workload state, or an outs over that works only in the extended-workload state, is a pass.

Robustness outranks raw point probability. Among otherwise eligible props for the same pitcher, only one primary market may be promoted.

## Conservative publication rules

Published side probability remains capped at 64% until a larger clean graded sample justifies relaxing it. Normal strikeout and outs-over plays require at least a six-point edge versus the de-vigged market. Outs unders require at least an eight-point edge. The live promotion layer also requires adequate LLM confidence and complete stress evidence.

These are guardrails, not fitted claims that the true optimal thresholds are exactly 6%, 8%, or 64%. Do not optimize them from a three-day sample.

## Auditability

Before handing off a live board, append a lane snapshot for Monte Carlo, numerical model, LLM, and the live decision layer. Record lane mode (`live` or `shadow`), status, prediction count, and whether that lane influenced the live result.

A missing shadow output is an auditability failure, not evidence about predictive quality. Never recreate a missing shadow prediction after the game result is known and present it as the original forecast.

All original predictions remain append-only and next-day grading remains lane-specific.

## What this change deliberately does not do

It does not retune the Monte Carlo equations from three slates of outcomes. The existing simulator has known workload simplifications, but changing its distribution parameters from a tiny recent sample would invite overfitting. Simulator changes should be made only after a larger, versioned backtest isolates a repeatable calibration defect.

It also does not complete the planned Postgres/per-lane-table architecture. The current repository still uses SQLite, so `lane_run_snapshots` is a transitional audit table. The future Postgres migration should preserve the authority and provenance rules in this document rather than changing them implicitly.
