# SportsAid Betting Aid App

SportsAid is a Flask-based sports analysis application with an MLB pitcher-props
pipeline designed around reproducible data collection, independent model lanes,
conservative decision rules, and postgame calibration.

The project is an analytical aid, not a guarantee of outcomes. Its primary engineering
goal is to produce timestamped, auditable forecasts that can be graded honestly and
improved without leaking future information into historical results.

## Current MLB architecture

The MLB pipeline has three distinct forecasting roles:

1. **Numerical simulator** — models pitcher outcomes from plate-appearance rates,
   workload, opponent context, park and umpire effects.
2. **Independent LLM audit** — researches current usage, health, pitch mix, lineup,
   bullpen, weather and role information without seeing the simulator forecast or
   market price.
3. **Bounded judge** — compares the independent lanes with the de-vigged market,
   applies risk gates, and may pass on a line. It cannot freely rewrite the forecast.

Predictions, line snapshots, results and calibration runs are stored in SQLite. The
pipeline deliberately separates model generation from grading so historical records
remain append-only and auditable.

See [docs/MLB_PIPELINE.md](docs/MLB_PIPELINE.md) for the full design.

The October audit is implemented as model `mlb-audit-v5` and chat policy
`chat-audited-v5.0`. Read [the findings and limitations](docs/MLB_AUDIT_2026-10-26.md)
before interpreting a board or backtest. Numerical forecasts remain shadow;
unverified prices, incomplete lineups and missing reviewed usage evidence produce
HOLD. PrizePicks leg forecasts have no entry EV or ROI without actual entry terms.

For reproducible development checks, install `requirements-dev.txt`, run `make test`,
and use `make audit-baseline` for the frozen annual-rate holdout. The latter is not
a daily-prop accuracy test or a betting-return backtest.

## Pipeline stages

```bash
python run_pipeline.py --stage odds
python run_pipeline.py --stage predict
python run_pipeline.py --stage llm
python run_pipeline.py --stage prizepicks
python run_pipeline.py --stage grade
python run_pipeline.py --stage calibrate
python run_pipeline.py --stage doctor
```

A full run executes the available stages in order:

```bash
python run_pipeline.py
```

Use `--date YYYY-MM-DD` to run a specific slate and `--sims` to change the simulator
sample count.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The live pipeline uses Flask, SQLite, NumPy/Pandas, HTTP data sources and an optional
Anthropic LLM lane. Training-only dependencies are kept separately from serving
requirements.

### Environment variables

The LLM lane requires:

```bash
export ANTHROPIC_API_KEY="..."
```

External projection systems are controlled independently:

```bash
# Current default while SportsAid builds a larger calibration sample
export SPORTSAID_BATX_MODE=bootstrap_gate

# Mature standalone-model mode
export SPORTSAID_BATX_MODE=advisory

# Ablation: evaluate SportsAid with external projections removed
export SPORTSAID_BATX_MODE=off
```

## THE BAT X policy

THE BAT X is **not** the SportsAid model and must not become the source of the base
projection or selected side.

During early model development, a large disagreement can act as a review gate because
it may reveal stale data, a missed role change or an unrealistic projection. As the
standalone model accumulates enough clean, priced and graded observations, THE BAT X
should move to advisory mode. In advisory mode it is logged as an external benchmark
and may reduce confidence, but it cannot independently veto or reverse a play.

The long-term target is:

- SportsAid creates the forecast independently.
- Market data and verified current information remain primary safeguards.
- THE BAT X is used for monitoring, diagnosis and comparison.
- Ablation runs measure whether SportsAid performs without external projections.

## Current decision safeguards

The v4 policy includes:

- Exact current line and platform verification.
- Confirmed opposing lineup and pitcher role before a play is finalized.
- Minimum six-percentage-point edge for normal markets.
- Higher hurdles for workload-based outs unders and fantasy-score markets.
- Short, normal and extended workload scenarios.
- A `+4 batters faced` stress test for strikeout unders.
- Conservative probability caps while the calibration sample is small.
- One primary official prop per pitcher to avoid counting correlated markets as
  independent opportunities.
- Append-only results and chronological calibration cutoffs.

## Testing

```bash
pytest
```

The regression suite covers the chat-aligned policy, bounded judge behavior, edge
thresholds, probability caps and same-pitcher exposure controls.

## Data integrity principles

- Never backfill a recommendation after the result is known.
- Never substitute an inferred or stale pick'em line for an inaccessible live board.
- Preserve line timestamps and payout type.
- Grade unique priced predictions, not repeated intraday copies.
- Track calibration and closing-line value in addition to raw hit rate.
- Prefer a pass over manufacturing a full card from a weak slate.

## Development status

The MLB pipeline is actively being calibrated. External-model gates are intentionally
configurable so they can be reduced as SportsAid becomes a stronger standalone
system. Changes that affect probability, selection or grading behavior should include
regression tests and a policy-version update.
