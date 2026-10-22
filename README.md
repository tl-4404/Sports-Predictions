# SportsAid Betting Aid App

For NBA data sources and the historical dataset builder, see
[docs/NBA_BACKTESTING.md](docs/NBA_BACKTESTING.md).

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

### Lifecycle directory map

```text
providers/          Shared external data clients
├── polymarket.py   Polymarket US moneyline prices
├── prizepicks.py   PrizePicks provider and existing MLB evaluation lane
└── http.py         Shared scraper HTTP helper
nba/
├── providers/      NBA Stats, NBA sportsbook odds, ESPN NBA injuries
├── data/           Historical training/backtest dataset builders
├── training/       PyTorch moneyline model trainer
├── prediction/     Daily features and prediction orchestration
├── cache/          Generated datasets (gitignored)
└── models/         Model weights, scaler and feature list (gitignored)
mlb/
├── providers/      MLB StatsAPI and MLB sportsbook props
├── data/           Historical collection and feature-cache builders
├── training/       Model fitting, calibration and backtesting
├── prediction/     Simulator, daily forecasts, betting math and LLM lanes
├── cache/          Generated datasets and accuracy caches (gitignored)
├── models/         Generated model artifacts (gitignored)
├── db.py           MLB persistence
├── grade.py        Postgame grading and CLV
└── doctor.py       Operational readiness checks
```

Dependencies flow from orchestration toward these stages; training-only modules are
never imported by the serving providers or numerical simulator.

See [docs/DIRECTORY_LAYOUT.md](docs/DIRECTORY_LAYOUT.md) for module responsibilities,
the current NBA inventory, commands, and the distinction between the two NBA feature
schemas. Run Python entrypoints as modules from the repository root:

```bash
python -m providers.polymarket --moneylines --limit 3
python -m nba.providers.odds
python -m nba.prediction.daily
```

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
make test
```

The regression suite covers the chat-aligned policy, bounded judge behavior, edge
thresholds, probability caps and same-pitcher exposure controls.

For a chronological real-game simulator backtest:

```bash
python -m mlb.data.build_backtest_data 2025-03-27 2025-09-28 \
  --warmup-start 2024-03-20 --out mlb/cache/backtest_2025.json
python -m mlb.training.backtest mlb/cache/backtest_2025.json --sims 10000
```

The builder uses prior games only and writes a metadata sidecar describing the
warmup, leakage controls and failures. See
[docs/MLB_PIPELINE.md](docs/MLB_PIPELINE.md#building-chronological-backtest-data).

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
