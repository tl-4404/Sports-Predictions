# SportsAid Betting Aid App

For NBA data sources and the historical dataset builder, see
[nba/docs/BACKTESTING.md](nba/docs/BACKTESTING.md).

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

See [mlb/docs/PIPELINE.md](mlb/docs/PIPELINE.md) for the full design.

### Repository organization

```text
nba/                        NBA ownership
  data/                     Cached datasets, generated outputs, fixture space
  data_fetching/            Dataset builders and NBA/ESPN/odds adapters
  models/                   prediction/ and training/ Python code
  saved_models/             Weights, scalers, and feature lists
  tests/                    NBA regression and forward-testing suites
  ledger/                   PostgreSQL paper-testing ledger
  docs/                     NBA data and evaluation documentation
  deployment/               NBA PostgreSQL Compose configuration
mlb/                        MLB ownership
  data/                     cache/, fixtures/, outputs/, audit ledgers/
  data_fetching/            Dataset/feature builders, StatsAPI and odds adapters
  models/                   prediction/ and training/ Python code
  saved_models/             Serialized trained model artifacts
  tests/                    MLB pipeline, grading, data and integration suites
  docs/                     MLB pipeline design
  pipeline.py               MLB orchestration
shared/                     Cross-league components
  data_fetching/            Polymarket, sportsbook API, PrizePicks and HTTP clients
  betting/                  Odds conversion, de-vigging, consensus and Kelly math
  data/                     Shared application SQLite database
  tests/                    Shared client and architecture tests
  web/                      Flask app, templates, static assets and web tests
  docs/                     Shared provider documentation
scripts/                    Developer utilities and the offline test runner
docs/                       Repository architecture and migration guide
app.py, run_pipeline.py     Stable launch wrappers
```

`models/` contains Python source; `saved_models/` contains generated binaries.
The previously versioned NBA checkpoint, scaler, feature list and current CSVs
remain visible to Git at their new paths; other generated artifacts are ignored.
Dataset builders live in `data_fetching/`; `data/` holds their artifacts. Shared
clients handle transport and parsing; each league owns market selection, model
inputs, predictions and persistence. Heavy training dependencies stay out of live
MLB serving imports.

See [docs/DIRECTORY_LAYOUT.md](docs/DIRECTORY_LAYOUT.md) for path ownership and
migration details. Run module commands from the repository root:

```bash
python -m shared.data_fetching.polymarket --moneylines --limit 3
python -m nba.data_fetching.providers.odds
python -m nba.models.prediction.daily
```

## Pipeline stages

```bash
python mlb/pipeline.py --stage odds
python mlb/pipeline.py --stage predict
python mlb/pipeline.py --stage llm
python mlb/pipeline.py --stage prizepicks
python mlb/pipeline.py --stage grade
python mlb/pipeline.py --stage calibrate
python mlb/pipeline.py --stage doctor
```

A full run executes the available stages in order:

```bash
python mlb/pipeline.py
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
python -m mlb.data_fetching.build_backtest_data 2025-03-27 2025-09-28 \
  --warmup-start 2024-03-20 --out mlb/data/cache/backtest_2025.json
python -m mlb.models.training.backtest mlb/data/cache/backtest_2025.json --sims 10000
```

The builder uses prior games only and writes a metadata sidecar describing the
warmup, leakage controls and failures. See
[mlb/docs/PIPELINE.md](mlb/docs/PIPELINE.md#building-chronological-backtest-data).

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

## Development checks

```bash
make install-dev
make test                   # all suites; network blocked; isolated processes
make test-nba
make test-mlb
make test-shared
make test-web
```

`python scripts/run_tests.py --suite nba` is the equivalent command with an active
virtual environment. Isolation preserves the existing script-style regression
suites, which intentionally replace module globals while exercising whole pipelines.
Install `requirements-training.txt` separately when training or using the saved NBA
neural network; install `nba/requirements-forward.txt` for the NBA ledger workflow.

Start the local NBA ledger from the repository root with:

```bash
docker compose --project-directory . -f nba/deployment/compose.yml up -d
```

The project-directory argument retains the root `.env` and existing Compose project
and volume naming after moving the configuration file.
