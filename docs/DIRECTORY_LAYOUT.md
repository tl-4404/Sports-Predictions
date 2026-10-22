# Project directory and NBA inventory

Cross-league clients live in root-level `providers/`. League-specific clients live
in `nba/providers/` and `mlb/providers/`, beside each league's data, training,
prediction, and grading logic. Run commands from the
repository root with `python -m ...` so shared package imports resolve correctly.

## Provider ownership

| Module | Current responsibility |
| --- | --- |
| `providers/polymarket.py` | Public US league/events and NBA moneyline bid/ask odds; read only |
| `nba/providers/odds.py` | The Odds API NBA moneylines, normalization, current-board SQLite writes |
| `mlb/providers/odds.py` | MLB pitcher-prop API/scraper/stub ingestion and line snapshots |
| `providers/prizepicks.py` | PrizePicks fetch/parser plus the existing MLB evaluation and grading lane |
| `nba/providers/nba_api.py` | NBA Stats transport for scoreboard, team/player stats and historical game logs |
| `mlb/providers/mlb_api.py` | MLB StatsAPI schedule, profiles, splits and actual results |
| `nba/providers/espn.py` | ESPN NBA injuries filtered by NBA Stats top scorers |
| `providers/http.py` | Shared scraper sessions and retries |

Provider modules are imported individually. Importing `providers` itself triggers
no network calls. NBA Stats SDK imports are lazy, allowing offline reconstruction
without installing `nba_api`.

The existing PrizePicks module still includes an MLB-only consumer lane, and the
odds modules include persistence. Moving them does not make those features generic
or create an NBA PrizePicks model. Historical Statcast/weather feature builders stay
under `mlb/data/`; they are heavyweight dataset jobs rather than shared live clients.

## NBA lifecycle

| Stage | Module | What exists |
| --- | --- | --- |
| Historical collection | `nba/data/build_training.py` | Legacy prior-day NBA Stats team snapshots; Base, Advanced, Misc, Four Factors and Opponent data |
| Historical reconstruction | `nba/data/build_backtest_data.py` | Season and rolling team features, rest and back-to-back indicators, home-win labels; emits pregame features before ingesting each day's outcomes |
| Training | `nba/training/model.py` | One PyTorch neural network: differential features, standardization, a 16-unit hidden layer, dropout and a win-probability output |
| Daily features | `nba/prediction/features.py` | Joins daily scoreboard games with team-stat snapshots through `nba.providers.nba_api` |
| Daily prediction | `nba/prediction/daily.py` | Loads NN/scaler/feature list; predicts home-win probability and adds ESPN injury notes |
| Web display | `app.py:load_ev_picks` | Matches the current prediction CSV to sportsbook moneylines, de-vigs each pair and surfaces probability edges with injury flags |

Saved NBA artifacts remain `nba/models/nba_model.pt`, `nba_scaler.joblib`, and
`feature_columns.csv`. Generated datasets remain under `nba/cache/`. Current output
files remain `nba/current_games_features.csv` and `nba/current_games_with_probs.csv`.
SQLite remains at the project root; a provider move must not create a second DB.

Only the NN is implemented for NBA. No NBA LightGBM or XGBoost trainer/artifacts
were found in the inspected checkout. The Polymarket client fetches quotes but is
not yet matched to NBA model probabilities or the web board. Automated trading
and an NBA agent forecast lane are not implemented.

The new backtest dataset and the legacy NN use different feature schemas. The new
builder is a dataset preparation step, not a completed evaluation of the saved NN.
Do not feed it directly into the legacy trainer: use the explicit pregame feature
allowlist from its metadata. See `docs/NBA_BACKTESTING.md` for validation guidance.

Known existing limitations include the hardcoded `2025-26` daily stats season, a
random training/validation split in the legacy NN, current-board odds refreshes
that overwrite previous NBA prices, and no append-only NBA prediction/grading
ledger. These behaviors are unchanged by directory organization.

## Commands

```bash
python -m providers.polymarket --moneylines --limit 3
python -m nba.providers.odds
python -m nba.prediction.features
python -m nba.prediction.daily
python -m nba.data.build_training
python -m nba.data.build_backtest_data --seasons 2025-26
python -m nba.training.model
python run_pipeline.py --stage odds
```

The trainer starts training when run; daily prediction calls live providers. The
organization checks use offline tests and import/compilation checks rather than
executing those state-changing workflows.

## Import examples

```python
from providers.polymarket import fetch_moneylines
from nba.providers import nba_api
from mlb.providers import mlb_api
from nba.providers.espn import get_team_injuries
from nba.prediction.features import build_feature_rows_for_date
```

Polymarket and PrizePicks are shared provider entrypoints. The Odds API adapters
remain league-specific because NBA moneylines and MLB pitcher props have different
parsing, schemas, and workflows. An ESPN source implemented only for NBA injuries
also belongs to NBA until we implement genuinely shared behavior.

The old `integrations.polymarket`, root `odds_client.py`, and flat NBA script
locations have been replaced. Use the module commands above.
