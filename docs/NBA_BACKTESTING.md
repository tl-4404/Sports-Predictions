# NBA data collection and backtesting

## Existing sources

| Script | Source and behavior | Historical limitation |
| --- | --- | --- |
| `nba/data/build_training.py` | NBA Stats via `nba_api`; five families of prior-day season-to-date team stats, game IDs and win labels; caches daily snapshots | Defaults to 2021-22 through 2023-24; many API calls; last-date resume can skip earlier failures |
| `nba/prediction/features.py` | NBA scoreboard plus team stats for a requested day | Hardcoded `SEASON = ["2025-26"]`; requests stats through game day, which can include results during historical replay |
| `nba/prediction/daily.py` | Saved PyTorch model, scaler and feature list; current ESPN injury flags | Overwrites `current_games_with_probs.csv`; no prediction archive or historical injury replay |
| `nba/providers/odds.py` | The Odds API current NBA moneylines | Selects one preferred book; deletes previous events/prices each refresh |

Current local inspection found seven rows in the current prediction CSV, model
artifacts, and six events/twelve prices in SQLite. There was no `nba/cache/`
directory. These are snapshots, not a historical prediction database.

## Build historical data

The new builder uses one team-game-log request per season, cached locally. It
derives season-to-date means, last-N-game means, win percentage, rest days and
back-to-back indicators. It emits features for a whole date before incorporating
that date's outcomes. Histories reset at the season boundary. Regular season only.

```bash
# Activate the project environment first. nba_api is required only for live pulls.
source .venv/bin/activate
pip install nba_api
python -m nba.data.build_backtest_data --seasons 2021-22 2022-23 2023-24 2024-25 2025-26

# Refresh an unfinished season rather than reuse its cached logs:
python -m nba.data.build_backtest_data --seasons 2026-27 --refresh

# Offline replay from a LeagueGameFinder team-log CSV:
python -m nba.data.build_backtest_data --seasons 2025-26 --raw-csv season_logs.csv
```

Output: `nba/cache/backtest/nba_games.csv` and `nba_games.meta.json`.
Raw logs: `nba/cache/game_logs/<season>_regular.csv`.
`--window` defaults to 10; `--min-games` defaults to 5 prior games per team.
Early games are excluded by default; `--min-games 0` retains missing-history
values, which require train-only imputation. Invalid/incomplete pairs are skipped.
Raw CSV input must contain exactly the selected season's NBA regular-season logs.
NBA Stats can time out or block requests; this builder retries three times and
retains successfully cached seasons for a later rerun.

These features are a **new baseline dataset**. They do not match the Advanced,
Miscellaneous, Four Factors and Opponent feature schema of the existing neural
network. Do not pass this CSV directly to `nba/training/model.py`: its broad numeric-column
selection would admit postgame point targets as features. A new trainer must use
only the explicit `feature_columns` allowlist in the metadata; `home_win` is the
label. Game IDs are strings, preserving leading zeroes.

## Honest evaluation

Use earlier seasons to train, the next season for tuning/calibration, and a later
untouched season for final evaluation. For example: train 2021-24, validate
2024-25, test 2025-26. For walk-forward evaluation, refit using only dates before
each test block. Fit scaling and imputation on training rows only. The existing
`nba/training/model.py` uses a random split and early stopping on its reported test set;
that score is not an untouched chronological backtest.

Measure Brier score, log loss and calibration as well as accuracy. Historical
reconstructions are newly generated forecasts, not proof that a prediction was
made before the game. Current saved weights cannot be assumed out of sample
without training-date provenance.

## Odds, injuries and prediction history still needed

To measure betting ROI, merge actual historical moneylines captured before each
game at the intended decision time. Match provider IDs using teams and scheduled
start times, keeping NBA and odds-provider IDs separately. De-vig both sides
within each book, then aggregate across books for consensus. Calculate realized
PnL using executable odds. CLV requires a separate last-pregame market snapshot.

The Odds API offers [historical snapshots](https://the-odds-api.com/historical-odds-data/)
from June 2020 on paid subscriptions. The current client has no historical
endpoint integration; do not run it expecting an odds archive.

Historical injury states are not supplied by this builder. Today's ESPN injury
report and today's top-scorer list cannot stand in for what was known years ago.
Evaluate the team-stat baseline first; add injuries only with timestamped records.

For future live runs, add append-only prediction, input, injury and multi-book
odds snapshots with UTC capture times, game start time, model hash/training cutoff,
run ID and subsequent results. This is separate from the existing current-board
tables. Until that exists, copy current snapshots before refreshing them if you
need to preserve them.

Offline verification:

```bash
python tests/test_nba_backtest_builder.py
```
