# NBA data collection and backtesting

## Existing sources

| Script | Source and behavior | Historical limitation |
| --- | --- | --- |
| `nba/data_fetching/build_training.py` | NBA Stats via `nba_api`; five families of prior-day season-to-date team stats, game IDs and win labels; caches daily snapshots | Defaults to 2021-22 through 2023-24; many API calls; last-date resume can skip earlier failures |
| `nba/models/prediction/features.py` | NBA scoreboard plus team stats for a requested day | Hardcoded `SEASON = ["2025-26"]`; requests stats through game day, which can include results during historical replay |
| `nba/models/prediction/daily.py` | Saved PyTorch model, scaler and feature list; current ESPN injury flags | Overwrites `current_games_with_probs.csv`; no prediction archive or historical injury replay |
| `nba/data_fetching/providers/odds.py` | The Odds API current NBA moneylines | Selects one preferred book; deletes previous events/prices each refresh |

Current prediction CSVs and the existing board tables are snapshots, not a
historical prediction database. The moneyline history below is stored separately.

## Collected moneyline dataset

Collected from NBA Stats on October 22, 2026, using five completed regular seasons:

| Season | Source games | Eligible pregame rows |
| --- | ---: | ---: |
| 2021-22 | 1,230 | 1,152 |
| 2022-23 | 1,230 | 1,151 |
| 2023-24 | 1,230 | 1,149 |
| 2024-25 | 1,230 | 1,152 |
| 2025-26 | 1,230 | 1,151 |
| Total | 6,150 | 5,755 |

Each source season has 30 teams with 82 games per team (2,460 team rows). The target
is `home_win`: 1 when the home team wins the full game, 0 when the away team wins.
Final results include overtime. The dataset covers regular-season games only;
contract rules must match this outcome when prices are joined.

Raw files are `nba/data/cache/game_logs/<season>_regular.csv`. The combined output is
`nba/data/cache/backtest/nba_games.csv`, with `nba_games.meta.json` alongside it. Both
locations are gitignored generated data and can be rebuilt from the command below.

There are 30 allowed differential features: prior game counts, rest days,
back-to-back status, win percentage, and season-to-date/last-10-game means for
points, shooting percentages, rebounds, assists, steals, blocks, turnovers, fouls,
and scoring margin. Each differential is home minus away. Home/away feature values
are retained for audit. Shooting features average game percentages rather than
recomputing a season percentage from total attempts.

The eligible dataset runs from October 29, 2021 through April 12, 2026. Its observed
home-win rate is 55.38%. Feature checks found no missing or infinite values among
the 30 allowed features. Every cutoff precedes the game date, and game IDs are
unique. The 395 excluded games all lack five prior games for one or both teams.
There are zero incomplete-result or unresolved-home/away exclusions.

Some source games repeat the same `AWAY @ HOME` matchup on both team rows instead
of using a separate `HOME vs. AWAY` string. The builder resolves the designated
teams using abbreviations and retains `shared_matchup` as an audit field (nine
eligible games). This flag does not establish venue or neutral-site status. Venue
enrichment is needed before treating designated home as actual home-court advantage
for those games. All completed games, including early warmup games, contribute to
later history after their date has passed.

The metadata records schema `nba_moneyline_v1`, the feature allowlist, audits,
source-data hashes and the output file hash. Postgame point totals are labels for
audit only and are explicitly excluded from the model feature list.

This dataset supplies basketball features and results. It does not yet contain
historical sportsbook or Polymarket odds, exact tipoff timestamps, or historical
injury states. It supports probability-model evaluation; betting ROI needs a
separate timestamped odds dataset. The historical source is a present-day download
of final game logs, not a point-in-time archive of provider revisions.

## Build historical data

The new builder uses one team-game-log request per season, cached locally. It
derives season-to-date means, last-N-game means, win percentage, rest days and
back-to-back indicators. It emits features for a whole date before incorporating
that date's outcomes. Histories reset at the season boundary. Regular season only.

```bash
# Activate the project environment first. nba_api is required only for live pulls.
source .venv/bin/activate
pip install nba_api
python -m nba.data_fetching.build_backtest_data --seasons 2021-22 2022-23 2023-24 2024-25 2025-26

# Refresh an unfinished season rather than reuse its cached logs:
python -m nba.data_fetching.build_backtest_data --seasons 2026-27 --refresh

# Offline replay from a LeagueGameFinder team-log CSV:
python -m nba.data_fetching.build_backtest_data --seasons 2025-26 --raw-csv season_logs.csv
```

Output: `nba/data/cache/backtest/nba_games.csv` and `nba_games.meta.json`.
Raw logs: `nba/data/cache/game_logs/<season>_regular.csv`.
`--window` defaults to 10; `--min-games` defaults to 5 prior games per team.
Early games are excluded by default; `--min-games 0` retains missing-history
values, which require train-only imputation. Invalid/incomplete pairs are skipped.
Raw CSV input must contain exactly the selected season's NBA regular-season logs.
NBA Stats can time out or block requests; this builder retries three times and
retains successfully cached seasons for a later rerun.

The metadata distinguishes insufficient-history exclusions from invalid result or
home/away pairs. Duplicate team/game rows, inconsistent scores, mismatched season
IDs (when supplied), duplicate output game IDs, and invalid feature cutoffs fail
validation. A cached rebuild requires no network calls.

These features are a **new baseline dataset**. They do not match the Advanced,
Miscellaneous, Four Factors and Opponent feature schema of the existing neural
network. Do not pass this CSV directly to `nba/models/training/model.py`: its broad numeric-column
selection would admit postgame point targets as features. A new trainer must use
only the explicit `feature_columns` allowlist in the metadata; `home_win` is the
label. Game IDs are strings, preserving leading zeroes.

## Enriched historical features

`python -m nba.data_fetching.enrich_moneylines` builds a separate enriched dataset for
2021-22 through 2025-26. It preserves `nba_games.csv` and writes
`nba/data/cache/backtest/nba_games_enriched.csv` plus `.meta.json`.

The enriched schema (`nba_moneyline_enriched_v2`) has 116 model features:
season-to-date and last-10-game home-minus-away means for 56 statistics, plus
four differentials for prior games, rest, back-to-back status and win percentage.
Home and away values are also retained for inspection.

The October 22, 2026 build contains all 5,755 eligible games listed above, with
the same game IDs and labels as the baseline. All 116 final features are finite;
all 50 saved NN stat concepts have a corresponding enriched season feature.
The 395 insufficient-history exclusions are unchanged. Offline tests cover
coverage mismatches, duplicates, date mismatches, missing values, zero-attempt
rates, feature selection and exclusion of current/future/same-day results.

Beyond the original box scores, the selected official team-game measures add:

- Advanced: offensive/defensive/net ratings, pace/estimated pace, possessions,
  true shooting, effective field goal percentage, assist and rebounding rates,
  turnover percentage, and PIE.
- Four Factors: free-throw attempt rate and opponent shooting, free-throw,
  turnover and offensive-rebounding rates.
- Miscellaneous: scoring from turnovers, second chances, fast breaks and the
  paint, with opponent equivalents.
- Opponent: shooting percentages, rebounds, assists, turnovers, steals, blocks,
  fouls and points. Base measures also supply fouls drawn.

Each season uses five whole-season `TeamGameLogs` measure requests, then joins
them to the existing base logs by game ID and team ID. This avoids requesting
five season-to-date snapshots for every historical date. Every source must have
exactly matching team/game coverage and dates; duplicate rows, missing fields,
unexpected missing values and infinities fail the build. Successful raw responses
are cached under `nba/data/cache/game_logs/measures/`, so reruns reuse completed pulls.
`--refresh` refetches all selected source tables; `--seasons`, `--window`,
`--min-games`, `--workers` and `--output` are configurable.

Features still use only games before the prediction date, never that day's
results, and reset each season. Percentages and ratings are arithmetic means of
per-game values, **not** pooled season ratios. An undefined shooting percentage
with zero attempts stays missing in the raw data and is excluded from its
historical mean; it is not replaced by a fabricated 0%. For example, Boston had
zero free-throw attempts on April 9, 2024. Final default-window features must all
be finite.

The metadata records the selected statistics, explicit feature allowlist,
source hashes, output hash, exclusions, audit and a concept-level mapping to
the saved neural network's 50 original features. Matching stat concepts does
not make the enriched data compatible with its saved scaler/weights: the new
aggregation and feature schema require a newly trained model. Use only metadata
`feature_columns` as inputs; postgame point totals remain audit labels.

This is team-level enrichment, not lineup/player/injury, travel, opponent-strength
adjustment or historical market enrichment. It does not yet supply betting ROI.

## Honest evaluation

Use earlier seasons to train, the next season for tuning/calibration, and a later
untouched season for final evaluation. For example: train 2021-24, validate
2024-25, test 2025-26. For walk-forward evaluation, refit using only dates before
each test block. Fit scaling and imputation on training rows only. The existing
`nba/models/training/model.py` uses a random split and early stopping on its reported test set;
that score is not an untouched chronological backtest.

For the collected dataset, that proposed split would contain 3,452 training games,
1,152 validation games, and 1,151 test games. No model was fitted as part of data
collection, and the later season should remain untouched during tuning.

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
python nba/tests/test_nba_backtest_builder.py
python nba/tests/test_nba_enrichment.py
```
