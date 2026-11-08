# Repository organization

NBA and MLB own their source, data, models, artifacts and tests. Cross-league
services live in `shared/`. The repository root contains developer configuration,
dependency lists, documentation and the two stable application launch wrappers.

## League layout

| Folder | Responsibility |
| --- | --- |
| `<league>/data/cache/` | Historical data, feature caches, backtest datasets and metadata; generated and ignored |
| `<league>/data/outputs/` | Current prediction/feature exports; generated and ignored |
| `<league>/data/fixtures/` | Small, versioned offline input samples; MLB samples shipped today |
| `<league>/data_fetching/` | Dataset assembly, feature-cache builders, league-specific PrizePicks configuration |
| `<league>/data_fetching/providers/` | League APIs and market-specific sportsbook normalization/persistence |
| `<league>/models/prediction/` | Daily inference, simulation, context lookups, research and forecast policies |
| `<league>/models/training/` | Fitting, calibration and backtesting source code |
| `<league>/saved_models/` | Serialized weights, scalers, feature columns and other trained artifacts; ignored |
| `<league>/tests/` | Offline league regression and integration suites |
| `<league>/docs/` | League-specific operating and model documentation |
| `<league>/paths.py` | Canonical dataset and artifact paths, independent of the current working directory |

NBA additionally owns `ledger/`, `forward.py`, `deployment/compose.yml` and optional
`requirements-forward.txt`. MLB owns `db.py`, `grade.py`, `doctor.py` and
`pipeline.py`. Historical MLB audit JSON lives in `mlb/data/ledgers/`; the files
retain their original content and provenance.

## Shared components

| Location | Responsibility |
| --- | --- |
| `shared/data_fetching/polymarket.py` | Existing public Polymarket event, quote and settlement client/CLI |
| `shared/data_fetching/sportsbooks.py` | The Odds API authentication and request transport used by NBA and MLB |
| `shared/data_fetching/prizepicks.py` | League-independent, paginated projection client and parser; caller supplies league ID and stat mapping |
| `shared/data_fetching/http.py` | Shared scraper sessions, headers and retry handling |
| `shared/betting/math.py` | Odds conversion, de-vigging, consensus and Kelly functions |
| `shared/data/sportsaid.db` | The single existing SQLite application database used by Flask and league adapters |
| `shared/paths.py` | Repository root and shared data/database paths |
| `shared/tests/` | Shared HTTP, Polymarket, PrizePicks, sportsbook transport and ownership checks |
| `shared/web/` | Flask implementation, templates, CSS and web-specific tests |

Shared transport and betting modules must not import NBA or MLB code. League
adapters can import shared code. The MLB PrizePicks consumer stays in
`mlb/models/prediction/prizepicks.py`, with mapping and fixture selection in
`mlb/data_fetching/prizepicks.py`; the shared client never imports its simulator
or tables. DraftKings/FanDuel pitcher-prop parsing remains an MLB adapter because
its market rules and source IDs are specific to MLB.

## Migration map

| Previous location | Current location |
| --- | --- |
| `providers/` | `shared/data_fetching/` (MLB PrizePicks consumer split out) |
| `<league>/providers/` | `<league>/data_fetching/providers/` |
| `<league>/data/*.py` | `<league>/data_fetching/*.py` |
| `<league>/training/` | `<league>/models/training/` |
| `<league>/prediction/` | `<league>/models/prediction/` |
| `mlb/prediction/betting_math.py` | `shared/betting/math.py` |
| `<league>/models/` artifacts | `<league>/saved_models/` |
| `<league>/cache/` | `<league>/data/cache/` |
| `mlb/fixtures/` | `mlb/data/fixtures/` |
| `nba/current_games*.csv` | `nba/data/outputs/current_games*.csv` |
| Root `sportsaid.db` | `shared/data/sportsaid.db` |
| Root `tests/` | `nba/tests/`, `mlb/tests/`, `shared/tests/`, `shared/web/tests/` |
| Root Flask implementation/assets | `shared/web/` |
| `compose.nba.yml` | `nba/deployment/compose.yml` |
| `requirements-nba-forward.txt` | `nba/requirements-forward.txt` |

Every cached dataset, fixture, ledger and saved model is moved without rewriting
its contents. Source paths come from league path modules or `shared.paths`.
The five previously versioned NBA checkpoint/scaler/feature-list and current CSV
artifacts have explicit ignore exceptions, preserving their version-control status.
Other generated data and saved models remain ignored.
`.env` stays at the repository root and relocated modules load it explicitly.
Update external scheduled commands using old dotted module names or artifact
paths to their equivalents above. Root `python app.py` and
`python run_pipeline.py` remain valid. The NBA PostgreSQL schema and DSN behavior
remain unchanged.

## Commands

```bash
python app.py
python run_pipeline.py --stage odds
python -m mlb.pipeline --stage predict
python -m shared.data_fetching.polymarket --moneylines --limit 3
python -m nba.data_fetching.providers.odds
python -m nba.models.prediction.features
python -m nba.models.prediction.daily
python -m nba.data_fetching.build_backtest_data --seasons 2025-26
python -m nba.data_fetching.enrich_moneylines
python -m nba.models.training.model
python -m nba.forward --help
python -m mlb.models.training.backtest --synthetic --flat-market
docker compose --project-directory . -f nba/deployment/compose.yml up -d
make install-dev
make test
make test-nba
make test-mlb
make test-shared
make test-web
```

The test runner executes each suite in a fresh process and blocks external network
connections. This supports the existing script-style end-to-end checks and
pytest/unittest suites without leaking their global test doubles into one another.
No training, live odds refresh, database reinitialization or trading is needed to
verify this organization.

See [NBA backtesting](../nba/docs/BACKTESTING.md) and
[MLB pipeline](../mlb/docs/PIPELINE.md) for model and evaluation details. The saved
legacy NBA NN and the reconstructed backtest dataset have different feature
schemas; a directory change does not make those schemas compatible. Existing model
behavior, calibration rules and live-data limitations are preserved.
