# Betting Aid App

## Project overview
A Flask web app (SportsAid) with two prediction pipelines:
- **MLB pitcher props** (strikeouts, pitcher outs): a plate-appearance-level Monte Carlo simulator produces P(over/under) per posted line, compares against the de-vigged multi-book market, and logs edges/bets/Kelly stakes. Real-money focus: every prediction is logged and graded next-day (ROI + CLV).
- **NBA moneylines**: a PyTorch model generates win probabilities and surfaces +EV picks with injury flags for star players.

## Structure
- `nba/` and `mlb/` own their league-specific code, data, saved artifacts and tests.
- Each league uses `data/` for artifacts (`cache/`, `outputs/`, `fixtures/`),
  `data_fetching/` for builders/providers, `models/prediction/` and
  `models/training/` for source, `saved_models/` for serialized artifacts,
  `tests/` for offline regression suites, and `docs/` for league design notes.
- NBA additionally has `ledger/`, `forward.py`, `deployment/compose.yml`, and
  `requirements-forward.txt`. MLB owns `pipeline.py`, `db.py`, `doctor.py`, and `grade.py`.
- `shared/data_fetching/` owns HTTP, Polymarket, sportsbook API transport and the
  league-independent PrizePicks parser. Shared modules do not import league code.
- `shared/betting/math.py` owns odds, de-vigging, consensus and Kelly utilities.
- `shared/web/` owns the Flask implementation, `templates/`, `static/`, and web tests.
- `shared/data/sportsaid.db` is the single SQLite application database.
- Root `app.py` and `run_pipeline.py` are compatibility launch wrappers.
- `scripts/run_tests.py` runs isolated, network-blocked suites; `make test`
  runs all, and `make test-nba|test-mlb|test-shared|test-web` select ownership.
- `docs/DIRECTORY_LAYOUT.md` contains the complete migration map and commands.

## Quick start
`make setup` (creates `.venv`, installs serving deps, scaffolds `.env`, runs the preflight) → edit `.env` (at least `ANTHROPIC_API_KEY`; `ODDS_SOURCE=scraper` is the free default) → `make doctor` to confirm readiness → `make run` then `make web`. `make help` lists all targets; `make install-training` adds the heavyweight deps (torch/pybaseball) only needed to build caches or train NBA.

## Daily workflow (MLB)
0. `python mlb/pipeline.py --stage doctor` — preflight: confirms StatsAPI, the odds source, the LLM key, PrizePicks, and caches are ready before a live run
1. `python mlb/pipeline.py` — full run: snapshot prop lines → simulate slate → LLM lanes (if `ANTHROPIC_API_KEY` set) → PrizePicks lane → grade yesterday
2. `python mlb/pipeline.py --stage odds` — hourly line snapshots (line history feeds CLV)
3. `python mlb/pipeline.py --stage predict` — re-run after lineups are confirmed
4. `python mlb/pipeline.py --stage llm` — facts → Codex predictor → judge only
5. `python shared/web/app.py` — Flask dev server; home page shows sportsbook props (judge>llm>sim per prop) and the PrizePicks best-lines board

## Daily workflow (NBA)
1. `python -m nba.data_fetching.providers.odds` — refresh odds in DB
2. `python -m nba.models.prediction.daily` — predictions + injury flags
3. `python shared/web/app.py` — Flask dev server

## Environment (.env)
- `SECRET_KEY` — Flask session key (required in production)
- `ODDS_API_KEY` — The Odds API key; props need a paid plan (without it the stub fetcher runs)
- `ODDS_SOURCE` — odds-source toggle: `oddsapi` | `scraper` | `stub` (unset ⇒ oddsapi when `ODDS_API_KEY` is set, else stub). `scraper` uses the free multi-book scraper (`SportsbookScraperFetcher`, DraftKings + FanDuel by default so edges de-vig against a real consensus); pick books with `SCRAPE_BOOKS` and override drifting ids via `DK_BASE`/`DK_MLB_EVENTGROUP`/`DK_PITCHER_CATEGORIES` and `FD_BASE`/`FD_MLB_PAGE`
- `PRIZEPICKS_LIVE` — opt in to live PrizePicks scraping (else the offline stub runs); `PRIZEPICKS_MLB_LEAGUE_ID`/`PRIZEPICKS_URL` override the endpoint
- `ANTHROPIC_API_KEY` — enables the LLM lanes (skipped quietly when absent)
- `SPORTSAID_LLM_MODEL` — optional override, defaults to `Codex-opus-4-8`
- `ADMIN_EMAILS` — comma-separated emails allowed to view `/admin/tickets`

## Key conventions
- Artifact paths come from `nba.paths`, `mlb.paths`, and `shared.paths`; never derive data locations from a module’s nesting or the current working directory. Use the approved locations in `docs/DIRECTORY_LAYOUT.md`
- Serving-time MLB code (`shared/data_fetching/`, `prediction/`, `grade.py`) must not import pybaseball/statcast/torch — those are training-only
- Prop tables are append-only; never UPDATE or DELETE line snapshots (CLV grading depends on history)
- Edges are always computed against the de-vigged multi-book consensus, never a single book's vig-included price; minimum edge to flag a bet is `MIN_EDGE` (3%)
- Team name mismatches between NBA API and Odds API are handled via `TEAM_NAME_FIXES` in `shared/web/app.py`; MLB full-name → abbreviation mapping is `TEAM_ABBR` in `mlb/models/prediction/factors.py`
- Park, umpire, and pitcher-form K factors are read from `mlb/data/cache/*.json` via `mlb/models/prediction/factors.py` and `mlb/models/prediction/form.py`, always falling back to 1.0 / no-op (neutral) — build them with the modules under `mlb/data_fetching/` (heavyweight, run per season/weekly)
- The simulator scales K odds by `k_park * k_ump * k_form`; the pitcher K rate is CSW-blended (`form.blended_k_rate`) before simulating — all four collapse to neutral when their cache is missing, so serving never hard-fails
- Before trusting the simulator with real money, run `python -m mlb.models.training.backtest <dataset> [--odds archive.csv]` — the model must be calibrated (low ECE) and, with odds merged, show positive ROI and CLV
- Injury flags only apply to top-50 players by PPG (configurable via `get_top_players(n=50)`)
- Suppressed picks (Out/Doubtful star) are shown dimmed with red badge; warned picks (Questionable/Day-To-Day) show orange badge

## Instructions
- After completing a task that involves tool use, provide a summary of the work you've done.
- Make all independent tool calls in parallel. Only call tools sequentially when a later call depends on the result of an earlier one. Never use placeholders or guess missing parameters.
- Never speculate about code you have not opened. Read relevant files before answering questions about the codebase.
