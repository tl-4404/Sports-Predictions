# Betting Aid App

## Project overview
A Flask web app (SportsAid) with two prediction pipelines:
- **MLB pitcher props** (strikeouts, pitcher outs): a plate-appearance-level Monte Carlo simulator produces P(over/under) per posted line, compares against the de-vigged multi-book market, and logs edges/bets/Kelly stakes. Real-money focus: every prediction is logged and graded next-day (ROI + CLV).
- **NBA moneylines**: a PyTorch model generates win probabilities and surfaces +EV picks with injury flags for star players.

## Structure
- `app.py` — Flask app, routes, DB helpers, `load_ev_picks()` (NBA) + `load_mlb_props()` (MLB)
- `run_pipeline.py` — hosting-agnostic MLB pipeline entrypoint (`--stage odds|predict|grade|all`)
- `nba/providers/odds.py` — fetches NBA moneyline odds from The Odds API, writes to SQLite
- `providers/` — shared cross-league Polymarket US, PrizePicks, and HTTP clients
- `sportsaid.db` — SQLite DB (events, prices, users, saved_picks, support_tickets, prop_lines, prop_predictions, prop_results)
- `mlb/` — all MLB code
  - `db.py` — props schema: append-only line history, predictions, graded results
  - `doctor.py` — preflight readiness check (`python run_pipeline.py --stage doctor`): deps, LLM key, StatsAPI, odds source, PrizePicks, accuracy caches
  - `grade.py` — next-day grading vs box scores: outcome, PnL, CLV vs closing consensus — reported per lane (sim vs llm vs judge)
  - `data/` — historical dataset assembly and heavyweight cache builders (`build_training`, backtest data, pitcher/lineup/park/weather/umpire/form features)
  - `training/` — PyTorch fitting, chronological calibration, and simulator backtesting
  - `prediction/` — PA simulator, betting math, context-factor lookups, daily prediction orchestration, and independent LLM/judge lanes
  - `fixtures/` — shipped stub prop lines; `cache/` (gitignored) — training data, park factors, custom stub lines
- `tests/test_pipeline_offline.py` — no-network end-to-end test: `python tests/test_pipeline_offline.py`
- `mlb/providers/` — MLB StatsAPI and MLB sportsbook odds
- `nba/providers/` — NBA Stats, NBA sportsbook odds, and ESPN NBA injuries
- `nba/` — all NBA model code
  - `data/` — historical training and chronological backtest dataset builders
  - `training/model.py` — trains the PyTorch model, saves to `nba/models/`
  - `prediction/features.py` — assembles daily game/team features via shared providers
  - `prediction/daily.py` — NN predictions and injury flags, writes `current_games_with_probs.csv`
  - Injury provider is at `nba/providers/espn.py`
  - `cache/` — cached team snapshots and training data
  - `models/` — saved model weights, scaler, feature columns
- `templates/` — Jinja2 HTML templates
- `static/style.css` — all CSS (dark theme, CSS variables)

## Quick start
`make setup` (creates `.venv`, installs serving deps, scaffolds `.env`, runs the preflight) → edit `.env` (at least `ANTHROPIC_API_KEY`; `ODDS_SOURCE=scraper` is the free default) → `make doctor` to confirm readiness → `make run` then `make web`. `make help` lists all targets; `make install-training` adds the heavyweight deps (torch/pybaseball) only needed to build caches or train NBA.

## Daily workflow (MLB)
0. `python run_pipeline.py --stage doctor` — preflight: confirms StatsAPI, the odds source, the LLM key, PrizePicks, and caches are ready before a live run
1. `python run_pipeline.py` — full run: snapshot prop lines → simulate slate → LLM lanes (if `ANTHROPIC_API_KEY` set) → PrizePicks lane → grade yesterday
2. `python run_pipeline.py --stage odds` — hourly line snapshots (line history feeds CLV)
3. `python run_pipeline.py --stage predict` — re-run after lineups are confirmed
4. `python run_pipeline.py --stage llm` — facts → Claude predictor → judge only
5. `python app.py` — Flask dev server; home page shows sportsbook props (judge>llm>sim per prop) and the PrizePicks best-lines board

## Daily workflow (NBA)
1. `python -m nba.providers.odds` — refresh odds in DB
2. `python -m nba.prediction.daily` — predictions + injury flags
3. `python app.py` — Flask dev server

## Environment (.env)
- `SECRET_KEY` — Flask session key (required in production)
- `ODDS_API_KEY` — The Odds API key; props need a paid plan (without it the stub fetcher runs)
- `ODDS_SOURCE` — odds-source toggle: `oddsapi` | `scraper` | `stub` (unset ⇒ oddsapi when `ODDS_API_KEY` is set, else stub). `scraper` uses the free multi-book scraper (`SportsbookScraperFetcher`, DraftKings + FanDuel by default so edges de-vig against a real consensus); pick books with `SCRAPE_BOOKS` and override drifting ids via `DK_BASE`/`DK_MLB_EVENTGROUP`/`DK_PITCHER_CATEGORIES` and `FD_BASE`/`FD_MLB_PAGE`
- `PRIZEPICKS_LIVE` — opt in to live PrizePicks scraping (else the offline stub runs); `PRIZEPICKS_MLB_LEAGUE_ID`/`PRIZEPICKS_URL` override the endpoint
- `ANTHROPIC_API_KEY` — enables the LLM lanes (skipped quietly when absent)
- `SPORTSAID_LLM_MODEL` — optional override, defaults to `claude-opus-4-8`
- `ADMIN_EMAILS` — comma-separated emails allowed to view `/admin/tickets`

## Key conventions
- Shared paths come from `mlb.paths`; moving a module must not change its cache, fixture, model, or database location
- Serving-time MLB code (`providers/`, `prediction/`, `grade.py`) must not import pybaseball/statcast/torch — those are training-only
- Prop tables are append-only; never UPDATE or DELETE line snapshots (CLV grading depends on history)
- Edges are always computed against the de-vigged multi-book consensus, never a single book's vig-included price; minimum edge to flag a bet is `MIN_EDGE` (3%)
- Team name mismatches between NBA API and Odds API are handled via `TEAM_NAME_FIXES` in `app.py`; MLB full-name → abbreviation mapping is `TEAM_ABBR` in `mlb/prediction/factors.py`
- Park, umpire, and pitcher-form K factors are read from `mlb/cache/*.json` via `mlb/prediction/factors.py` and `mlb/prediction/form.py`, always falling back to 1.0 / no-op (neutral) — build them with the modules under `mlb/data/` (heavyweight, run per season/weekly)
- The simulator scales K odds by `k_park * k_ump * k_form`; the pitcher K rate is CSW-blended (`form.blended_k_rate`) before simulating — all four collapse to neutral when their cache is missing, so serving never hard-fails
- Before trusting the simulator with real money, run `python -m mlb.training.backtest <dataset> [--odds archive.csv]` — the model must be calibrated (low ECE) and, with odds merged, show positive ROI and CLV
- Injury flags only apply to top-50 players by PPG (configurable via `get_top_players(n=50)`)
- Suppressed picks (Out/Doubtful star) are shown dimmed with red badge; warned picks (Questionable/Day-To-Day) show orange badge

## Instructions
- After completing a task that involves tool use, provide a quick summary of the work you've done.
- Make all independent tool calls in parallel. Only call tools sequentially when a later call depends on the result of an earlier one. Never use placeholders or guess missing parameters.
- Never speculate about code you have not opened. Read relevant files before answering questions about the codebase.
