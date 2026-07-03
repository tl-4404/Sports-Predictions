# Betting Aid App

## Project overview
A Flask web app (SportsAid) with two prediction pipelines:
- **MLB pitcher props** (strikeouts, pitcher outs): a plate-appearance-level Monte Carlo simulator produces P(over/under) per posted line, compares against the de-vigged multi-book market, and logs edges/bets/Kelly stakes. Real-money focus: every prediction is logged and graded next-day (ROI + CLV).
- **NBA moneylines**: a PyTorch model generates win probabilities and surfaces +EV picks with injury flags for star players.

## Structure
- `app.py` — Flask app, routes, DB helpers, `load_ev_picks()` (NBA) + `load_mlb_props()` (MLB)
- `run_pipeline.py` — hosting-agnostic MLB pipeline entrypoint (`--stage odds|predict|grade|all`)
- `odds_client.py` — fetches NBA moneyline odds from The Odds API, writes to SQLite
- `sportsaid.db` — SQLite DB (events, prices, users, saved_picks, support_tickets, prop_lines, prop_predictions, prop_results)
- `mlb/` — all MLB code
  - `db.py` — props schema: append-only line history, predictions, graded results
  - `betting_math.py` — odds conversion, multiplicative de-vig, fractional Kelly (quarter-Kelly, 2% cap)
  - `odds_props.py` — prop odds ingestion; `TheOddsAPIPropsFetcher` (requires paid plan + `ODDS_API_KEY`) or `StubPropsFetcher` (fixtures), auto-selected by env
  - `data_today.py` — serving-time data from the free MLB StatsAPI: schedule/probables/lineups, pitcher & batter per-PA rates (shrunk toward league average), box scores for grading
  - `simulator.py` — PA-level Monte Carlo: odds-ratio (log5) matchups, park/ump K factors, pitch-count leash, DP-aware outs tracking
  - `predict.py` — glue: slate → rates → sim → P(over) vs posted lines → edge/bet side/Kelly → `prop_predictions`
  - `llm.py` — Claude lanes: `facts` (web search, qualitative only), `predict` (independent Claude P(over) per line; sees the line, never the price or the sim output), `judge` (blend sim+llm, clamped to blend ± `JUDGE_BAND`, may veto bets). Rows share the sim run_id, sources `llm`/`judge`
  - `grade.py` — next-day grading vs box scores: outcome, PnL, CLV vs closing consensus — reported per lane (sim vs llm vs judge)
  - `build_training.py`, `data_pitcher.py`, `data_lineup.py`, `data_park.py`, `data_weather.py`, `model.py` — training-side (statcast/pybaseball; heavyweight, never imported at serving time)
  - `fixtures/` — shipped stub prop lines; `cache/` (gitignored) — training data, park factors, custom stub lines
- `tests/test_pipeline_offline.py` — no-network end-to-end test: `python tests/test_pipeline_offline.py`
- `nba/` — all NBA model code
  - `build_training.py` — fetches historical game data from NBA API, builds training CSVs
  - `model.py` — trains the PyTorch model, saves to `nba/models/`
  - `data_today.py` — fetches today's team stats from NBA API
  - `predict.py` — runs model on today's games, writes `current_games_with_probs.csv`
  - `data_injuries.py` — ESPN injury fetcher, filters to top-50 players by PPG
  - `cache/` — cached team snapshots and training data
  - `models/` — saved model weights, scaler, feature columns
- `templates/` — Jinja2 HTML templates
- `static/style.css` — all CSS (dark theme, CSS variables)

## Daily workflow (MLB)
1. `python run_pipeline.py` — full run: snapshot prop lines → simulate slate → LLM lanes (if `ANTHROPIC_API_KEY` set) → grade yesterday
2. `python run_pipeline.py --stage odds` — hourly line snapshots (line history feeds CLV)
3. `python run_pipeline.py --stage predict` — re-run after lineups are confirmed
4. `python run_pipeline.py --stage llm` — facts → Claude predictor → judge only
5. `python app.py` — Flask dev server; home page shows the latest run's props (judge row wins per prop, then llm, then sim)

## Daily workflow (NBA)
1. `python odds_client.py` — refresh odds in DB
2. `python nba/predict.py` — predictions + injury flags
3. `python app.py` — Flask dev server

## Environment (.env)
- `SECRET_KEY` — Flask session key (required in production)
- `ODDS_API_KEY` — The Odds API key; props need a paid plan (without it the stub fetcher runs)
- `ANTHROPIC_API_KEY` — enables the LLM lanes (skipped quietly when absent)
- `SPORTSAID_LLM_MODEL` — optional override, defaults to `claude-opus-4-8`
- `ADMIN_EMAILS` — comma-separated emails allowed to view `/admin/tickets`

## Key conventions
- Model paths use `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` for portability
- Serving-time MLB code (`data_today.py`, `simulator.py`, `predict.py`, `grade.py`) must not import pybaseball/statcast/torch — those are training-only
- Prop tables are append-only; never UPDATE or DELETE line snapshots (CLV grading depends on history)
- Edges are always computed against the de-vigged multi-book consensus, never a single book's vig-included price; minimum edge to flag a bet is `MIN_EDGE` (3%)
- Team name mismatches between NBA API and Odds API are handled via `TEAM_NAME_FIXES` in `app.py`; MLB full-name → abbreviation mapping is `TEAM_ABBR` in `mlb/predict.py`
- Injury flags only apply to top-50 players by PPG (configurable via `get_top_players(n=50)`)
- Suppressed picks (Out/Doubtful star) are shown dimmed with red badge; warned picks (Questionable/Day-To-Day) show orange badge

## Instructions
- After completing a task that involves tool use, provide a quick summary of the work you've done.
- Make all independent tool calls in parallel. Only call tools sequentially when a later call depends on the result of an earlier one. Never use placeholders or guess missing parameters.
- Never speculate about code you have not opened. Read relevant files before answering questions about the codebase.
