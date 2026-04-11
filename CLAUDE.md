# Betting Aid App

## Project overview
A Flask web app (SportsAid) that shows NBA and MLB odds, runs PyTorch models to generate predictions, and surfaces +EV picks with injury flags for star players.

## Structure
- `app.py` — Flask app, routes, DB helpers, `load_ev_picks()` logic
- `odds_client.py` — fetches NBA/MLB odds from The Odds API, writes to SQLite
- `sportsaid.db` — SQLite DB (events, prices, users, saved_picks, support_tickets)
- `nba/` — NBA win-probability model
  - `build_training.py` — fetches historical game data from NBA API, builds training CSVs
  - `model.py` — trains the PyTorch model, saves to `nba/models/`
  - `data_today.py` — fetches today's team stats from NBA API
  - `predict.py` — runs model on today's games, writes `current_games_with_probs.csv`
  - `data_injuries.py` — ESPN injury fetcher, filters to top-50 players by PPG
  - `cache/` — cached team snapshots and training data
  - `models/` — saved model weights (`nba_model.pt`), scaler, feature columns
- `mlb/` — MLB strikeout/outs prediction model
  - `build_training.py` — fetches Statcast data via pybaseball, builds `cache/training_data.csv`
  - `model.py` — two-headed PyTorch network predicting Ks and outs per game
  - `data_pitcher.py` — per-game pitcher features from Statcast (velo, pitch mix, K rate, whiff rate)
  - `data_lineup.py` — lineup-weighted batter features against pitcher handedness
  - `data_park.py` — park factor features from Statcast (HR, hit, hard-hit rates)
  - `data_weather.py` — weather features (temp, wind) keyed to stadium coordinates
  - `cache/` — cached park factors, weather, and training data
- `templates/` — Jinja2 HTML templates
- `static/style.css` — all CSS (dark theme, CSS variables)

## Daily workflow
1. Run `python odds_client.py` to refresh odds in DB
2. Run `python nba/predict.py` to generate NBA predictions + injury flags
3. Run `python app.py` to start the Flask dev server

## Key conventions
- All sport-module paths use `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` for portability
- Both `nba/` and `mlb/` follow the same file-naming structure: `build_training.py`, `model.py`, `data_*.py`, `predict.py`
- Team name mismatches between NBA API and Odds API are handled via `TEAM_NAME_FIXES` in `app.py`
- Injury flags only apply to top-50 players by PPG (configurable via `get_top_players(n=50)`)
- Suppressed picks (Out/Doubtful star) are shown dimmed with red badge; warned picks (Questionable/Day-To-Day) show orange badge
- MLB model uses a two-headed architecture: shared layers → separate Ks head and outs head

## Instructions
- After completing a task that involves tool use, provide a quick summary of the work you've done.
- Make all independent tool calls in parallel. Only call tools sequentially when a later call depends on the result of an earlier one. Never use placeholders or guess missing parameters.
- Never speculate about code you have not opened. Read relevant files before answering questions about the codebase.
