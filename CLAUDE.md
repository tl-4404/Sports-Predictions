# Betting Aid App

## Project overview
A Flask web app (SportsAid) that shows NBA moneyline odds, runs a PyTorch model to generate win probability predictions, and surfaces +EV picks with injury flags for star players.

## Structure
- `app.py` — Flask app, routes, DB helpers, `load_ev_picks()` logic
- `odds_client.py` — fetches NBA odds from The Odds API, writes to SQLite
- `sportsaid.db` — SQLite DB (events, prices, users, saved_picks, support_tickets)
- `nba/` — all NBA model code
  - `ML_data.py` — fetches historical game data from NBA API, builds training CSVs
  - `model.py` — trains the PyTorch model, saves to `nba/models/`
  - `fetch_current.py` — fetches today's team stats from NBA API
  - `predict_current.py` — runs model on today's games, writes `current_games_with_probs.csv`
  - `injuries.py` — ESPN injury fetcher, filters to top-50 players by PPG
  - `cache/` — cached team snapshots and training data
  - `models/` — saved model weights, scaler, feature columns
- `templates/` — Jinja2 HTML templates
- `static/style.css` — all CSS (dark theme, CSS variables)

## Daily workflow
1. Run `python odds_client.py` to refresh odds in DB
2. Run `python nba/predict_current.py` to generate predictions + injury flags
3. Run `python app.py` to start the Flask dev server

## Key conventions
- All NBA model paths use `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` for portability
- Team name mismatches between NBA API and Odds API are handled via `TEAM_NAME_FIXES` in `app.py`
- Injury flags only apply to top-50 players by PPG (configurable via `get_top_players(n=50)`)
- Suppressed picks (Out/Doubtful star) are shown dimmed with red badge; warned picks (Questionable/Day-To-Day) show orange badge

## Instructions
- After completing a task that involves tool use, provide a quick summary of the work you've done.
- Make all independent tool calls in parallel. Only call tools sequentially when a later call depends on the result of an earlier one. Never use placeholders or guess missing parameters.
- Never speculate about code you have not opened. Read relevant files before answering questions about the codebase.
