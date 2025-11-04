from datetime import datetime, timedelta
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Mock data you can replace with your models/APIs ---
LEAGUES = ["NFL", "NBA", "NHL", "MLB"]

def _mk_game(id_, league, home, away, kickoff_hours=3, home_odds=-135, away_odds=115, total=45.5, spread=-2.5):
    start = datetime.utcnow() + timedelta(hours=kickoff_hours)
    return {
        "id": id_,
        "league": league,
        "start_utc": start,
        "home": home,
        "away": away,
        "moneyline": {"home": home_odds, "away": away_odds},
        "spread": {"line": spread, "fav": home if spread < 0 else away},
        "total": total,
        "market": "Main",
    }

GAMES = [
    _mk_game(1, "NFL", "Eagles", "Cowboys", home_odds=-145, away_odds=125, total=47.5, spread=-3.0),
    _mk_game(2, "NFL", "Chiefs", "Bills", kickoff_hours=20, home_odds=-120, away_odds=105, total=52.0, spread=-1.5),
    _mk_game(3, "NBA", "Celtics", "Knicks", kickoff_hours=6, home_odds=-180, away_odds=150, total=222.5, spread=-4.5),
    _mk_game(4, "NHL", "Rangers", "Devils", kickoff_hours=10, home_odds=-115, away_odds=105, total=5.5, spread=-1.5),
    _mk_game(5, "MLB", "Dodgers", "Giants", kickoff_hours=30, home_odds=-135, away_odds=120, total=8.5, spread=-1.5),
]

MODEL_PICKS = [
    {
        "league": "NFL",
        "market": "Spread",
        "pick": "Eagles -3.0",
        "edge": "+4.1% EV",
        "confidence": 0.72,
        "reason": "Model favors PHI by 5.4 vs line -3.0",
        "game_id": 1,
    },
    {
        "league": "NBA",
        "market": "Total",
        "pick": "BOS/NYK Over 222.5",
        "edge": "+2.6% EV",
        "confidence": 0.64,
        "reason": "Pace-up spot; projection 226.3",
        "game_id": 3,
    },
]

def american_to_prob(american):
    """Convert American odds to implied probability (no vigorish removed)."""
    if american is None:
        return None
    american = float(american)
    if american > 0:
        return 100 / (american + 100)
    return -american / (-american + 100)

@app.context_processor
def inject_globals():
    return {"current_year": datetime.utcnow().year}

@app.template_filter("implied")
def implied_filter(odds):
    p = american_to_prob(odds)
    return f"{p*100:.1f}%" if p is not None else "—"

@app.template_filter("localtime")
def localtime_filter(dt):
    # In a real app, use pytz/zoneinfo and user’s tz; for demo show local-ish string.
    # (Jinja renders server time; front-end can re-format with Intl later.)
    return dt.strftime("%b %d, %I:%M %p UTC")

@app.route("/")
def index():
    league = request.args.get("league", "ALL").upper()
    q = request.args.get("q", "").strip().lower()

    filtered = GAMES
    if league != "ALL":
        filtered = [g for g in filtered if g["league"].upper() == league]

    if q:
        filtered = [
            g for g in filtered
            if q in g["home"].lower() or q in g["away"].lower()
        ]

    # Sort by start time
    filtered = sorted(filtered, key=lambda g: g["start_utc"])

    # Picks filtered to selected league (keep some content if none match)
    picks = [p for p in MODEL_PICKS if league in ("ALL", p["league"].upper())] or MODEL_PICKS[:1]

    return render_template(
        "index.html",
        leagues=LEAGUES,
        active_league=league,
        games=filtered,
        picks=picks,
        query=q,
    )

if __name__ == "__main__":
    app.run(debug=True)
