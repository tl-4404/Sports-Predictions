import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from flask import Flask, render_template, request, redirect, url_for, flash, g, abort
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def mlb_today():
    try:
        zone = ZoneInfo(os.getenv("MLB_SLATE_TIMEZONE", "America/New_York"))
    except ZoneInfoNotFoundError:
        zone = ZoneInfo("America/New_York")
    return datetime.now(zone).date()

app = Flask(__name__, template_folder="templates", static_folder="static")
# set SECRET_KEY in .env for production; the fallback is dev-only
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-insecure")

# emails allowed to view /admin/tickets (comma-separated in .env)
ADMIN_EMAILS = {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}

# minimum edge vs de-vigged market before a pick is surfaced
MIN_EDGE = 0.03

DB_PATH = Path(
    os.getenv(
        "SPORTSAID_DB_PATH",
        str(Path(__file__).resolve().parent / "sportsaid.db"),
    )
).expanduser()
NBA_PROBS_PATH = Path(__file__).resolve().parent / "nba" / "current_games_with_probs.csv"

NBA_ABBR = {
    "ATL": "Atlanta Hawks",       "BOS": "Boston Celtics",        "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",   "CHI": "Chicago Bulls",         "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",    "DEN": "Denver Nuggets",        "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors","HOU": "Houston Rockets",      "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers","LAL": "Los Angeles Lakers",    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",          "MIL": "Milwaukee Bucks",       "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans","NYK": "New York Knicks",       "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",       "PHI": "Philadelphia 76ers",    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers","SAC": "Sacramento Kings",    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",     "UTA": "Utah Jazz",             "WAS": "Washington Wizards",
}

# sql helpers
def get_db():
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db

# close db when not in use
@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()
    
# initializes db
def init_db():
    db = get_db()

    # users
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)

    # support tickets
    db.execute("""
        CREATE TABLE IF NOT EXISTS support_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            subject TEXT NOT NULL,
            category TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open'
        );
    """)

    # user picks
    # connects via users(id)
    db.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS saved_picks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        game_id INTEGER,
        league TEXT,
        market TEXT NOT NULL,          
        selection TEXT NOT NULL,        
        odds_american INTEGER,          
        stake_cents INTEGER DEFAULT 0,  
        notes TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_saved_picks_user ON saved_picks(user_id);
    """)

    # games + odds
    db.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS events (
      event_id TEXT PRIMARY KEY,
      league   TEXT NOT NULL,
      commence_time TEXT,
      home     TEXT NOT NULL,
      away     TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS prices (
      event_id TEXT NOT NULL,
      book     TEXT NOT NULL,
      market   TEXT NOT NULL,     
      outcome  TEXT NOT NULL,     
      line     REAL,           
      odds_american INTEGER NOT NULL,
      as_of    TEXT NOT NULL,
      PRIMARY KEY (event_id, book, market, outcome),
      FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_prices_event ON prices(event_id);
    CREATE INDEX IF NOT EXISTS idx_prices_market ON prices(market);
    """)

    db.commit()

# log-in 
login_manager = LoginManager(app)
login_manager.login_view = "login"

# User object
class User(UserMixin):
    def __init__(self, row):
        self.id = row["id"]
        self.email = row["email"]
        self.password_hash = row["password_hash"]

# loads user
@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return User(row) if row else None

# fix name mismatch
TEAM_NAME_FIXES = {
    "LA Clippers": "Los Angeles Clippers",
}

# get injurys
def _parse_injury_note(note):
    if not note or not note.strip():
        return []
    result = []
    for entry in note.split(";"):
        entry = entry.strip()
        if ":" in entry:
            name, status = entry.split(":", 1)
            result.append({"name": name.strip(), "status": status.strip()})
    return result

# de-vig a two-way market (multiplicative): returns fair home probability
def devig_pair(home_odds, away_odds):
    p_h = american_to_prob(home_odds)
    p_a = american_to_prob(away_odds)
    if p_h is None or p_a is None or (p_h + p_a) <= 0:
        return None
    return p_h / (p_h + p_a)

# get all +ev picks based on model
def load_ev_picks(db):
    if not NBA_PROBS_PATH.exists():
        return []
    picks = []
    with open(NBA_PROBS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            home_full = TEAM_NAME_FIXES.get(row.get("home_TEAM_NAME", "").strip(), row.get("home_TEAM_NAME", "").strip())
            away_full = TEAM_NAME_FIXES.get(row.get("away_TEAM_NAME", "").strip(), row.get("away_TEAM_NAME", "").strip())
            if not home_full or not away_full:
                continue
            try:
                home_win_prob = float(row["home_win_prob"])
            except (KeyError, ValueError):
                continue
            ml_rows = db.execute(
                """SELECT p.outcome, p.odds_american, e.commence_time
                   FROM prices p
                   JOIN events e ON e.event_id = p.event_id
                   WHERE e.home = ? AND e.away = ? AND p.market = 'h2h'""",
                (home_full, away_full)
            ).fetchall()
            if not ml_rows:
                continue
            commence_time = ml_rows[0]["commence_time"]
            ml_map = {m["outcome"]: m["odds_american"] for m in ml_rows}
            home_injuries = _parse_injury_note(row.get("home_injury_note", ""))
            away_injuries = _parse_injury_note(row.get("away_injury_note", ""))

            # de-vigged market probability — comparing the model to raw
            # vig-included prices manufactures phantom edges
            fair_home = devig_pair(ml_map.get("Home"), ml_map.get("Away"))
            for side, model_prob in (("Home", home_win_prob), ("Away", 1 - home_win_prob)):
                odds = ml_map.get(side)
                if odds is None:
                    continue
                implied = american_to_prob(odds)
                fair = fair_home if side == "Home" else (1 - fair_home if fair_home is not None else None)
                edge = model_prob - (fair if fair is not None else implied)
                if edge >= MIN_EDGE:
                    side_injuries = home_injuries if side == "Home" else away_injuries
                    suppressed = any(p["status"] in ("Out", "Doubtful") for p in side_injuries)
                    warnings = [p for p in side_injuries if p["status"] == "Questionable"]
                    picks.append({
                        "home": home_full,
                        "away": away_full,
                        "team": home_full if side == "Home" else away_full,
                        "side": side,
                        "model_prob": model_prob,
                        "implied_prob": implied,
                        "edge": edge,
                        "odds": odds,
                        "commence_time": commence_time,
                        "suppressed": suppressed,
                        "suppressed_players": [p for p in side_injuries if p["status"] in ("Out", "Doubtful")],
                        "warnings": warnings,
                    })
    picks.sort(key=lambda x: x["edge"], reverse=True)
    return picks

# latest MLB prop predictions for a date (most recent pipeline run only).
# Per prop, the simulator is canonical unless the adversarial judge has
# reviewed it. The analysis-only LLM lane must never hide/create a decision.
def load_mlb_props(db, game_date=None, bets_only=False):
    game_date = game_date or mlb_today().isoformat()
    try:
        run = db.execute(
            "SELECT MAX(run_id) r FROM prop_predictions WHERE game_date = ?",
            (game_date,),
        ).fetchone()
    except sqlite3.OperationalError:
        return []  # prop tables not created yet — pipeline never ran
    if not run or not run["r"]:
        return []
    rows = db.execute(
        """SELECT * FROM prop_predictions
           WHERE game_date = ? AND run_id = ?""",
        (game_date, run["r"]),
    ).fetchall()

    rank = {"llm": 0, "sim": 1, "judge": 2}
    best = {}
    sim_means = {}
    lane_probs = {}   # key -> {source: p_over} for the three-way comparison
    for r in rows:
        key = (r["player_name"], r["market"], r["line"])
        lane_probs.setdefault(key, {})[r["source"]] = r["p_over"]
        if r["source"] == "sim" and r["mean"] is not None:
            sim_means[key] = r["mean"]
        cur = best.get(key)
        if cur is None or rank.get(r["source"], -1) > rank.get(cur["source"], -1):
            best[key] = r

    props = []
    for key, r in best.items():
        if bets_only and not r["bet_side"]:
            continue
        d = dict(r)
        d["market_label"] = "Strikeouts" if r["market"] == "pitcher_strikeouts" else "Pitcher Outs"
        d["edge"] = max(r["edge_over"] or 0, r["edge_under"] or 0) if r["edge_over"] is not None else None
        d["model_edge"] = (
            abs(r["p_over"] - r["market_p_over"])
            if r["market_p_over"] is not None
            else None
        )
        d["decision_p_over"] = (
            r["decision_p_over"]
            if "decision_p_over" in r.keys()
            and r["decision_p_over"] is not None
            else r["p_over"]
        )
        d["lane"] = r["source"]                       # which lane made the call
        d["lanes"] = lane_probs.get(key, {})          # sim/llm/judge P(over) side by side
        if d.get("mean") is None:
            d["mean"] = sim_means.get(key)            # judge/llm rows show the sim mean
        # judge/llm rationale is stored as JSON — pull the human text out
        if d.get("rationale"):
            try:
                meta = json.loads(d["rationale"])
                d["rationale_text"] = meta.get("rationale") or ""
                d["judge_action"] = meta.get("action")
                d["hold_reasons"] = meta.get("hold_reasons") or []
            except (ValueError, TypeError):
                d["rationale_text"] = d["rationale"]
        props.append(d)
    props.sort(key=lambda x: x["edge"] if x["edge"] is not None else -1, reverse=True)
    return props


# PrizePicks is a separate surface: pick'em projections (no two-way odds) scored
# off the simulator distribution vs the parlay break-even. Kept apart from the
# sportsbook board above so the two are never conflated.
def load_prizepicks_board(db, game_date=None, playable_only=False):
    game_date = game_date or mlb_today().isoformat()
    try:
        run = db.execute(
            "SELECT MAX(run_id) r FROM prizepicks_picks WHERE game_date = ?",
            (game_date,),
        ).fetchone()
    except sqlite3.OperationalError:
        return []  # PrizePicks lane never ran
    if not run or not run["r"]:
        return []
    rows = db.execute(
        """SELECT * FROM prizepicks_picks
           WHERE game_date = ? AND run_id = ?
           ORDER BY edge DESC""",
        (game_date, run["r"]),
    ).fetchall()
    picks = []
    for r in rows:
        if playable_only and not r["playable"]:
            continue
        d = dict(r)
        d["market_label"] = ("Strikeouts" if r["market"] == "pitcher_strikeouts"
                             else "Pitcher Outs")
        picks.append(d)
    return picks


def load_prizepicks_cards(db, game_date=None, playable_only=True):
    """Actual entry-level decisions with the configured displayed payout."""
    game_date = game_date or mlb_today().isoformat()
    try:
        run = db.execute(
            "SELECT MAX(run_id) r FROM prizepicks_cards WHERE game_date = ?",
            (game_date,),
        ).fetchone()
        if not run or not run["r"]:
            return []
        clause = "AND c.playable = 1" if playable_only else ""
        rows = db.execute(
            f"""SELECT c.*, p1.player_name AS player_1,
                       p1.market AS market_1, p1.line AS line_1,
                       p1.pick_side AS side_1,
                       p2.player_name AS player_2,
                       p2.market AS market_2, p2.line AS line_2,
                       p2.pick_side AS side_2
                FROM prizepicks_cards c
                JOIN prizepicks_picks p1 ON p1.id = c.pick_1_id
                JOIN prizepicks_picks p2 ON p2.id = c.pick_2_id
                WHERE c.game_date = ? AND c.run_id = ? {clause}
                ORDER BY c.expected_value DESC""",
            (game_date, run["r"]),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(row) for row in rows]


# per-lane real-money scorecard over a trailing window — which lane to trust.
# ROI is profit (already at each bet's logged Kelly stake) divided by total
# staked; CLV is the honest "did we beat the close" signal.
def load_lane_scoreboard(db, days=14):
    since = (mlb_today() - timedelta(days=days)).isoformat()
    try:
        rows = db.execute(
            """WITH ranked AS (
                 SELECT p.*,
                        ROW_NUMBER() OVER (
                          PARTITION BY p.game_date, COALESCE(p.game_pk,-1),
                                       LOWER(p.player_name), p.market, p.line,
                                       p.source, COALESCE(p.policy_version,'legacy')
                          ORDER BY p.created_at DESC, p.id DESC
                        ) AS recency_rank
                 FROM prop_predictions p
                 WHERE p.game_date >= ?
                   AND COALESCE(p.policy_version,'legacy') = 'sportsaid-v5.0-merged-audit'
               )
               SELECT p.source,
                      COUNT(*)                                    AS graded,
                      SUM(CASE WHEN p.bet_side IS NOT NULL THEN 1 ELSE 0 END) AS bets,
                      SUM(COALESCE(r.bet_won, 0))                 AS wins,
                      SUM(COALESCE(r.profit_units, 0))            AS pnl,
                      SUM(CASE WHEN p.bet_side IS NOT NULL THEN COALESCE(p.kelly_frac,0) ELSE 0 END) AS staked,
                      AVG(r.clv)                                  AS avg_clv,
                      AVG(CASE WHEN r.clv IS NOT NULL AND r.clv > 0 THEN 1.0
                               WHEN r.clv IS NOT NULL THEN 0.0 END) AS clv_beat_rate
               FROM ranked p
               JOIN prop_results r ON r.prediction_id = p.id
               WHERE p.recency_rank = 1
               GROUP BY p.source""",
            (since,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []  # sportsbook lanes never ran — still report PrizePicks below
    order = {"judge": 0, "sim": 1, "llm": 2, "prizepicks": 3}
    board = []
    for r in rows:
        bets = r["bets"] or 0
        board.append({
            "source": r["source"],
            "bets": bets,
            "wins": r["wins"] or 0,
            "hit_rate": (r["wins"] / bets) if bets else None,
            "pnl": r["pnl"] or 0.0,
            "roi": (r["pnl"] / r["staked"]) if (r["staked"] or 0) > 0 else None,
            "avg_clv": r["avg_clv"],
            "clv_beat_rate": r["clv_beat_rate"],
        })
    pp_row = _prizepicks_lane_row(db, since)
    if pp_row:
        board.append(pp_row)
    board.sort(key=lambda x: order.get(x["source"], 9))
    return board


# PrizePicks is scored at the actual entry/card level. Per-leg pseudo-ROI would
# ignore ties and the displayed multiplier, so it is deliberately not used.
def _prizepicks_lane_row(db, since):
    try:
        rows = db.execute(
            """SELECT c.id, r.outcome, r.profit_units
               FROM prizepicks_cards c
               JOIN prizepicks_card_results r ON r.card_id = c.id
               WHERE c.game_date >= ? AND c.playable = 1""",
            (since,),
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    if not rows:
        return None
    n = len(rows)
    wins = sum(1 for r in rows if r["outcome"] == "win")
    pnl = sum(float(r["profit_units"] or 0.0) for r in rows)
    return {
        "source": "prizepicks",
        "bets": n,
        "wins": wins,
        "hit_rate": wins / n,
        "pnl": pnl,
        "roi": pnl / n,
        "avg_clv": None,         # no two-way market / closing line
        "clv_beat_rate": None,
    }

# card maker helpers (converts american odds to probability)
def american_to_prob(american):
    """Convert American odds to implied probability (no vigorish removed)."""
    if american is None:
        return None
    american = float(american)
    if american > 0:
        return 100 / (american + 100)
    return -american / (-american + 100)

# globals (return currect year)
@app.context_processor
def inject_globals():
    return {"current_year": datetime.utcnow().year}

# prob filter
@app.template_filter("implied")
def implied_filter(odds):
    p = american_to_prob(odds)
    return f"{p*100:.1f}%" if p is not None else "—"

# local time filter
@app.template_filter("localtime")
def localtime_filter(ts):
    if not ts:
        return "TBD"
    if isinstance(ts, str):
        ts = ts.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            return ts  
    else:
        dt = ts
    return dt.strftime("%b %d, %I:%M %p UTC")

# routes

# default page
@app.route("/")
def index():
    db = get_db()

    leagues = ["NBA", "MLB"]
    active_league = request.args.get("league", "NBA")
    if active_league not in leagues:
        active_league = "NBA"
    query = (request.args.get("q") or "").strip()

    # Pull events (order by start time)
    events = db.execute("""
        SELECT event_id, league, commence_time, home, away
        FROM events
        WHERE league = 'NBA'
        ORDER BY IFNULL(commence_time, '') ASC, home ASC
    """).fetchall()
    if query:
        ql = query.lower()
        events = [e for e in events
                  if ql in (e["home"] or "").lower() or ql in (e["away"] or "").lower()]

    games = []
    preferred = ("DraftKings", "FanDuel", "Caesars", "BetMGM")

    # get odds/organize per game
    for ev in events:
        if not ev["home"] or not ev["away"]:
            continue

        # organize each game
        row = {
            "id": ev["event_id"],
            "league": ev["league"],
            "commence_time": ev["commence_time"],
            "home": ev["home"],
            "away": ev["away"],
            "book": None,
            "moneyline": {"home": None, "away": None},
            "spread": None,
            "total": None,
        }

        # get ML for specific book
        def moneyline_for(book):
            return db.execute(
                """SELECT outcome, odds_american
                   FROM prices
                   WHERE event_id=? AND book=? AND market='h2h'""",
                (ev["event_id"], book)
            ).fetchall()

        ml = []
        # get ML for all preferred books
        for b in preferred:
            ml = moneyline_for(b)
            if ml:
                row["book"] = b
                break
        # if no preferred select random book
        if not ml:
            any_book = db.execute(
                "SELECT DISTINCT book FROM prices WHERE event_id=? LIMIT 1",
                (ev["event_id"],)
            ).fetchone()
            if any_book:
                row["book"] = any_book["book"]
                ml = moneyline_for(row["book"])
        # get all odds in ML
        if ml:
            ml_map = {m["outcome"]: m["odds_american"] for m in ml}
            row["moneyline"]["home"] = ml_map.get("Home")
            row["moneyline"]["away"] = ml_map.get("Away")
        games.append(row)

    ev_picks = load_ev_picks(db)
    mlb_props = load_mlb_props(db)
    prizepicks_picks = load_prizepicks_board(db)
    prizepicks_cards = load_prizepicks_cards(db)
    mlb_scoreboard = load_lane_scoreboard(db)
    return render_template(
        "index.html",
        games=games,
        ev_picks=ev_picks,
        mlb_props=mlb_props,
        prizepicks_picks=prizepicks_picks,
        prizepicks_cards=prizepicks_cards,
        mlb_scoreboard=mlb_scoreboard,
        leagues=leagues,
        active_league=active_league,
        query=query,
    )

# register page
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        # missing email or password
        if not email or not password:
            flash("Email and password are required.", "error")
            return redirect(url_for("register"))

        db = get_db()
        # check if email is in use
        existing = db.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            flash("Email already registered.", "error")
            return redirect(url_for("register"))
        # add user to db
        db.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, generate_password_hash(password), datetime.utcnow().isoformat())
        )
        db.commit()
        # log user in
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        login_user(User(user))
        return redirect(url_for("index"))

    return render_template("register.html")

# login page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # get passed in fields
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        db = get_db()
        row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        # check if email & password match
        if row and check_password_hash(row["password_hash"], password):
            login_user(User(row), remember=True)
            return redirect(url_for("index"))
        # if info does not match, flash error
        flash("Invalid login.", "error")
        return redirect(url_for("login"))

    return render_template("login.html")

# logout page
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

# saved picks (card) page
@app.route("/my-card")
@login_required
def my_card():
    db = get_db()
    # query all saved picks for user
    picks = db.execute("""
        SELECT id, league, market, selection, odds_american, stake_cents, created_at
        FROM saved_picks
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (current_user.id,)).fetchall()
    return render_template("my_card.html", picks=picks)

# add a pick to card
@app.route("/picks/add", methods=["POST"])
@login_required
def picks_add():
    # get all pick info
    game_id     = request.form.get("game_id", type=int)
    league      = (request.form.get("league") or "").strip()
    market      = (request.form.get("market") or "").strip()   
    side        = (request.form.get("side") or "").strip()       
    team_home   = (request.form.get("team_home") or "").strip()
    team_away   = (request.form.get("team_away") or "").strip()
    line_str    = (request.form.get("line") or "").strip()        
    odds        = request.form.get("odds", type=int)     
    stake_cents = request.form.get("stake_cents", type=int) or 0
    notes       = request.form.get("notes", "")
    # check for valid market & side
    if not market or not side:
        flash("Pick needs a market and side.", "error")
        return redirect(url_for("index"))
    # get selection
    selection = ""
    if market.lower() == "moneyline":
        team = team_home if side.lower() == "home" else team_away
        selection = f"{team} ML"
    elif market.lower() == "spread":
        team = team_home if side.lower() == "home" else team_away
        selection = f"{team} {line_str}" if line_str else f"{team} (Spread)"
    elif market.lower() == "total":
        label = "Over" if side.lower() == "over" else "Under"
        selection = f"{label} {line_str}" if line_str else label
    else:
        selection = f"{market} · {side}"
    # save pick (add to user's card)
    db = get_db()
    db.execute("""
        INSERT INTO saved_picks (user_id, game_id, league, market, selection, odds_american, stake_cents, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (current_user.id, game_id, league, market, selection, odds, stake_cents, notes, datetime.utcnow().isoformat()))
    db.commit()

    flash(f"Added: {market} - {selection} @ {odds}", "success")
    return redirect(url_for("my_card"))

# delete a saved pick (from my-card)
# NOTE: this was previously (mis)registered at /support with invalid SQL,
# which both broke deletion and shadowed the real support page.
@app.route("/picks/delete", methods=["POST"])
@login_required
def picks_delete():
    pick_id = request.form.get("pick_id", type=int)
    if not pick_id:
        flash("Missing pick id.", "error")
        return redirect(url_for("my_card"))
    db = get_db()
    cur = db.execute(
        "DELETE FROM saved_picks WHERE id = ? AND user_id = ?",
        (pick_id, current_user.id),
    )
    db.commit()
    if cur.rowcount:
        flash("Pick deleted.", "success")
    else:
        flash("Pick not found.", "error")
    return redirect(url_for("my_card"))

# support page
@app.route("/support", methods=["GET", "POST"])
@login_required
def support():
    if request.method == "POST":
        # get user info
        user_email = (current_user.email if getattr(current_user, "is_authenticated", False)
                                    else request.form.get("email", "").strip().lower())
        subject = request.form.get("subject", "").strip()
        category = request.form.get("category", "General").strip()
        message = request.form.get("message", "").strip()
        honeypot = request.form.get("website")
        # bot/spam detection
        if honeypot: 
            flash("Submission failed. Please try again. Bot???", "error")
            return redirect(url_for("support"))
        # check for empty fields
        if not user_email or not subject or not message:
            flash("Email, subject, and message are required.", "error")
            return redirect(url_for("support"))
        # save ticket
        db = get_db()
        db.execute(
            """
            INSERT INTO support_tickets (user_email, subject, category, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_email, subject, category, message, datetime.utcnow().isoformat())
        )
        db.commit()

        flash("Thanks! Your ticket has been received. We'll email you soon.", "success")
        return redirect(url_for("support_success"))

    return render_template("support.html")

# ticket success page
@app.route("/support/success")
def support_success():
    return render_template("support_success.html")

# admin page for support
@app.route("/admin/tickets")
@login_required
def admin_tickets():
    # only allow-listed admins may read tickets (ADMIN_EMAILS in .env)
    if current_user.email.lower() not in ADMIN_EMAILS:
        abort(403)
    db = get_db()
    rows = db.execute("SELECT * FROM support_tickets ORDER BY id DESC LIMIT 100").fetchall()
    return render_template("admin_tickets.html", rows=rows)

if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(debug=True)
