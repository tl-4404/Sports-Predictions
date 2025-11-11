from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, g
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "12345"

DB_PATH = Path(__file__).resolve().parent / "sportsaid.db"

# sql helpers
def get_db():
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()

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

class User(UserMixin):
    def __init__(self, row):
        self.id = row["id"]
        self.email = row["email"]
        self.password_hash = row["password_hash"]

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return User(row) if row else None

# card maker helpers
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
@app.route("/")
def index():
    db = get_db()

    # Pull events with the correct time column name
    events = db.execute("""
        SELECT event_id, league, commence_time, home, away
        FROM events
        WHERE league = 'NBA'
        ORDER BY IFNULL(commence_time, '') ASC, home ASC
    """).fetchall()

    games = []
    preferred = ("DraftKings", "FanDuel", "Caesars", "BetMGM")

    for ev in events:
        if not ev["home"] or not ev["away"]:
            continue

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

        def moneyline_for(book):
            return db.execute(
                """SELECT outcome, odds_american
                   FROM prices
                   WHERE event_id=? AND book=? AND market='h2h'""",
                (ev["event_id"], book)
            ).fetchall()

        ml = []
        for b in preferred:
            ml = moneyline_for(b)
            if ml:
                row["book"] = b
                break
        if not ml:
            any_book = db.execute(
                "SELECT DISTINCT book FROM prices WHERE event_id=? LIMIT 1",
                (ev["event_id"],)
            ).fetchone()
            if any_book:
                row["book"] = any_book["book"]
                ml = moneyline_for(row["book"])

        if ml:
            ml_map = {m["outcome"]: m["odds_american"] for m in ml}
            row["moneyline"]["home"] = ml_map.get("Home")
            row["moneyline"]["away"] = ml_map.get("Away")
        games.append(row)

    return render_template("index.html", games=games)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if not email or not password:
            flash("Email and password are required.", "error")
            return redirect(url_for("register"))

        db = get_db()
        existing = db.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            flash("Email already registered.", "error")
            return redirect(url_for("register"))

        db.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, generate_password_hash(password), datetime.utcnow().isoformat())
        )
        db.commit()

        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        login_user(User(user))
        return redirect(url_for("index"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        db = get_db()
        row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row and check_password_hash(row["password_hash"], password):
            login_user(User(row), remember=True)
            return redirect(url_for("index"))

        flash("Invalid login.", "error")
        return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/my-card")
@login_required
def my_card():
    db = get_db()
    picks = db.execute("""
        SELECT id, league, market, selection, odds_american, stake_cents, created_at
        FROM saved_picks
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (current_user.id,)).fetchall()
    return render_template("my_card.html", picks=picks)

@app.route("/picks/add", methods=["POST"])
@login_required
def picks_add():
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

    if not market or not side:
        flash("Pick needs a market and side.", "error")
        return redirect(url_for("index"))

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

    db = get_db()
    db.execute("""
        INSERT INTO saved_picks (user_id, game_id, league, market, selection, odds_american, stake_cents, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (current_user.id, game_id, league, market, selection, odds, stake_cents, notes, datetime.utcnow().isoformat()))
    db.commit()

    flash(f"Added: {market} – {selection} @ {odds}", "success")
    return redirect(url_for("my_card"))

@app.route("/support", methods=["GET", "POST"])
@login_required
def support():
    if request.method == "POST":
        user_email = (current_user.email if getattr(current_user, "is_authenticated", False)
                                    else request.form.get("email", "").strip().lower())
        subject = request.form.get("subject", "").strip()
        category = request.form.get("category", "General").strip()
        message = request.form.get("message", "").strip()
        honeypot = request.form.get("website")

        if honeypot: 
            flash("Submission failed. Please try again. Bot???", "error")
            return redirect(url_for("support"))

        if not user_email or not subject or not message:
            flash("Email, subject, and message are required.", "error")
            return redirect(url_for("support"))
        
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

@app.route("/support/success")
def support_success():
    return render_template("support_success.html")

@app.route("/admin/tickets")
@login_required
def admin_tickets():
    db = get_db()
    rows = db.execute("SELECT * FROM support_tickets ORDER BY id DESC LIMIT 100").fetchall()
    return render_template("admin_tickets.html", rows=rows)

if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(debug=True)
