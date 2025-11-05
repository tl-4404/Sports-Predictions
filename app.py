from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, g
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "12345"

DB_PATH = Path("sportsaid.db")

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

# place holders
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
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)

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
def localtime_filter(dt):
    return dt.strftime("%b %d, %I:%M %p UTC")

# routes
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

    filtered = sorted(filtered, key=lambda g: g["start_utc"])

    picks = [p for p in MODEL_PICKS if league in ("ALL", p["league"].upper())] or MODEL_PICKS[:1]

    return render_template(
        "index.html",
        leagues=LEAGUES,
        active_league=league,
        games=filtered,
        picks=picks,
        query=q,
    )

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
    return render_template("my_card.html")

@app.route("/support", methods=["GET", "POST"])
@login_required
def support():
    if request.method == "POST":
        email = (current_user.email if hasattr(current_user, "is_authenticated") and current_user.is_authenticated
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
            (email, subject, category, message, datetime.utcnow().isoformat())
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
