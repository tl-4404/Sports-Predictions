"""Offline test of the Flask surfacing layer (no network).

    python tests/test_app_offline.py

Covers the routes, the MLB props surfacing (judge > llm > sim per prop, with
the three-lane comparison), the lane scorecard from graded results, pick
add/delete, and the admin gate.
"""
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mlb"))

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


tmp = Path(tempfile.mkdtemp())
import app as appmod
appmod.DB_PATH = tmp / "app.db"
app = appmod.app
app.config["TESTING"] = True

import db as propdb
propdb.DB_PATH = tmp / "app.db"

today = appmod.mlb_today().isoformat()
yest = (appmod.mlb_today() - timedelta(days=1)).isoformat()


def mkrow(gd, src, p_over, bet="over", kelly=0.02):
    return {"run_id": "r1", "created_at": propdb.utcnow(), "game_date": gd,
            "game_pk": 1, "player_name": "Ace Example", "player_id": 111,
            "market": "pitcher_strikeouts", "line": 6.5, "source": src,
            "p_over": p_over, "mean": 6.9 if src == "sim" else None,
            "book": "BookA", "over_odds": -110, "under_odds": -110,
            "market_p_over": 0.5, "edge_over": p_over - 0.5,
            "edge_under": 0.5 - p_over, "bet_side": bet, "kelly_frac": kelly,
            "policy_version": "sportsaid-v5.0-merged-audit",
            "candidate_side": bet, "decision_status": "official",
            "official": 1, "selected_for_card": 1,
            "rationale": '{"action":"bet","rationale":"deep start likely"}'
                         if src != "sim" else None}


with app.app_context():
    appmod.init_db()
    conn = propdb.connect()
    propdb.ensure_schema(conn)

    # today: all three lanes on one prop → 3-way comparison
    for src, p in [("sim", 0.62), ("llm", 0.66), ("judge", 0.64)]:
        propdb.insert_prediction(conn, mkrow(today, src, p))

    # yesterday: one graded bet per lane → scorecard (sim/judge win, llm loses)
    for src, p, won in [("sim", 0.62, 1), ("llm", 0.66, 0), ("judge", 0.64, 1)]:
        pid = propdb.insert_prediction(conn, mkrow(yest, src, p))
        profit = 0.02 * 0.909 if won else -0.02
        conn.execute(
            """INSERT INTO prop_results (prediction_id, graded_at, actual,
               outcome, settlement_status, bet_won, profit_units, closing_p_over,
               entry_p_side, closing_p_side, clv, clv_available, roi_eligible)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pid, propdb.utcnow(), 8, "over", "win" if won else "loss",
             won, profit, 0.55, 0.50, 0.55, 0.09 if won else -0.02, 1, 1))
    conn.commit()

    # data-layer checks
    props = appmod.load_mlb_props(propdb.connect())
    check("load_mlb_props returns the prop", len(props) == 1)
    p = props[0]
    check("judge wins the lane race", p["lane"] == "judge")
    check("three-lane comparison attached",
          p["lanes"].get("sim") == 0.62 and p["lanes"].get("llm") == 0.66
          and p["lanes"].get("judge") == 0.64)
    check("judge/llm row inherits sim mean", p["mean"] == 6.9)
    check("rationale text extracted", p.get("rationale_text") == "deep start likely")

    board = {b["source"]: b for b in appmod.load_lane_scoreboard(propdb.connect())}
    check("scorecard has all three lanes", set(board) == {"sim", "llm", "judge"})
    check("sim ROI positive", board["sim"]["roi"] > 0)
    check("llm ROI negative", board["llm"]["roi"] < 0)
    check("judge beat-close rate computed",
          board["judge"]["clv_beat_rate"] == 1.0)

# route + render checks
c = app.test_client()
for path, want in [("/", 200), ("/login", 200), ("/register", 200),
                   ("/support", 302), ("/my-card", 302), ("/admin/tickets", 302)]:
    check(f"GET {path} -> {want}", c.get(path).status_code == want)

html = c.get("/").data.decode()
check("scorecard section renders", "Lane scorecard" in html)
check("props section renders", "MLB pitcher props" in html)
check("three lane probs shown", all(x in html for x in ("62%", "66%", "64%")))

# auth + pick lifecycle
c.post("/register", data={"email": "t@t.com", "password": "x"}, follow_redirects=True)
r = c.post("/picks/add", data={"league": "MLB", "market": "Moneyline",
                               "side": "home", "team_home": "Mets",
                               "team_away": "Braves", "odds": "-110"},
           follow_redirects=True)
check("pick added", b"Mets ML" in r.data)
import sqlite3
cc = sqlite3.connect(appmod.DB_PATH)
pick_id = cc.execute("SELECT id FROM saved_picks").fetchone()[0]
cc.close()
r = c.post("/picks/delete", data={"pick_id": pick_id}, follow_redirects=True)
check("pick deleted via /picks/delete", b"Pick deleted" in r.data)
check("/support renders (not shadowed)", c.get("/support").status_code == 200)
check("/admin/tickets 403 for non-admin", c.get("/admin/tickets").status_code == 403)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
