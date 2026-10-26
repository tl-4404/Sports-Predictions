"""Regression checks for the July 2026 MLB pipeline audit.

Run directly: python tests/test_mlb_audit_regressions.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlb"))

import calibration
import db

FAILURES = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not condition:
        FAILURES.append(name)


tmp = tempfile.mkdtemp()
db.DB_PATH = Path(tmp) / "audit.db"
conn = db.connect()
db.ensure_schema(conn)

print("sportsbook line snapshots:")
base = {
    "game_date": "2026-07-20",
    "game_pk": 1,
    "event_id": "event-1",
    "player_name": "Line Mover",
    "player_id": 10,
    "market": "pitcher_strikeouts",
    "over_odds": -110,
    "under_odds": -110,
}
removed = {
    **base,
    "player_name": "Removed Player",
    "player_id": 11,
}
db.insert_lines(conn, [
    {**base, "book": "Book A", "line": 5.5,
     "as_of": "2026-07-20T17:00:00+00:00"},
    {**removed, "book": "Book A", "line": 6.5,
     "as_of": "2026-07-20T17:00:00+00:00"},
    {**base, "book": "Book A", "line": 4.5,
     "as_of": "2026-07-20T17:05:00+00:00"},
    # A second ladder row from the same current snapshot remains active.
    {**base, "book": "Book A", "line": 3.5,
     "as_of": "2026-07-20T17:05:00+00:00"},
    {**base, "book": "Book B", "line": 5.5,
     "as_of": "2026-07-20T17:04:00+00:00"},
])
active = db.latest_lines(conn, "2026-07-20")
book_a = sorted(row["line"] for row in active if row["book"] == "Book A")
book_b = sorted(row["line"] for row in active if row["book"] == "Book B")
check("obsolete Book A threshold removed", book_a == [3.5, 4.5], f"({book_a})")
check("removed Book A player disappears",
      all(row["player_name"] != "Removed Player" for row in active))
check("independent Book B snapshot retained", book_b == [5.5], f"({book_b})")

print("calibration sample hygiene:")
prediction_base = {
    "run_id": "run-1",
    "game_date": "2026-07-20",
    "game_pk": 1,
    "player_name": "Calibration Pitcher",
    "player_id": 20,
    "market": "pitcher_strikeouts",
    "line": 5.5,
    "source": "sim",
    "mean": 5.8,
    "book": "Book A",
    "over_odds": -110,
    "under_odds": -110,
    "market_p_over": 0.50,
    "edge_over": 0.10,
    "edge_under": -0.10,
    "bet_side": "over",
    "kelly_frac": 0.01,
    "rationale": None,
    "raw_p_over": 0.70, "model_version": "mlb-audit-v5",
    "game_start_time": "2026-07-20T19:00:00+00:00", "price_verified": 1,
}
old_id = db.insert_prediction(conn, {
    **prediction_base,
    "created_at": "2026-07-20T16:00:00+00:00",
    "p_over": 0.70,
})
new_id = db.insert_prediction(conn, {
    **prediction_base,
    "run_id": "run-2",
    "created_at": "2026-07-20T16:10:00+00:00",
    "p_over": 0.60,
    "raw_p_over": 0.60,
})
synthetic_id = db.insert_prediction(conn, {
    **prediction_base,
    "run_id": "run-synthetic",
    "created_at": "2026-07-20T16:15:00+00:00",
    "line": 6.5,
    "p_over": 0.50,
    "book": None,
    "over_odds": None,
    "under_odds": None,
    "market_p_over": None,
    "edge_over": None,
    "edge_under": None,
    "bet_side": None,
    "kelly_frac": 0.0,
})
for prediction_id in (old_id, new_id, synthetic_id):
    conn.execute(
        """INSERT INTO prop_results
           (prediction_id, graded_at, actual, outcome, bet_won,
            profit_units, closing_p_over, clv)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (prediction_id, "2026-07-21T12:00:00+00:00", 6, "over", 1,
         0.01, 0.52, 0.08),
    )
conn.commit()
eligible = list(calibration._eligible_rows(conn, "2026-07-20"))
check("repeated reruns count once", len(eligible) == 1, f"(n={len(eligible)})")
check("first pregame forecast retained", eligible and abs(eligible[0]["p_over"] - 0.70) < 1e-9)
check("synthetic no-market row excluded",
      all(row["market_p_over"] is not None for row in eligible))

conn.close()
print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
