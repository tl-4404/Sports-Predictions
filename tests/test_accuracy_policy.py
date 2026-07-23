"""Regression checks for the fail-closed accuracy_v2 decision policy.

No network, no pytest:
    python tests/test_accuracy_policy.py
"""
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mlb"))

import calibration
import db
import decision
import numpy as np
import predict
from timeutils import slate_today

FAILURES = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not condition:
        FAILURES.append(name)


print("hard data gates:")
check("nine unique hitters confirms lineup",
      decision.complete_lineup(range(1, 10)))
check("partial lineup rejected", not decision.complete_lineup(range(1, 9)))
check(
    "duplicate batting slot rejected",
    not decision.complete_lineup([1, 2, 3, 4, 5, 6, 7, 8, 8]),
)

print("validation and market anchoring:")
tmp = Path(tempfile.mkdtemp())
db.DB_PATH = tmp / "accuracy.db"
conn = db.connect()
db.ensure_schema(conn)

base = {
    "created_at": "2026-06-01T12:00:00+00:00",
    "game_pk": 1,
    "player_name": "Validated Pitcher",
    "market": "pitcher_strikeouts",
    "line": 5.5,
    "source": "sim",
    "p_over": 0.70,
    "raw_p_over": 0.70,
    "decision_p_over": 0.50,
    "p_push": 0.0,
    "mean": 6.0,
    "book": "Book A",
    "over_odds": -110,
    "under_odds": -110,
    "market_p_over": 0.50,
    "edge_over": 0.0,
    "edge_under": 0.0,
    "bet_side": None,
    "kelly_frac": 0.0,
    "rationale": json.dumps({"rationale": "paper"}),
    "policy_version": decision.POLICY_VERSION,
    "quality_json": "{}",
}

unproven = calibration.validation_status(
    conn, "pitcher_strikeouts", "sim"
)
held = decision.decide(
    0.57,
    0.50,
    book_count=2,
    validation=unproven,
)
check("unproven model cannot bet", held.bet_side is None)
check("held decision is market anchored",
      abs(held.decision_p_over - 0.50) < 1e-12)

for index in range(decision.MIN_VALIDATION_ROWS):
    prediction_id = db.insert_prediction(conn, {
        **base,
        "run_id": f"paper-{index}",
        "game_date": f"2026-05-{index % 28 + 1:02d}",
        "game_pk": 1000 + index,
        "player_id": 2000 + index,
    })
    conn.execute(
        """INSERT INTO prop_results
           (prediction_id, graded_at, actual, outcome, bet_won,
            profit_units, closing_p_over, clv)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            prediction_id,
            "2026-06-02T12:00:00+00:00",
            7,
            "over",
            None,
            0.0,
            0.52,
            0.02,
        ),
    )
conn.commit()

proven = calibration.validation_status(
    conn, "pitcher_strikeouts", "sim"
)
check(
    "validation counts unique accuracy_v2 rows",
    proven["sample_size"] == decision.MIN_VALIDATION_ROWS,
    f"(n={proven['sample_size']})",
)
check(
    "raw model beats market in validation fixture",
    proven["model_brier"] < proven["market_brier"],
)

approved = decision.decide(
    0.57,
    0.50,
    book_count=2,
    validation=proven,
)
check("validated 7-point discrepancy can bet", approved.bet_side == "over")
check(
    "one-book edge remains HOLD",
    decision.decide(
        0.57, 0.50, book_count=1, validation=proven
    ).bet_side is None,
)
check(
    "small discrepancy treated as noise",
    decision.decide(
        0.54, 0.50, book_count=2, validation=proven
    ).bet_side is None,
)
check(
    "large discrepancy freezes for investigation",
    decision.decide(
        0.60, 0.50, book_count=2, validation=proven
    ).bet_side is None,
)
check(
    "analysis-only LLM cannot manufacture a play",
    decision.decide(
        0.57,
        0.50,
        book_count=2,
        validation=proven,
        allow_auto=False,
    ).bet_side is None,
)
predict._log_market(
    conn,
    "candidate-run",
    "2026-06-03",
    {"game_pk": 9999},
    {"id": 9999, "name": "Candidate Pitcher"},
    "pitcher_strikeouts",
    np.asarray([1] * 57 + [0] * 43),
    [
        {"book": "Book A", "line": 0.5, "over_odds": -110, "under_odds": -110},
        {"book": "Book B", "line": 0.5, "over_odds": -110, "under_odds": -110},
    ],
)
candidate_row = conn.execute(
    """SELECT * FROM prop_predictions
       WHERE run_id='candidate-run'"""
).fetchone()
check(
    "simulator candidate cannot execute before judge",
    candidate_row["candidate_side"] == "over"
    and candidate_row["bet_side"] is None
    and candidate_row["kelly_frac"] == 0,
)
check(
    "judge can only erode an over",
    abs(decision.erode_only(0.57, 0.99, "over", 0.03) - 0.57) < 1e-12
    and abs(decision.erode_only(0.57, 0.52, "over", 0.03) - 0.54) < 1e-12,
)

print("live-line freshness:")
today = slate_today()
now = datetime.now(timezone.utc)
line = {
    "game_date": today,
    "game_pk": 99,
    "event_id": "event",
    "player_name": "Fresh Pitcher",
    "player_id": 99,
    "market": "pitcher_strikeouts",
    "line": 5.5,
    "over_odds": -110,
    "under_odds": -110,
}
db.insert_lines(conn, [
    {
        **line,
        "book": "Stale Book",
        "as_of": (now - timedelta(minutes=45)).isoformat(),
    },
    {
        **line,
        "book": "Fresh Book",
        "as_of": (now - timedelta(minutes=5)).isoformat(),
    },
])
active = db.active_lines(conn, today, now=now)
check(
    "stale book excluded from current board",
    {row["book"] for row in active} == {"Fresh Book"},
)
conn.close()

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
    raise SystemExit(1)
print("All checks passed.")
