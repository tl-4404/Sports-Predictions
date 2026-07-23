"""Regression checks for failures that can create false live-betting signals.

No network, no pytest:  python tests/test_safety_regressions.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mlb"))

import db
import data_today
import grade
import prizepicks
import simulator
from timeutils import is_open_slate_event, slate_date

FAILURES = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


print("time + live-event gates:")
check(
    "late UTC game stays on prior Eastern slate",
    slate_date("2026-07-11T02:10:00Z") == "2026-07-10",
)
check(
    "future event accepted",
    is_open_slate_event("2026-07-11T02:10:00Z", "2026-07-10", "2026-07-10T20:00:00Z"),
)
check(
    "started event rejected",
    not is_open_slate_event(
        "2026-07-11T02:10:00Z", "2026-07-10", "2026-07-11T02:10:00Z"
    ),
)

print("active line snapshots:")
tmp = Path(tempfile.mkdtemp())
db.DB_PATH = tmp / "test.db"
conn = db.connect()
db.ensure_schema(conn)


def line(line_value, as_of, book="Book A", over=-110, under=-110):
    return {
        "game_date": "2026-07-10",
        "game_pk": 1,
        "event_id": "e1",
        "player_name": "Pitcher One",
        "player_id": 111,
        "market": "pitcher_strikeouts",
        "book": book,
        "line": line_value,
        "over_odds": over,
        "under_odds": under,
        "as_of": as_of,
    }


db.insert_lines(conn, [line(5.5, "2026-07-10T16:00:00+00:00")])
# Two alternates in one current snapshot should survive together, while 5.5
# from the older snapshot must disappear.
db.insert_lines(
    conn,
    [line(6.5, "2026-07-10T17:00:00+00:00"), line(7.5, "2026-07-10T17:00:00+00:00")],
)
active = db.latest_lines(conn, "2026-07-10")
check("stale moved line removed", {r["line"] for r in active} == {6.5, 7.5})
pre_move = db.latest_lines(conn, "2026-07-10", as_of_before="2026-07-10T16:30:00+00:00")
check(
    "historical cutoff recovers then-current line",
    [r["line"] for r in pre_move] == [5.5],
)

print("usage leash:")
rng = np.random.default_rng(7)
limits = [simulator.sample_pitch_limit([24, 28, 31], rng) for _ in range(1000)]
check(
    "opener is not forced to old 55-pitch floor",
    np.mean(limits) < 40,
    f"(mean={np.mean(limits):.1f})",
)

print("grading integrity + CLV:")
# Separate date so the line-snapshot test above cannot affect closing prices.
db.insert_lines(
    conn,
    [
        {
            **line(6.5, "2026-07-01T12:00:00+00:00", over=-110, under=-110),
            "game_date": "2026-07-01",
            "game_pk": 999,
        },
        {
            **line(6.5, "2026-07-01T18:00:00+00:00", over=-150, under=130),
            "game_date": "2026-07-01",
            "game_pk": 999,
        },
    ],
)
pred_id = db.insert_prediction(
    conn,
    {
        "run_id": "run1",
        "created_at": "2026-07-01T12:01:00+00:00",
        "game_date": "2026-07-01",
        "game_pk": 999,
        "player_name": "Pitcher One",
        "player_id": 111,
        "market": "pitcher_strikeouts",
        "line": 6.5,
        "source": "sim",
        "p_over": 0.62,
        "mean": 7.1,
        "book": "Book A",
        "over_odds": -110,
        "under_odds": -110,
        "market_p_over": 0.5,
        "edge_over": 0.12,
        "edge_under": -0.12,
        "bet_side": "over",
        "kelly_frac": 0.01,
        "rationale": None,
    },
)


def final_schedule(game_date=None):
    return [
        {
            "game_pk": 999,
            "game_date": "2026-07-01",
            "start_time": "2026-07-01T19:00:00+00:00",
            "status": "Final",
        }
    ]


data_today.get_schedule = final_schedule


def broken_box(_):
    raise ConnectionError("temporary StatsAPI failure")


data_today.get_starter_actuals = broken_box
check("box-score failure stays ungraded", grade.grade_date("2026-07-01") == 0)
still_missing = conn.execute(
    "SELECT 1 FROM prop_results WHERE prediction_id=?", (pred_id,)
).fetchone()
check("no permanent void written on fetch failure", still_missing is None)

data_today.get_starter_actuals = lambda _: {
    111: {"name": "Pitcher One", "ks": 8, "outs": 18}
}
check("later retry grades successfully", grade.grade_date("2026-07-01") == 1)
result = conn.execute(
    "SELECT * FROM prop_results WHERE prediction_id=?", (pred_id,)
).fetchone()
check(
    "CLV measures entry-to-close market movement",
    result["clv"] > 0,
    f"(clv={result['clv']:.4f})",
)
conn.close()

print("environment flags:")
os.environ["PRIZEPICKS_LIVE"] = "0"
check(
    "PRIZEPICKS_LIVE=0 remains offline",
    isinstance(prizepicks.get_fetcher(), prizepicks.StubPrizePicksFetcher),
)
os.environ.pop("PRIZEPICKS_LIVE", None)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
    raise SystemExit(1)
print("All checks passed.")
