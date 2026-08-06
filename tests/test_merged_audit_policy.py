"""Focused regression tests for the merged v4.2/v5 policy."""
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mlb"))

import calibration
import db
import decision
import simulator

# The real repository supplies these modules. Tiny stubs keep this focused test
# runnable against the isolated changed-file bundle used during review.
try:
    import grade
except ModuleNotFoundError:
    import types
    data_today_stub = types.ModuleType("data_today")
    timeutils_stub = types.ModuleType("timeutils")
    timeutils_stub.slate_today = lambda: "2026-08-05"
    betting_math_stub = types.ModuleType("betting_math")
    betting_math_stub.american_to_decimal = lambda odds: (1 + 100/abs(odds)) if odds < 0 else (1 + odds/100)
    betting_math_stub.consensus_p_over = lambda rows: None
    sys.modules["data_today"] = data_today_stub
    sys.modules["timeutils"] = timeutils_stub
    sys.modules["betting_math"] = betting_math_stub
    import grade


def proven(market="pitcher_strikeouts"):
    return {"market": market, "sample_size": 250, "model_brier": 0.20,
            "market_brier": 0.24, "average_clv": 0.015, "clean_slates": 18}


def test_decision_thresholds_and_shadow():
    assert decision.complete_lineup(range(1, 10))
    assert not decision.complete_lineup([1,2,3,4,5,6,7,8,8])
    shadow = decision.decide(
        0.61, 0.52, book_count=2, validation=proven(),
        market="pitcher_strikeouts", force_shadow=True,
    )
    assert shadow.status == "candidate" and shadow.bet_side is None
    live = decision.decide(
        0.61, 0.52, book_count=2, validation=proven(),
        market="pitcher_strikeouts", force_shadow=False,
    )
    assert live.status == "official" and live.bet_side == "over"
    weak = decision.decide(
        0.59, 0.52, book_count=2, validation=proven(),
        market="pitcher_strikeouts", force_shadow=False,
    )
    assert weak.status != "official"
    disagree = decision.decide(
        0.61, 0.52, book_count=2, validation=proven(),
        market="pitcher_strikeouts", lane_gap=0.13, force_shadow=False,
    )
    assert disagree.status != "official"


def test_outs_disabled_and_erode_only():
    call = decision.decide(
        0.39, 0.50, book_count=2, validation=proven("pitcher_outs"),
        market="pitcher_outs", force_shadow=False,
    )
    if not decision.OUTS_OFFICIAL_ENABLED:
        assert call.status != "official"
    assert decision.erode_only(0.62, 0.70, "over", 0.03) == 0.62
    assert abs(decision.erode_only(0.62, 0.55, "over", 0.03)-0.59) < 1e-12


def test_state_mixture_and_stress():
    mix = simulator.workload_states([92, 96, 88, 101, 74])
    assert {s["name"] for s in mix["states"]} == {"short","normal","extended"}
    assert abs(sum(s["weight"] for s in mix["states"])-1.0) < 1e-9
    ks = np.array([4,5,6,7,3,5,4,6]*100)
    bf = np.array([20,21,22,24,18,21,19,23]*100)
    stress = simulator.k_under_stress(ks, bf, 5.5)
    assert stress["base"] >= stress["plus_2"] >= stress["plus_4"]
    assert stress["basis"] in {"skill_under","workload_under"}


def test_ledger_first_snapshot_and_refund_exclusion():
    tmp = Path(tempfile.mkdtemp())
    db.DB_PATH = tmp / "audit.db"
    conn = db.connect(); db.ensure_schema(conn)
    created = "2026-08-01T16:00:00+00:00"
    start = "2026-08-01T23:00:00+00:00"
    base = {
        "run_id":"run-1","created_at":created,"game_date":"2026-08-01",
        "game_pk":1,"player_name":"Ledger Pitcher","player_id":7,
        "market":"pitcher_strikeouts","line":5.5,"source":"sim",
        "p_over":0.61,"raw_p_over":0.61,"decision_p_over":0.52,"p_push":0,
        "mean":6.2,"book":"A","over_odds":-110,"under_odds":-110,
        "market_p_over":0.52,"edge_over":0,"edge_under":0,
        "candidate_side":"over","bet_side":None,"kelly_frac":0,
        "rationale":"{}","policy_version":decision.POLICY_VERSION,
        "quality_json":"{}","decision_status":"candidate","official":0,
        "provenance":"model","line_as_of":"2026-08-01T15:55:00+00:00",
        "event_start":start,"line_verified":1,"line_tier":"standard",
        "lineup_status":"confirmed","starter_verified":1,"workload_verified":1,
    }
    first = db.insert_prediction(conn, base)
    later = db.insert_prediction(conn, {**base,"run_id":"run-2",
        "created_at":"2026-08-01T18:00:00+00:00","p_over":0.70,"raw_p_over":0.70})
    assert conn.execute("SELECT COUNT(*) n FROM decision_ledger").fetchone()["n"] == 2
    for pid, status, outcome in ((first,"graded","over"),(later,"refund","push")):
        conn.execute(
            """INSERT INTO prop_results
               (prediction_id,graded_at,actual,outcome,settlement_status,bet_won,
                profit_units,closing_p_over,clv,clv_available,roi_eligible)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (pid,"2026-08-02T12:00:00+00:00",6 if outcome=="over" else 5.5,
             outcome,status,None,0,0.54,0.02 if status=="graded" else None,
             1 if status=="graded" else 0,0),
        )
    conn.commit()
    rows = calibration._eligible_rows(conn,"2026-08-01",decision.POLICY_VERSION)
    assert len(rows) == 1 and rows[0]["id"] == first
    conn.close()


def test_refunds_are_not_roi_losses():
    pred = {"official":1,"decision_status":"official","bet_side":"over",
            "kelly_frac":0.01,"over_odds":-110,"under_odds":-110}
    settlement = grade._settle(pred,"push","over")
    assert settlement[0] == "refund" and settlement[3] == 0 and settlement[4] == 0
    settlement = grade._settle(pred,"void","over")
    assert settlement[0] == "refund" and settlement[3] == 0 and settlement[4] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
