"""Offline regression checks for audited PrizePicks live-fetch behavior.

Run directly: python tests/test_prizepicks_fetcher_audit.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlb"))

import prizepicks as pp

FAILURES = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not condition:
        FAILURES.append(name)


DATE = "2026-07-01"

print("jurisdiction guard:")
os.environ.pop("PRIZEPICKS_STATE_CODE", None)
try:
    pp.PrizePicksFetcher(state_code="").fetch(DATE)
    missing_state_rejected = False
except RuntimeError:
    missing_state_rejected = True
check("missing state code rejected before fetch", missing_state_rejected)

print("pagination and board parsing:")
original_get_json = pp.netfetch.get_json
calls = []


def projection(pid, player_id, line, odds_type="standard"):
    return {
        "id": pid,
        "type": "projection",
        "attributes": {
            "stat_type": "Pitcher Strikeouts",
            "line_score": line,
            "odds_type": odds_type,
            "status": "pre_game",
            "start_time": "2026-07-01T19:05:00-04:00",
        },
        "relationships": {"new_player": {"data": {"id": player_id}}},
    }


PAGE_1 = {
    "meta": {"total_pages": 2},
    "data": [projection("p1", "n1", 5.5)],
    "included": [{
        "id": "n1", "type": "new_player",
        "attributes": {"display_name": "First Pitcher"},
    }],
}
PAGE_2 = {
    "meta": {"total_pages": 2},
    "data": [projection("p2", "n2", 6.5, "demon")],
    "included": [{
        "id": "n2", "type": "new_player",
        "attributes": {"name": "Second Pitcher"},
    }],
}


def fake_get_json(url, params=None, **kwargs):
    params = params or {}
    calls.append(dict(params))
    return PAGE_1 if params.get("page") == 1 else PAGE_2


pp.netfetch.get_json = fake_get_json
try:
    rows = pp.PrizePicksFetcher(state_code="CA").fetch(DATE)
finally:
    pp.netfetch.get_json = original_get_json

check("all pages requested", [call.get("page") for call in calls] == [1, 2],
      f"({[call.get('page') for call in calls]})")
check("state code sent on every page",
      all(call.get("state_code") == "CA" for call in calls))
check("pre-game non-live filters sent",
      all(call.get("is_live") == "false" and call.get("game_mode") == "pickem"
          for call in calls))
check("both pages returned", len(rows) == 2, f"(n={len(rows)})")
check("display_name fallback works",
      any(row["player_name"] == "First Pitcher" for row in rows))
check("odds tiers preserved",
      {row["odds_type"] for row in rows} == {"standard", "demon"})

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
