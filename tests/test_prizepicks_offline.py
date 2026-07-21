"""Offline end-to-end test of the PrizePicks lane.

No network, no pytest:  python tests/test_prizepicks_offline.py

Covers payout math, schema, line ingestion, stale-line replacement, conservative
playability gates, evaluation, and next-day grading. StatsAPI is monkeypatched.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlb"))

import db
import data_today
import prizepicks as pp

for _key in ("PRIZEPICKS_LIVE", "ODDS_SOURCE", "ODDS_API_KEY"):
    os.environ.pop(_key, None)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not condition:
        FAILURES.append(name)


print("prizepicks math:")
be2 = pp.power_breakeven(2)
be3 = pp.power_breakeven(3)
check("2-pick break-even ~57.7%", abs(be2 - 0.5774) < 1e-3, f"({be2:.4f})")
check("3-pick break-even ~58.5%", abs(be3 - (0.2 ** (1 / 3))) < 1e-9,
      f"({be3:.4f})")
check("EV zero exactly at break-even", abs(pp.power_ev(be2, 2)) < 1e-9)
check("EV positive above break-even", pp.power_ev(0.62, 2) > 0)
check("EV negative below break-even", pp.power_ev(0.55, 2) < 0)
check("default play margin is conservative", pp.PLAYABLE_MARGIN >= 0.04)

print("prizepicks pipeline (temp db, mocked network):")
tmp = tempfile.mkdtemp()
db.DB_PATH = Path(tmp) / "test.db"
DATE = "2026-07-01"


def fake_schedule(game_date=None):
    return [{
        "game_pk": 999001, "game_date": DATE, "status": "Preview",
        "home_team": "New York Mets", "away_team": "Atlanta Braves",
        "venue": "Citi Field",
        "home_pitcher": {"id": 111, "name": "Ace Example", "hand": "R"},
        "away_pitcher": {"id": 222, "name": "Soft Tosser", "hand": "L"},
        # Ace has a confirmed opposing lineup; Soft Tosser does not.
        "home_lineup": [], "away_lineup": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }]


def fake_profile(pid, season=None):
    if pid == 111:
        return {"k": 0.30, "bb": 0.06, "hit": 0.19, "batters_faced": 400,
                "recent_pitch_counts": [98, 95, 101, 97, 99]}
    return {"k": 0.16, "bb": 0.09, "hit": 0.24, "batters_faced": 350,
            "recent_pitch_counts": [82, 78, 85, 80, 79]}


def fake_batter(bid, hand, season=None):
    return {"k": 0.22 + (bid % 3) * 0.02, "bb": 0.08, "hit": 0.21}


data_today.get_schedule = fake_schedule
data_today.get_pitcher_profile = fake_profile
data_today.get_batter_rates = fake_batter

n_lines = pp.refresh(DATE)
check("stub projections stored", n_lines == 4, f"(n={n_lines})")

conn = db.connect()
lines = conn.execute(
    "SELECT * FROM prizepicks_lines WHERE game_date=?", (DATE,)
).fetchall()
check("stat_type mapped to markets",
      {line["market"] for line in lines} == {"pitcher_strikeouts", "pitcher_outs"})
ace_outs_line = [
    line for line in lines
    if line["player_name"] == "Ace Example" and line["market"] == "pitcher_outs"
][0]
check("PP outs line kept at its own number (16.5)", ace_outs_line["line"] == 16.5)

# A moved threshold must replace the old value rather than coexist with it.
pp._insert_lines(conn, [{
    "game_date": DATE, "player_name": "Moved Line", "player_id": None,
    "market": "pitcher_strikeouts", "line": 5.5,
    "stat_type": "Pitcher Strikeouts", "projection_id": "move-1",
    "as_of": "2026-07-01T10:00:00+00:00",
}])
pp._insert_lines(conn, [{
    "game_date": DATE, "player_name": "Moved Line", "player_id": None,
    "market": "pitcher_strikeouts", "line": 4.5,
    "stat_type": "Pitcher Strikeouts", "projection_id": "move-1",
    "as_of": "2026-07-01T10:05:00+00:00",
}])
active_moved = [
    line for line in pp._latest_lines(conn, DATE)
    if line["player_name"] == "Moved Line"
]
check("old moved line removed from active board",
      len(active_moved) == 1 and active_moved[0]["line"] == 4.5,
      f"({[line['line'] for line in active_moved]})")

n_picks = pp.evaluate_date(DATE, n_sims=6000)
check("picks logged for both starters", n_picks == 4, f"(n={n_picks})")

picks = conn.execute(
    "SELECT * FROM prizepicks_picks WHERE game_date=?", (DATE,)
).fetchall()
ace_k = [
    pick for pick in picks
    if pick["player_name"] == "Ace Example"
    and pick["market"] == "pitcher_strikeouts"
][0]
check("ace strikeouts picks MORE", ace_k["pick_side"] == "more",
      f"(p_over={ace_k['p_over']:.3f})")
check("ace strikeouts clears conservative margin", ace_k["playable"] == 1,
      f"(edge={ace_k['edge']:+.3f})")

soft_k = [
    pick for pick in picks
    if pick["player_name"] == "Soft Tosser"
    and pick["market"] == "pitcher_strikeouts"
][0]
check("soft-tosser strikeouts picks LESS", soft_k["pick_side"] == "less",
      f"(p_over={soft_k['p_over']:.3f})")
check("unconfirmed opposing lineup forces HOLD",
      all(pick["playable"] == 0 for pick in picks
          if pick["player_name"] == "Soft Tosser"))
check("p_pick is the chosen side's probability",
      all(abs(pick["p_pick"] - (
          pick["p_over"] if pick["pick_side"] == "more" else 1 - pick["p_over"]
      )) < 1e-9 for pick in picks))
check("edge = p_pick - break-even",
      all(abs(pick["edge"] - (pick["p_pick"] - pick["breakeven"])) < 1e-9
          for pick in picks))
check("confirmed plays require the configured margin",
      all(bool(pick["playable"]) == (pick["edge"] >= pp.PLAYABLE_MARGIN)
          for pick in picks if pick["player_name"] == "Ace Example"))


def fake_actuals(game_pk):
    return {
        111: {"name": "Ace Example", "ks": 8, "outs": 19},
        222: {"name": "Soft Tosser", "ks": 3, "outs": 14},
    }


data_today.get_starter_actuals = fake_actuals
n_graded = pp.grade_date(DATE)
check("all picks graded", n_graded == n_picks, f"({n_graded})")
res = conn.execute(
    """SELECT p.player_name, p.market, p.line, p.pick_side,
              r.actual, r.outcome, r.pick_won
       FROM prizepicks_results r JOIN prizepicks_picks p ON p.id = r.pick_id"""
).fetchall()
ace_k_res = [
    row for row in res
    if row["player_name"] == "Ace Example"
    and row["market"] == "pitcher_strikeouts"
][0]
check("ace 8 Ks graded MORE over 6.5",
      ace_k_res["actual"] == 8 and ace_k_res["outcome"] == "more")
check("ace MORE pick won", ace_k_res["pick_won"] == 1)
soft_out_res = [
    row for row in res
    if row["player_name"] == "Soft Tosser"
    and row["market"] == "pitcher_outs"
][0]
check("soft-tosser 14 outs vs 14.5 -> LESS", soft_out_res["outcome"] == "less")
check("re-grade grades nothing new", pp.grade_date(DATE) == 0)
conn.close()

print("prizepicks live parser:")
PP_PAYLOAD = {
    "data": [
        {"id": "proj-1", "type": "projection", "attributes": {
            "stat_type": "Pitcher Strikeouts", "line_score": 6.5,
            "start_time": "2026-07-01T19:05:00-04:00"},
         "relationships": {"new_player": {"data": {"id": "np-1"}}}},
        {"id": "proj-2", "type": "projection", "attributes": {
            "stat_type": "Pitching Outs", "line_score": 16.5,
            "start_time": "2026-07-01T19:05:00-04:00"},
         "relationships": {"new_player": {"data": {"id": "np-1"}}}},
        {"id": "proj-3", "type": "projection", "attributes": {
            "stat_type": "Pitcher Strikeouts", "line_score": 5.5,
            "start_time": "2026-07-02T13:10:00-04:00"},
         "relationships": {"new_player": {"data": {"id": "np-2"}}}},
        {"id": "proj-4", "type": "projection", "attributes": {
            "stat_type": "Hits Allowed", "line_score": 4.5,
            "start_time": "2026-07-01T19:05:00-04:00"},
         "relationships": {"new_player": {"data": {"id": "np-1"}}}},
    ],
    "included": [
        {"id": "np-1", "type": "new_player",
         "attributes": {"name": "Ace Example"}},
        {"id": "np-2", "type": "new_player",
         "attributes": {"name": "Tomorrow Guy"}},
    ],
}

parsed = pp._parse_projections(PP_PAYLOAD, DATE, "2026-07-01T12:00:00Z")
check("parsed today's mapped projections (2)", len(parsed) == 2, f"({len(parsed)})")
parsed_by_market = {row["market"]: row for row in parsed}
check("strikeouts projection parsed",
      "pitcher_strikeouts" in parsed_by_market
      and parsed_by_market["pitcher_strikeouts"]["player_name"] == "Ace Example"
      and parsed_by_market["pitcher_strikeouts"]["line"] == 6.5)
check("player name resolved from included sidecar",
      all(row["player_name"] == "Ace Example" for row in parsed))
check("projection_id carried through",
      parsed_by_market["pitcher_strikeouts"]["projection_id"] == "proj-1")
check("other-day projection excluded",
      all(row["player_name"] != "Tomorrow Guy" for row in parsed))
check("unmapped stat_type excluded",
      all(row["market"] in ("pitcher_strikeouts", "pitcher_outs")
          for row in parsed))

# A string value of "0" must not accidentally enable the live network path.
os.environ["PRIZEPICKS_LIVE"] = "0"
check("PRIZEPICKS_LIVE=0 remains offline",
      isinstance(pp.get_fetcher(), pp.StubPrizePicksFetcher))
os.environ.pop("PRIZEPICKS_LIVE", None)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
