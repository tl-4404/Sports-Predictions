"""Assemble a real backtest dataset from the free MLB StatsAPI.

Run on your own machine (makes many HTTP calls; NOT used at serving time):

    python mlb/build_backtest_data.py 2024-04-01 2024-09-29 \
        --out mlb/cache/backtest_2024.json
    python mlb/backtest.py mlb/cache/backtest_2024.json

For each completed game we record, per starting pitcher:
  * per-PA rates estimated from the PRIOR season (season-1) — no look-ahead
    leakage, and it mirrors how you'd actually project a start
  * the opposing lineup that batted, with each batter's prior-season rates
    vs the starter's hand
  * recent pitch counts (prior-season game logs)
  * park + home-plate-umpire K factors
  * the ACTUAL strikeouts and outs the starter recorded

StatsAPI carries no betting odds, so rows have no `lines[].over_odds`: the
backtest reports calibration / log-loss / Brier (the honest signal test). To
add ROI/CLV, merge in closing lines from an odds archive keyed by
(game_date, pitcher, market, line).
"""
import argparse
import json
from datetime import date, timedelta

import requests

import data_today
import factors

BASE = "https://statsapi.mlb.com/api/v1"


def _get(path, **params):
    r = requests.get(f"{BASE}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _final_game_pks(game_date):
    data = _get("/schedule", sportId=1, date=game_date)
    pks = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("codedGameState") == "F":
                pks.append(g["gamePk"])
    return pks


def _lines_for():
    """Predeclared diagnostic grid, independent of the realized outcome.

    These are not historical offers and cannot establish betting profitability.
    """
    return ([{"market": "pitcher_strikeouts", "line": x + 0.5} for x in range(2, 10)]
            + [{"market": "pitcher_outs", "line": x + 0.5} for x in range(9, 22)])


def build_game(game_pk, home_team_name=None):
    box = _get(f"/game/{game_pk}/boxscore")
    game = _get(f"/schedule", gamePk=game_pk)["dates"][0]["games"][0]
    gdate = game.get("officialDate") or game["gameDate"][:10]
    prev_season = date.fromisoformat(gdate).year - 1

    hp_ump = None
    for off in box.get("officials", []):
        if off.get("officialType") == "Home Plate":
            hp_ump = (off.get("official") or {}).get("fullName")
    k_ump = 1.0  # no dated historical umpire-factor snapshot is available

    home_name = (box.get("teams", {}).get("home", {}).get("team", {}) or {}).get("name") \
        or home_team_name or ""
    k_park = 1.0  # never apply today's learned park cache to an earlier season

    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        team = box["teams"][side]
        pitchers = team.get("pitchers", [])
        if not pitchers:
            continue
        starter_id = pitchers[0]
        pdata = team["players"].get(f"ID{starter_id}", {})
        stat = pdata.get("stats", {}).get("pitching", {})
        actual_ks = int(stat.get("strikeOuts", 0))
        actual_outs = data_today.innings_to_outs(stat.get("inningsPitched", "0.0"))
        name = (pdata.get("person") or {}).get("fullName")

        hand = data_today._pitch_hand(starter_id)
        profile = data_today.get_pitcher_profile(starter_id, season=prev_season, as_of_date=gdate)

        opp_order = box["teams"][opp].get("battingOrder", [])[:9]
        lineup = []
        for bid in opp_order:
            r = data_today.get_batter_rates(bid, hand, season=prev_season, as_of_date=gdate)
            lineup.append({"k": r["k"], "bb": r["bb"], "hit": r["hit"]})
        while len(lineup) < 9:
            lineup.append(dict(data_today.LEAGUE))

        rows.append({
            "game_date": gdate, "pitcher_name": name,
            "game_pk": game_pk, "pitcher_id": starter_id,
            "game_start_time": game["gameDate"],
            "evaluation_scope": "retrospective_skill_diagnostic_not_decision_replay",
            "lineup_provenance": "actual_boxscore_not_archived_pregame",
            "factor_provenance": "neutral_no_historical_snapshots",
            "pitcher": {"k": profile["k"], "bb": profile["bb"], "hit": profile["hit"]},
            "lineup": lineup,
            "recent_pitch_counts": profile["recent_pitch_counts"],
            "k_park": k_park, "k_ump": k_ump,
            "actual_ks": actual_ks, "actual_outs": actual_outs,
            "lines": _lines_for(),
        })
    return rows


def build_range(start, end, out_path):
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    dataset = []
    day = d0
    while day <= d1:
        ds = day.isoformat()
        try:
            pks = _final_game_pks(ds)
        except requests.RequestException:
            pks = []
        for pk in pks:
            try:
                dataset.extend(build_game(pk))
            except Exception as e:
                print(f"  game {pk}: skipped ({e})")
        print(f"{ds}: {len(pks)} finals -> {len(dataset)} starts total")
        day += timedelta(days=1)

    with open(out_path, "w") as f:
        json.dump(dataset, f)
    print(f"wrote {len(dataset)} starts -> {out_path}")
    return dataset


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--out", default="mlb/cache/backtest.json")
    args = ap.parse_args()
    build_range(args.start, args.end, args.out)
