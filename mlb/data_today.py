"""Today's MLB slate from the free MLB StatsAPI (statsapi.mlb.com).

Serving-time data source — no pybaseball/statcast here (those are training-
only and too slow for a daily run). Provides:

  get_schedule(date)        — games, probable pitchers, confirmed lineups
  get_pitcher_profile(id)   — season K%/BB%/hit rates + recent pitch counts
  get_batter_rates(id, hand)— season K%/BB%/hit rates vs a pitcher hand
  get_starter_actuals(pk)   — starter Ks + outs from the box score (grading)

All responses are cached in-process per run; a daily pipeline run makes a few
hundred small requests at most.
"""
from datetime import date as _date

import requests

BASE = "https://statsapi.mlb.com/api/v1"

# 2024-25 MLB league-average per-PA rates; used as priors / fallbacks
LEAGUE = {"k": 0.223, "bb": 0.085, "hit": 0.212}

_cache = {}


def _get(path, **params):
    key = (path, tuple(sorted(params.items())))
    if key in _cache:
        return _cache[key]
    r = requests.get(f"{BASE}{path}", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    _cache[key] = data
    return data


def get_schedule(game_date=None):
    """List of games with probables and (when posted) confirmed lineups.

    Returns: [{game_pk, game_date, home_team, away_team, venue,
               home_pitcher: {id, name, hand}|None, away_pitcher: ...,
               home_lineup: [batter_id,...], away_lineup: [...]}]
    """
    game_date = game_date or _date.today().isoformat()
    data = _get(
        "/schedule",
        sportId=1,
        date=game_date,
        hydrate="probablePitcher,lineups,team",
    )
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("gameType") not in ("R", "P", "F", "D", "L", "W"):
                continue  # skip exhibitions
            lineups = g.get("lineups", {}) or {}
            games.append({
                "game_pk": g["gamePk"],
                "game_date": game_date,
                "status": g.get("status", {}).get("abstractGameState"),
                "home_team": g["teams"]["home"]["team"]["name"],
                "away_team": g["teams"]["away"]["team"]["name"],
                "venue": (g.get("venue") or {}).get("name"),
                "home_pitcher": _probable(g["teams"]["home"]),
                "away_pitcher": _probable(g["teams"]["away"]),
                # lineup the HOME pitcher faces is the AWAY lineup, and vice versa
                "home_lineup": [p["id"] for p in lineups.get("homePlayers", [])],
                "away_lineup": [p["id"] for p in lineups.get("awayPlayers", [])],
            })
    return games


def _probable(team_side):
    pp = team_side.get("probablePitcher")
    if not pp:
        return None
    return {"id": pp["id"], "name": pp.get("fullName"), "hand": _pitch_hand(pp["id"])}


def _pitch_hand(pitcher_id):
    data = _get(f"/people/{pitcher_id}")
    people = data.get("people", [])
    if not people:
        return "R"
    return (people[0].get("pitchHand") or {}).get("code", "R")


def _bat_side(batter_id):
    data = _get(f"/people/{batter_id}")
    people = data.get("people", [])
    if not people:
        return "R"
    return (people[0].get("batSide") or {}).get("code", "R")


def get_pitcher_profile(pitcher_id, season=None):
    """Season per-PA rates + recent workload for the leash model.

    Returns {k, bb, hit, batters_faced, recent_pitch_counts: [ints]}
    Rates fall back toward league average when the sample is thin
    (simple shrinkage: (x + prior*50) / (bf + 50)).
    """
    season = season or _date.today().year
    stats = _get(
        f"/people/{pitcher_id}/stats",
        stats="season", group="pitching", season=season, sportId=1,
    )
    k = bb = hits = bf = 0
    for grp in stats.get("stats", []):
        for split in grp.get("splits", []):
            s = split.get("stat", {})
            k += int(s.get("strikeOuts", 0))
            bb += int(s.get("baseOnBalls", 0)) + int(s.get("hitByPitch", 0))
            hits += int(s.get("hits", 0))
            bf += int(s.get("battersFaced", 0))

    def shrink(x, prior, n, weight=50):
        return (x + prior * weight) / (n + weight) if (n + weight) > 0 else prior

    profile = {
        "k": shrink(k, LEAGUE["k"], bf, 50) if bf else LEAGUE["k"],
        "bb": shrink(bb, LEAGUE["bb"], bf, 50) if bf else LEAGUE["bb"],
        "hit": shrink(hits, LEAGUE["hit"], bf, 50) if bf else LEAGUE["hit"],
        "batters_faced": bf,
        "recent_pitch_counts": _recent_pitch_counts(pitcher_id, season),
    }
    return profile


def _recent_pitch_counts(pitcher_id, season, n_starts=5):
    """Pitch counts from the last n starts (gamesStarted only)."""
    logs = _get(
        f"/people/{pitcher_id}/stats",
        stats="gameLog", group="pitching", season=season, sportId=1,
    )
    counts = []
    for grp in logs.get("stats", []):
        for split in grp.get("splits", []):
            s = split.get("stat", {})
            if int(s.get("gamesStarted", 0)) < 1:
                continue
            np_ = s.get("numberOfPitches") or s.get("pitchesThrown")
            if np_:
                counts.append((split.get("date"), int(np_)))
    counts.sort()  # by date ascending
    return [c for _, c in counts[-n_starts:]]


def get_batter_rates(batter_id, pitcher_hand, season=None):
    """Season per-PA K/BB/hit rates vs the given pitcher hand ('L'/'R').

    Uses StatsAPI statSplits (sitCodes vl/vr); falls back to overall season
    rates, then league average. Shrunk toward league by 100 PA of prior.
    """
    season = season or _date.today().year
    sit = "vl" if pitcher_hand == "L" else "vr"
    try:
        data = _get(
            f"/people/{batter_id}/stats",
            stats="statSplits", group="hitting", season=season,
            sitCodes=sit, sportId=1,
        )
        rates = _hitting_rates(data)
        if rates:
            return rates
    except requests.RequestException:
        pass
    try:
        data = _get(
            f"/people/{batter_id}/stats",
            stats="season", group="hitting", season=season, sportId=1,
        )
        rates = _hitting_rates(data)
        if rates:
            return rates
    except requests.RequestException:
        pass
    return dict(LEAGUE)


def _hitting_rates(data, prior_weight=100):
    k = bb = hits = pa = 0
    for grp in data.get("stats", []):
        for split in grp.get("splits", []):
            s = split.get("stat", {})
            k += int(s.get("strikeOuts", 0))
            bb += int(s.get("baseOnBalls", 0)) + int(s.get("hitByPitch", 0))
            hits += int(s.get("hits", 0))
            pa += int(s.get("plateAppearances", 0))
    if pa == 0:
        return None
    w = prior_weight
    return {
        "k": (k + LEAGUE["k"] * w) / (pa + w),
        "bb": (bb + LEAGUE["bb"] * w) / (pa + w),
        "hit": (hits + LEAGUE["hit"] * w) / (pa + w),
    }


def get_home_plate_umpire(game_pk):
    """Home-plate umpire full name from the box score, or None if unassigned."""
    try:
        box = _get(f"/game/{game_pk}/boxscore")
    except requests.RequestException:
        return None
    for off in box.get("officials", []):
        if off.get("officialType") == "Home Plate":
            return (off.get("official") or {}).get("fullName")
    return None


def get_starter_actuals(game_pk):
    """Actual Ks and outs for both starters from the box score.

    Returns {pitcher_id: {"name":..., "ks": int, "outs": int}} for the two
    starting pitchers (first pitcher listed per team).
    """
    data = _get(f"/game/{game_pk}/boxscore")
    out = {}
    for side in ("home", "away"):
        team = data.get("teams", {}).get(side, {})
        pitcher_ids = team.get("pitchers", [])
        if not pitcher_ids:
            continue
        starter_id = pitcher_ids[0]  # listed in appearance order
        pdata = team.get("players", {}).get(f"ID{starter_id}", {})
        stats = pdata.get("stats", {}).get("pitching", {})
        ip = stats.get("inningsPitched", "0.0")
        out[starter_id] = {
            "name": (pdata.get("person") or {}).get("fullName"),
            "ks": int(stats.get("strikeOuts", 0)),
            "outs": innings_to_outs(ip),
        }
    return out


def innings_to_outs(ip_str):
    """MLB innings notation: '5.2' = 5 innings + 2 outs = 17 outs."""
    try:
        whole, _, frac = str(ip_str).partition(".")
        return int(whole) * 3 + (int(frac) if frac else 0)
    except (ValueError, TypeError):
        return 0
