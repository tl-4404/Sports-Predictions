"""MLB StatsAPI provider (statsapi.mlb.com).

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
import os

import requests

BASE = "https://statsapi.mlb.com/api/v1"

# 2024-25 MLB league-average per-PA rates; used as priors / fallbacks
LEAGUE = {"k": 0.223, "bb": 0.085, "hit": 0.212}

# Observed rates are left untouched after these sample sizes.  Before the
# threshold, the unobserved portion is supplied by the player's previous
# season; league average is only the final fallback for missing history.
PITCHER_RATE_THRESHOLD = int(os.getenv("MLB_PITCHER_RATE_THRESHOLD", "200"))
BATTER_SPLIT_RATE_THRESHOLD = int(
    os.getenv("MLB_BATTER_SPLIT_RATE_THRESHOLD", "150")
)
BATTER_OVERALL_RATE_THRESHOLD = int(
    os.getenv("MLB_BATTER_OVERALL_RATE_THRESHOLD", "250")
)

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

    Returns {k, bb, hit, batters_faced, recent_pitch_counts: [ints]}.

    A current-season sample at or above PITCHER_RATE_THRESHOLD is used as
    observed.  A smaller sample is filled to that threshold with the player's
    previous-season rate.  The previous season is itself credibility-weighted
    against league average when it is short, so rookies and tiny historical
    samples still have a stable fallback without diluting every veteran.
    """
    season = season or _date.today().year
    current_data = _get(
        f"/people/{pitcher_id}/stats",
        stats="season", group="pitching", season=season, sportId=1,
    )
    current = _pitching_totals(current_data)

    try:
        previous_data = _get(
            f"/people/{pitcher_id}/stats",
            stats="season", group="pitching", season=season - 1, sportId=1,
        )
        previous = _pitching_totals(previous_data)
    except requests.RequestException:
        previous = _empty_totals()

    historical_prior = _credible_rates(previous, PITCHER_RATE_THRESHOLD)
    rates = _threshold_rates(
        current, historical_prior, PITCHER_RATE_THRESHOLD
    )

    profile = {
        **rates,
        "batters_faced": current["n"],
        "prior_batters_faced": previous["n"],
        "recent_pitch_counts": _recent_pitch_counts(pitcher_id, season),
    }
    return profile


def _empty_totals():
    return {"k": 0, "bb": 0, "hit": 0, "n": 0}


def _pitching_totals(data):
    totals = _empty_totals()
    for grp in data.get("stats", []):
        for split in grp.get("splits", []):
            stat = split.get("stat", {})
            totals["k"] += int(stat.get("strikeOuts", 0))
            totals["bb"] += (
                int(stat.get("baseOnBalls", 0))
                + int(stat.get("hitByPitch", 0))
            )
            totals["hit"] += int(stat.get("hits", 0))
            totals["n"] += int(stat.get("battersFaced", 0))
    return totals


def _credible_rates(totals, threshold):
    """Historical rate prior, discounted toward league when history is thin."""
    n = totals["n"]
    if n <= 0:
        return dict(LEAGUE)
    credibility = min(n / float(threshold), 1.0)
    return {
        event: credibility * (totals[event] / n)
        + (1.0 - credibility) * LEAGUE[event]
        for event in LEAGUE
    }


def _threshold_rates(totals, prior_rates, threshold):
    """Use observed rates after threshold; fill only a small sample's gap."""
    n = totals["n"]
    if n <= 0:
        return dict(prior_rates)
    if n >= threshold:
        return {event: totals[event] / n for event in LEAGUE}
    missing = threshold - n
    return {
        event: (totals[event] + prior_rates[event] * missing) / threshold
        for event in LEAGUE
    }


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
    rates, then league average. Small current-season samples use the same
    hitter's prior-season split/overall rate before falling back to league.
    """
    season = season or _date.today().year
    sit = "vl" if pitcher_hand == "L" else "vr"
    try:
        data = _get(
            f"/people/{batter_id}/stats",
            stats="statSplits", group="hitting", season=season,
            sitCodes=sit, sportId=1,
        )
        current = _hitting_totals(data)
        previous = _previous_hitting_totals(batter_id, season, sit)
        if not previous["n"]:
            previous = _previous_hitting_totals(batter_id, season, None)
        if current["n"]:
            prior = _credible_rates(previous, BATTER_SPLIT_RATE_THRESHOLD)
            return _threshold_rates(current, prior, BATTER_SPLIT_RATE_THRESHOLD)
        if previous["n"]:
            return _credible_rates(previous, BATTER_SPLIT_RATE_THRESHOLD)
    except requests.RequestException:
        pass
    try:
        data = _get(
            f"/people/{batter_id}/stats",
            stats="season", group="hitting", season=season, sportId=1,
        )
        current = _hitting_totals(data)
        if current["n"]:
            previous = _previous_hitting_totals(batter_id, season, None)
            prior = _credible_rates(previous, BATTER_OVERALL_RATE_THRESHOLD)
            return _threshold_rates(current, prior, BATTER_OVERALL_RATE_THRESHOLD)
    except requests.RequestException:
        pass
    return dict(LEAGUE)


def _previous_hitting_totals(batter_id, season, sit):
    params = {
        "stats": "statSplits" if sit else "season",
        "group": "hitting",
        "season": season - 1,
        "sportId": 1,
    }
    if sit:
        params["sitCodes"] = sit
    try:
        data = _get(f"/people/{batter_id}/stats", **params)
    except requests.RequestException:
        return _empty_totals()
    return _hitting_totals(data)


def _hitting_totals(data):
    totals = _empty_totals()
    for grp in data.get("stats", []):
        for split in grp.get("splits", []):
            s = split.get("stat", {})
            totals["k"] += int(s.get("strikeOuts", 0))
            totals["bb"] += (
                int(s.get("baseOnBalls", 0))
                + int(s.get("hitByPitch", 0))
            )
            totals["hit"] += int(s.get("hits", 0))
            totals["n"] += int(s.get("plateAppearances", 0))
    return totals


def _hitting_rates(data, prior_weight=100):
    """Backward-compatible helper retained for callers outside this module."""
    totals = _hitting_totals(data)
    if totals["n"] == 0:
        return None
    threshold = totals["n"] + prior_weight
    return _threshold_rates(totals, LEAGUE, threshold)


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
