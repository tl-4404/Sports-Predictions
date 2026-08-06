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
from datetime import datetime as _datetime

import requests

from timeutils import is_open_slate_event, slate_today

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
    game_date = game_date or slate_today()
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
                "status_code": g.get("status", {}).get("codedGameState"),
                "status_detail": g.get("status", {}).get("detailedState"),
                "start_time": g.get("gameDate"),
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


def get_pitcher_profile(pitcher_id, season=None, as_of_date=None):
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

    recent_starts = _recent_starts(
        pitcher_id, season, as_of_date=as_of_date
    )
    profile = {
        "k": shrink(k, LEAGUE["k"], bf, 50) if bf else LEAGUE["k"],
        "bb": shrink(bb, LEAGUE["bb"], bf, 50) if bf else LEAGUE["bb"],
        "hit": shrink(hits, LEAGUE["hit"], bf, 50) if bf else LEAGUE["hit"],
        "batters_faced": bf,
        "recent_pitch_counts": [
            start["pitches"] for start in recent_starts if start.get("pitches")
        ][-5:],
        "recent_starts": recent_starts,
        "workload": workload_assessment(recent_starts, as_of_date),
    }
    return profile


def _recent_starts(pitcher_id, season, n_starts=10, as_of_date=None):
    """Dated workload and performance evidence from recent starts."""
    logs = _get(
        f"/people/{pitcher_id}/stats",
        stats="gameLog", group="pitching", season=season, sportId=1,
    )
    starts = []
    for grp in logs.get("stats", []):
        for split in grp.get("splits", []):
            s = split.get("stat", {})
            if int(s.get("gamesStarted", 0)) < 1:
                continue
            start_date = split.get("date")
            if as_of_date and start_date and start_date >= as_of_date:
                continue
            np_ = s.get("numberOfPitches") or s.get("pitchesThrown")
            if not np_:
                continue
            starts.append({
                "date": start_date,
                "pitches": int(np_),
                "outs": innings_to_outs(s.get("inningsPitched", "0.0")),
                "batters_faced": int(s.get("battersFaced", 0) or 0),
                "earned_runs": int(s.get("earnedRuns", 0) or 0),
                "hits": int(s.get("hits", 0) or 0),
                "walks": int(s.get("baseOnBalls", 0) or 0)
                + int(s.get("hitByPitch", 0) or 0),
            })
    starts.sort(key=lambda row: row.get("date") or "")
    return starts[-n_starts:]


def _recent_pitch_counts(pitcher_id, season, n_starts=5):
    """Backward-compatible pitch-count helper."""
    return [
        row["pitches"]
        for row in _recent_starts(pitcher_id, season, n_starts=n_starts)
    ]


def workload_assessment(recent_starts, target_date=None):
    """Fail-closed starter-role and leash assessment.

    Recent *realized* pitch counts mix planned ceilings with bad-outing hooks,
    so the simulator receives the full start records.  This function only
    decides whether there is enough stable evidence to authorize a wager.
    """
    starts = list(recent_starts or [])
    counts = [int(row["pitches"]) for row in starts if row.get("pitches")]
    reasons = []
    if len(counts) < 3:
        reasons.append(f"only {len(counts)} recent starts with pitch counts")

    tail = counts[-3:]
    if tail and max(tail) < 80:
        reasons.append("recent usage is opener/limited-starter range (<80 pitches)")
    if len(tail) >= 3 and max(tail) - min(tail) > 24:
        reasons.append("recent pitch-count role is unstable")
    if len(tail) >= 3 and tail[-1] < 85:
        changes = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
        if all(change >= 7 for change in changes):
            reasons.append("pitch-count ramp is still in progress")

    if target_date and starts:
        latest = starts[-1].get("date")
        try:
            rest_days = (
                _datetime.fromisoformat(target_date).date()
                - _datetime.fromisoformat(latest).date()
            ).days
        except (TypeError, ValueError):
            rest_days = None
        if rest_days is None:
            reasons.append("last-start date unavailable")
        elif rest_days < 4:
            reasons.append(f"short rest ({rest_days} days)")
        elif rest_days > 10:
            reasons.append(f"return/layoff risk ({rest_days} days since last start)")
    else:
        rest_days = None

    return {
        "confirmed": not reasons,
        "reasons": reasons,
        "n_starts": len(counts),
        "recent_pitch_counts": counts[-5:],
        "rest_days": rest_days,
    }


def game_gate_reasons(game, game_date, now=None):
    """Reasons a game is not safe for a current pre-game prediction."""
    reasons = []
    if game.get("status") != "Preview":
        reasons.append(
            f"game status is {game.get('status_detail') or game.get('status') or 'unknown'}"
        )
    start_time = game.get("start_time")
    if not start_time:
        reasons.append("first-pitch time unavailable")
    elif not is_open_slate_event(start_time, game_date, now):
        reasons.append("game has started or timestamp is off the requested slate")
    return reasons


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
