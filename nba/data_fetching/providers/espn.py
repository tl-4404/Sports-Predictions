"""ESPN NBA injuries, filtered with NBA Stats top-scorer data."""
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from nba.data_fetching.providers.nba_api import fetch_player_stats

ESPN_TEAM_IDS = {
    "Atlanta Hawks": 1,           "Boston Celtics": 2,          "Brooklyn Nets": 17,
    "Charlotte Hornets": 30,      "Chicago Bulls": 4,           "Cleveland Cavaliers": 5,
    "Dallas Mavericks": 6,        "Denver Nuggets": 7,          "Detroit Pistons": 8,
    "Golden State Warriors": 9,   "Houston Rockets": 10,        "Indiana Pacers": 11,
    "Los Angeles Clippers": 12,   "Los Angeles Lakers": 13,     "Memphis Grizzlies": 29,
    "Miami Heat": 14,             "Milwaukee Bucks": 15,        "Minnesota Timberwolves": 16,
    "New Orleans Pelicans": 3,    "New York Knicks": 18,        "Oklahoma City Thunder": 25,
    "Orlando Magic": 19,          "Philadelphia 76ers": 20,     "Phoenix Suns": 21,
    "Portland Trail Blazers": 22, "Sacramento Kings": 23,       "San Antonio Spurs": 24,
    "Toronto Raptors": 28,        "Utah Jazz": 26,              "Washington Wizards": 27,
}

SUPPRESS_STATUSES = {"Out", "Doubtful"}
WARN_STATUSES     = {"Questionable", "Day-To-Day", "Probable"}
ALL_STATUSES      = SUPPRESS_STATUSES | WARN_STATUSES

BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"

# cached at first use
_top_players: set[str] | None = None

def get_top_players(n=50, min_gp=10, season="2025-26") -> set[str]:
    """
    Returns a set of lowercase player names for the top N players
    by PPG this season (minimum min_gp games played).
    Cached after first call.
    """
    global _top_players
    if _top_players is not None:
        return _top_players
    try:
        df = fetch_player_stats(season)
        df = df[df["GP"] >= min_gp]
        df = df.sort_values("PTS", ascending=False).head(n)
        _top_players = set(df["PLAYER_NAME"].str.lower())
    except Exception:
        _top_players = set()
    return _top_players

def _fetch(url, timeout=8):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _resolve_injury(ref_url, top_players):
    """Fetch one injury ref. Returns {name, status} only if the player
    is in top_players and has a notable status. Returns None otherwise."""
    try:
        inj = _fetch(ref_url)
        status = inj.get("status", "")
        if status not in ALL_STATUSES:
            return None
        athlete_ref = inj.get("athlete", {}).get("$ref", "")
        if not athlete_ref:
            return None
        athlete = _fetch(athlete_ref)
        name = athlete.get("displayName", "")
        if not name or name.lower() not in top_players:
            return None
        return {"name": name, "status": status}
    except Exception:
        return None

def get_team_injuries(team_name):
    """
    Returns list of {name, status} for players who are Out/Doubtful/
    Questionable/Day-To-Day AND are in the top 100 players by PPG.
    Returns [] on any failure.
    """
    team_id = ESPN_TEAM_IDS.get(team_name)
    if not team_id:
        return []
    top_players = get_top_players()
    try:
        data = _fetch(f"{BASE}/teams/{team_id}/injuries?limit=25")
        refs = [item["$ref"] for item in data.get("items", []) if "$ref" in item]
        if not refs:
            return []
        results = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_resolve_injury, ref, top_players) for ref in refs]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results
    except Exception:
        return []

def injury_note(injuries):
    """Serialize injury list to a string for CSV storage."""
    return "; ".join(f"{p['name']}:{p['status']}" for p in injuries)


def get_team_injury_snapshot(team_name):
    """Agent evidence: all returned injuries, with failures distinct from no reports.

    Unlike the legacy UI helper this does not silently drop non-top-50 players.
    'available' describes source access, not a guarantee of complete medical news.
    """
    from datetime import datetime, timezone
    report = {"provider": "espn", "team": team_name, "status": "unavailable",
              "injuries": [], "errors": []}
    try:
        team_id = ESPN_TEAM_IDS[team_name]
        report["url"] = f"{BASE}/teams/{team_id}/injuries?limit=100"
        data = _fetch(report["url"])
        items = data["items"]
        if not isinstance(items, list) or data.get("count", len(items)) > len(items):
            raise ValueError("Incomplete ESPN injury pagination")
        for item in items:
            injury = _fetch(item["$ref"])
            athlete = _fetch(injury["athlete"]["$ref"])
            name, status = athlete.get("displayName"), injury.get("status")
            if not name or not status:
                raise ValueError("Incomplete ESPN injury record")
            report["injuries"].append({"name": name, "status": status,
                                        "source_url": item["$ref"]})
        report["status"] = "available"
    except Exception as exc:
        report["errors"].append(type(exc).__name__)
    report["observed_at"] = datetime.now(timezone.utc).isoformat()
    return report

def parse_injury_note(note):
    """Deserialize injury string back to list of dicts."""
    if not note or not note.strip():
        return []
    result = []
    for entry in note.split(";"):
        entry = entry.strip()
        if ":" in entry:
            name, status = entry.split(":", 1)
            result.append({"name": name.strip(), "status": status.strip()})
    return result
