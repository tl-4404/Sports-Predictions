"""Serving-side park & umpire K factors (no heavy imports).

Both read small JSON caches produced by training-side builders and fall back
to 1.0 (neutral) when a team/umpire is missing or the cache doesn't exist —
so the serving pipeline never hard-fails on a missing factor.

  cache/park_factors.json : {team_abbr: {"k_factor": .., ...}}   (data/park.py)
  cache/ump_factors.json  : {ump_name:  {"k_factor": .., "games": ..}} (data/umpire.py)

A K factor > 1.0 means more strikeouts than league average (bigger zone /
hitter's park effect on Ks); the simulator scales the K odds by the product
of the park and umpire factors.
"""
import json
import unicodedata

from mlb.paths import CACHE_DIR

PARK_PATH = CACHE_DIR / "park_factors.json"
UMP_PATH = CACHE_DIR / "ump_factors.json"

# full team name (StatsAPI) → statcast/park-factor abbreviation
TEAM_ABBR = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "OAK",
    "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "Seattle Mariners": "SEA", "San Francisco Giants": "SF",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

_park_cache = None
_ump_cache = None


def _load(path):
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _norm_name(name):
    # strip accents (José -> jose) so name spellings match across sources
    decomposed = unicodedata.normalize("NFKD", name or "")
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(ch for ch in ascii_only.lower() if ch.isalpha())


def park_k_factor(home_team_name):
    global _park_cache
    if _park_cache is None:
        _park_cache = _load(PARK_PATH)
    abbr = TEAM_ABBR.get(home_team_name, "")
    return float((_park_cache.get(abbr) or {}).get("k_factor", 1.0))


def ump_k_factor(ump_name):
    """K factor for a home-plate umpire by full name; 1.0 if unknown."""
    global _ump_cache
    if _ump_cache is None:
        # index by normalized name so "José"/"Jose" and spacing don't matter
        raw = _load(UMP_PATH)
        _ump_cache = {_norm_name(k): v for k, v in raw.items()}
    if not ump_name:
        return 1.0
    entry = _ump_cache.get(_norm_name(ump_name))
    return float(entry.get("k_factor", 1.0)) if entry else 1.0


def _reset_caches():
    """Test hook — force re-read after a cache file is written."""
    global _park_cache, _ump_cache
    _park_cache = None
    _ump_cache = None
