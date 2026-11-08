"""Home-plate umpire K factors.

Serving side (fast, free MLB StatsAPI):
    get_home_plate_umpire(game_pk) -> "John Smith" | None

Training side (run once per season on your own machine — makes ~2500 HTTP
calls; NOT imported at serving time):
    build_ump_factors([2023, 2024, 2025]) -> writes cache/ump_factors.json

The factor is a shrunk ratio of the strikeouts-per-plate-appearance seen while
that umpire worked the plate to the league average over the same games. Values
> 1.0 mean a strikeout-friendly umpire (wider zone). Shrinkage pulls
low-sample umpires toward 1.0 so a few games can't produce an extreme factor.
"""
import json
import os
from datetime import date, timedelta

import requests

from mlb.paths import CACHE_DIR

BASE = "https://statsapi.mlb.com/api/v1"
UMP_PATH = CACHE_DIR / "ump_factors.json"

# prior strength for shrinkage, in plate appearances (~2 games' worth)
PRIOR_PA = 500


def _get(path, **params):
    r = requests.get(f"{BASE}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# serving-time lookup lives in mlb.data_fetching.providers.mlb_api.get_home_plate_umpire; this module is
# the training-side builder that produces cache/ump_factors.json.

# ── training-side builder ────────────────────────────────────────────────────

def _final_games(game_date):
    data = _get("/schedule", sportId=1, date=game_date, hydrate="officials")
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("codedGameState") == "F":
                yield g


def _game_k_and_pa(game_pk):
    """Total strikeouts and batters faced by both pitching staffs."""
    box = _get(f"/game/{game_pk}/boxscore")
    ks = pa = 0
    for side in ("home", "away"):
        team = box.get("teams", {}).get(side, {})
        pdata = team.get("players", {})
        for pid in team.get("pitchers", []):
            stat = pdata.get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
            ks += int(stat.get("strikeOuts", 0))
            pa += int(stat.get("battersFaced", 0))
    hp = None
    for off in box.get("officials", []):
        if off.get("officialType") == "Home Plate":
            hp = (off.get("official") or {}).get("fullName")
    return hp, ks, pa


def build_ump_factors(seasons=(2023, 2024, 2025), season_windows=None):
    """Compute + cache umpire K factors. Heavyweight — run once per season."""
    windows = season_windows or {s: (f"{s}-03-20", f"{s}-10-01") for s in seasons}
    tally = {}  # ump -> [ks, pa, games]
    tot_k = tot_pa = 0

    for season in seasons:
        start, end = windows[season]
        d0 = date.fromisoformat(start)
        d1 = date.fromisoformat(end)
        day = d0
        while day <= d1:
            ds = day.isoformat()
            try:
                games = list(_final_games(ds))
            except requests.RequestException:
                games = []
            for g in games:
                try:
                    hp, ks, pa = _game_k_and_pa(g["gamePk"])
                except requests.RequestException:
                    continue
                if not hp or pa == 0:
                    continue
                t = tally.setdefault(hp, [0, 0, 0])
                t[0] += ks
                t[1] += pa
                t[2] += 1
                tot_k += ks
                tot_pa += pa
            print(f"  {ds}: {len(games)} finals, {len(tally)} umps so far")
            day += timedelta(days=1)

    if tot_pa == 0:
        print("[data_umpire] no data fetched — nothing written")
        return {}

    league_rate = tot_k / tot_pa
    factors = {}
    for ump, (ks, pa, games) in tally.items():
        # shrink the per-PA rate toward league before taking the ratio
        shrunk_rate = (ks + league_rate * PRIOR_PA) / (pa + PRIOR_PA)
        factors[ump] = {
            "k_factor": round(shrunk_rate / league_rate, 4),
            "games": games,
            "pa": pa,
        }

    os.makedirs(UMP_PATH.parent, exist_ok=True)
    with open(UMP_PATH, "w") as f:
        json.dump(factors, f, indent=2, sort_keys=True)
    print(f"[data_umpire] wrote {len(factors)} umpire factors → {UMP_PATH} "
          f"(league {league_rate:.4f} K/PA)")
    return factors


if __name__ == "__main__":
    build_ump_factors()
