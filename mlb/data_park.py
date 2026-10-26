import os
import json
from datetime import datetime, timezone
import pandas as pd
from pybaseball import statcast

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "cache", "park_factors.json")

SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "missed_bunt",
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
}
IN_ZONE      = {1, 2, 3, 4, 5, 6, 7, 8, 9}
HIT_EVENTS   = {"single", "double", "triple", "home_run"}
HARD_HIT_MPH = 95


# aggregate pitch-level statcast rows → one row per game
# return pd.Series
def _agg_game(g):
    swings         = g[g["description"].isin(SWING_DESCS)]
    whiffs         = g[g["description"].isin({"swinging_strike", "swinging_strike_blocked"})]
    outside        = g[~g["zone"].isin(IN_ZONE)]
    outside_swings = outside[outside["description"].isin(SWING_DESCS)]
    bip            = g[g["launch_speed"].notna()]
    hard_hits      = bip[bip["launch_speed"] >= HARD_HIT_MPH]

    return pd.Series({
        "ks":            (g["events"] == "strikeout").sum(),
        "walks":         (g["events"] == "walk").sum(),
        "hrs":           (g["events"] == "home_run").sum(),
        "hits":          g["events"].isin(HIT_EVENTS).sum(),
        "whiff_rate":    len(whiffs) / len(swings)          if len(swings)  > 0 else None,
        "chase_rate":    len(outside_swings) / len(outside) if len(outside) > 0 else None,
        "hard_hit_rate": len(hard_hits) / len(bip)          if len(bip)     > 0 else None,
    })

# compute whole regular season(s) factors
# return dict of all teams with factors
def _compute_park_factors(season=2024):
    df = statcast(f"{season}-03-20", f"{season}-10-01")

    if df.empty:
        return {}

    # one row per game, tagged with home_team (proxy for park)
    game_stats = (
        df.groupby(["game_pk", "home_team"])
        .apply(_agg_game, include_groups=False)
        .reset_index()
    )

    # average each stat across all games played at that park
    cols     = ["ks", "walks", "hrs", "hits", "whiff_rate", "chase_rate", "hard_hit_rate"]
    park_avg = game_stats.groupby("home_team")[cols].mean()

    # divide by league average → factor (1.0 = league average)
    league_avg = park_avg.mean()
    factors    = (park_avg / league_avg).round(3)

    return {
        team: {
            "k_factor":        row["ks"],
            "walk_factor":     row["walks"],
            "hr_factor":       row["hrs"],
            "hit_factor":      row["hits"],
            "whiff_factor":    row["whiff_rate"],
            "chase_factor":    row["chase_rate"],
            "hard_hit_factor": row["hard_hit_rate"],
        }
        for team, row in factors.iterrows()
    }

# compute and cache factors
# return dict of factors for all teams
def build_park_factors(season=2024):
    # Compute and cache park factors for a full season. Run once per season
    factors = _compute_park_factors(season)
    for entry in factors.values():
        entry["created_at"] = datetime.now(timezone.utc).isoformat()
        entry["through_date"] = f"{season}-12-31"
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(factors, f, indent=2)
    print(f"Saved park factors for {len(factors)} teams → {CACHE_PATH}")
    return factors

# get factors
# return dict of factors
def get_park_factors(home_team):
    # Load cached park factors for a team. Returns dict of 7 factors (1.0 = neutral)
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            "Park factors cache not found. Run build_park_factors() first."
        )
    with open(CACHE_PATH) as f:
        all_factors = json.load(f)

    # fall back to 1.0 (neutral) if team not found
    return all_factors.get(home_team, {
        "k_factor":        1.0,
        "walk_factor":     1.0,
        "hr_factor":       1.0,
        "hit_factor":      1.0,
        "whiff_factor":    1.0,
        "chase_factor":    1.0,
        "hard_hit_factor": 1.0,
    })


if __name__ == "__main__":
    factors = build_park_factors(2024)
    for team, f in sorted(factors.items()):
        print(f"{team:4s}  k={f['k_factor']}  hr={f['hr_factor']}  hit={f['hit_factor']}  whiff={f['whiff_factor']}")
