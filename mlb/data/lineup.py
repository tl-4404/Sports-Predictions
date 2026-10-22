from pybaseball import statcast_batter
import pandas as pd

PITCH_TYPES    = ["FF", "SI", "FC", "SL", "ST", "CH", "FS", "CU", "KC"]
MIN_SAMPLE     = 25
RATE_TYPES     = ["k_rate", "whiff_rate", "chase_rate"]
LINEUP_WEIGHTS = {
    1: 1.00, 2: 0.97, 3: 0.95, 4: 0.93,
    5: 0.90, 6: 0.87, 7: 0.84, 8: 0.81, 9: 0.78
}

# swing descriptions — any of these means the batter offered at the pitch
SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "missed_bunt",
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
}
# zones 1-9 are in the strike zone, 11-14 are outside
IN_ZONE = {1, 2, 3, 4, 5, 6, 7, 8, 9}


# build features for a single batter
# p_throws filters to only PAs against that hand ("L" or "R"); None = all PAs
# if df is provided, slices it instead of calling the API
def get_batter_pitch_features(batter_id, date, p_throws=None, df=None):
    date = pd.Timestamp(date)

    if df is not None:
        # slice the pre-loaded DataFrame — no API call
        start  = date - pd.DateOffset(months=12)
        subset_df = df[
            (df["batter"] == batter_id) &
            (pd.to_datetime(df["game_date"]) >= start) &
            (pd.to_datetime(df["game_date"]) < date)
        ]
    else:
        # fetch from API
        start_date_str = (date - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
        end_date_str   = date.strftime("%Y-%m-%d")
        subset_df = statcast_batter(start_date_str, end_date_str, batter_id)

    if subset_df.empty:
        return {}

    # filter to PAs against the specified pitcher hand
    if p_throws is not None and "p_throws" in subset_df.columns:
        subset_df = subset_df[subset_df["p_throws"] == p_throws]
        if subset_df.empty:
            return {}

    features = {}
    for pitch in PITCH_TYPES:
        subset = subset_df[subset_df["pitch_type"] == pitch]
        p = pitch.lower()

        if len(subset) < MIN_SAMPLE:
            # not enough data — store None, will handle at training time
            features[f"k_rate_vs_{p}"]     = None
            features[f"whiff_rate_vs_{p}"] = None
            features[f"chase_rate_vs_{p}"] = None
            continue

        # K rate
        features[f"k_rate_vs_{p}"] = round(
            (subset["events"] == "strikeout").sum() / len(subset), 4
        )

        # whiff rate
        swings = subset[subset["description"].isin(SWING_DESCS)]
        whiffs = subset[subset["description"].isin(
            {"swinging_strike", "swinging_strike_blocked"}
        )]
        features[f"whiff_rate_vs_{p}"] = round(
            len(whiffs) / len(swings), 4
        ) if len(swings) > 0 else 0.0

        # chase rate
        outside        = subset[~subset["zone"].isin(IN_ZONE)]
        outside_swings = outside[outside["description"].isin(SWING_DESCS)]
        features[f"chase_rate_vs_{p}"] = round(
            len(outside_swings) / len(outside), 4
        ) if len(outside) > 0 else 0.0

    return features


# build position-weighted average features across the lineup
# p_throws filters batter stats to only PAs vs that hand ("L" or "R")
# batter_lookup: dict of {batter_id: pre-filtered DataFrame} for fast access
# if batter_lookup is provided it takes precedence over df
def get_lineup_features(lineup, date, p_throws=None, df=None, batter_lookup=None):
    # one accumulator per feature: {feature: [weighted_sum, total_weight]}
    accum = {
        f"{r}_vs_{p.lower()}": [0.0, 0.0]
        for p in PITCH_TYPES
        for r in RATE_TYPES
    }

    for pos, batter_id in lineup:
        batter_df = batter_lookup.get(batter_id, pd.DataFrame()) if batter_lookup else df
        batter_features = get_batter_pitch_features(batter_id, date, p_throws=p_throws, df=batter_df)
        weight = LINEUP_WEIGHTS[pos]
        for p in PITCH_TYPES:
            for r in RATE_TYPES:
                key = f"{r}_vs_{p.lower()}"
                val = batter_features.get(key)
                if val is None:
                    continue
                accum[key][0] += val * weight
                accum[key][1] += weight

    return {
        f"lineup_{key}": round(wsum / wtot, 4) if wtot > 0 else None
        for key, (wsum, wtot) in accum.items()
    }


if __name__ == "__main__":
    # Manny Machado MLBAM ID = 592518
    feats = get_batter_pitch_features(592518, pd.Timestamp("2024-09-01"))
    for k, v in feats.items():
        print(f"{k}: {v}")
