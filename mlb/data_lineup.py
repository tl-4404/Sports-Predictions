from pybaseball import statcast_batter
import pandas as pd
from datetime import date, datetime

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

# build features for batter
# return dict of all rates for each pitch
def get_batter_pitch_features(batter_id, date):
    start_date_str = (date - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    end_date_str   = date.strftime("%Y-%m-%d")
    df = statcast_batter(start_date_str, end_date_str, batter_id)

    if df.empty:
        return {}

    features = {}
    for pitch in PITCH_TYPES:
        subset = df[df["pitch_type"] == pitch]
        p = pitch.lower()

        if len(subset) < MIN_SAMPLE:
            # not enough data — store None
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
        outside = subset[~subset["zone"].isin(IN_ZONE)]
        outside_swings = outside[outside["description"].isin(SWING_DESCS)]
        features[f"chase_rate_vs_{p}"] = round(
            len(outside_swings) / len(outside), 4
        ) if len(outside) > 0 else 0.0

    return features

# build features for entire lineup
# return dict of all rates for each pitch weighted (based on batting pos)
def get_lineup_features(lineup, date):
    # one accumulator per feature: {feature: [weighted_sum, total_weight]}
    accum = {
        f"{r}_vs_{p.lower()}": [0.0, 0.0]
        for p in PITCH_TYPES
        for r in RATE_TYPES
    }
    # get rate for each pitch type (weighted)
    for pos, batter_id in lineup:
        batter_features = get_batter_pitch_features(batter_id, date)
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
