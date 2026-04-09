from pybaseball import statcast_pitcher
import pandas as pd


def _agg_game(g):
    # aggregate pitch-level rows → one row per game
    pitch_counts = g["pitch_type"].value_counts(normalize=True)

    def velo(pitch):
        return g[g["pitch_type"] == pitch]["release_speed"].mean()

    ff_velo = velo("FF")
    si_velo = velo("SI")
    # use sinker velo as fallback if no 4-seam (some pitchers are sinker-primary)
    fb_velo = ff_velo if not pd.isna(ff_velo) else si_velo

    return pd.Series({
        "game_date":     g["game_date"].iloc[0],
        "total_pitches": len(g),
        "k_total":       (g["events"] == "strikeout").sum(),
        "ff_pct":        pitch_counts.get("FF", 0),
        "si_pct":        pitch_counts.get("SI", 0),
        "fc_pct":        pitch_counts.get("FC", 0),
        "sl_pct":        pitch_counts.get("SL", 0),
        "st_pct":        pitch_counts.get("ST", 0),
        "ch_pct":        pitch_counts.get("CH", 0),
        "fs_pct":        pitch_counts.get("FS", 0),
        "cu_pct":        pitch_counts.get("CU", 0),
        "kc_pct":        pitch_counts.get("KC", 0),
        "ff_velo":       ff_velo,
        "si_velo":       velo("SI"),
        "fc_velo":       velo("FC"),
        "sl_velo":       velo("SL"),
        "st_velo":       velo("ST"),
        "ch_velo":       velo("CH"),
        "fs_velo":       velo("FS"),
        "cu_velo":       velo("CU"),
        "fb_ch_gap":     fb_velo - velo("CH"),
        "fb_fs_gap":     fb_velo - velo("FS"),
    })


# returns the pitcher's last n_starts game logs before end_date
# if df is provided, slices it instead of calling the API
def get_pitcher_game_logs(pitcher_id, end_date, n_starts=5, df=None):
    end_date = pd.Timestamp(end_date)

    if df is not None:
        # slice the pre-loaded DataFrame — no API call
        prior = df[
            (df["pitcher"] == pitcher_id) &
            (pd.to_datetime(df["game_date"]) < end_date)
        ]
    else:
        # fetch from API
        start_date_str = (end_date - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        end_date_str   = end_date.strftime("%Y-%m-%d")
        prior = statcast_pitcher(start_date_str, end_date_str, pitcher_id)

    if prior.empty:
        return pd.DataFrame()

    logs = (
        prior.groupby("game_pk")
        .apply(_agg_game, include_groups=False)
        .reset_index(drop=True)
        .sort_values("game_date")
    )

    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs[logs["game_date"] < end_date]
    return logs.tail(n_starts)


# returns pitcher features as a dict — pass df to skip the API call
# includes p_throws: 0 = left-handed, 1 = right-handed
def get_pitcher_features(pitcher_id, date, n_starts=5, df=None):
    logs = get_pitcher_game_logs(pitcher_id, date, n_starts, df=df)

    if logs.empty:
        return {}

    numeric_cols = logs.select_dtypes(include="number").columns
    means = logs[numeric_cols].mean()
    feats = {f"{col}_last{n_starts}": round(val, 4) for col, val in means.items()}

    # add pitcher handedness — look it up from the source data
    if df is not None:
        hand = df[df["pitcher"] == pitcher_id]["p_throws"].iloc[0]
    else:
        from pybaseball import statcast_pitcher as sp
        end_str   = pd.Timestamp(date).strftime("%Y-%m-%d")
        start_str = (pd.Timestamp(date) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        raw = sp(start_str, end_str, pitcher_id)
        hand = raw["p_throws"].iloc[0] if not raw.empty else "R"

    feats["p_throws"] = 0 if hand == "L" else 1
    return feats


if __name__ == "__main__":
    # Paul Skenes
    logs = get_pitcher_game_logs(694973, pd.Timestamp("2024-09-01"))
    print(logs)
