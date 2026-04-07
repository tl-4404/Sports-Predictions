from pybaseball import statcast_pitcher
import pandas as pd


# Returns a DataFrame of the pitcher's last n_starts game logs
# before end_date. Each row is one start.
def get_pitcher_game_logs(pitcher_id, end_date, n_starts=5):
    start_date_str = (end_date - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    df = statcast_pitcher(start_date_str, end_date_str, pitcher_id)

    print("rows fetched:", len(df))
    if df.empty:
        return pd.DataFrame()

    # aggregate pitch-level rows → one row per game
    def agg_game(g):
        total = len(g)
        ks = (g["events"] == "strikeout").sum()
        pitch_counts = g["pitch_type"].value_counts(normalize=True)

        def velo(pitch):
            return g[g["pitch_type"] == pitch]["release_speed"].mean()
        ff_velo = velo("FF")
        si_velo = velo("SI")
        ch_velo = velo("CH")
        fc_velo = velo("FC")
        sl_velo = velo("SL")
        st_velo = velo("ST")
        fs_velo = velo("FS")
        cu_velo = velo("CU")

        # use sinker velo as fallback if no 4-seam (some pitchers are sinker-primary)
        fb_velo = ff_velo if not pd.isna(ff_velo) else si_velo

        return pd.Series({
            "game_date":      g["game_date"].iloc[0],
            "total_pitches":  total,
            "k_total":        ks,
            # pitch mix
            "ff_pct":         pitch_counts.get("FF", 0),  # 4-seam fastball
            "si_pct":         pitch_counts.get("SI", 0),  # sinker
            "fc_pct":         pitch_counts.get("FC", 0),  # cutter
            "sl_pct":         pitch_counts.get("SL", 0),  # slider
            "st_pct":         pitch_counts.get("ST", 0),  # sweeper
            "ch_pct":         pitch_counts.get("CH", 0),  # changeup
            "fs_pct":         pitch_counts.get("FS", 0),  # splitter
            "cu_pct":         pitch_counts.get("CU", 0),  # curveball
            "kc_pct":         pitch_counts.get("KC", 0),  # knuckle curve
            # velocity per pitch type
            "ff_velo":        ff_velo,
            "si_velo":        si_velo,
            "fc_velo":        fc_velo,
            "sl_velo":        sl_velo,
            "st_velo":        st_velo,
            "ch_velo":        ch_velo,
            "fs_velo":        fs_velo,
            "cu_velo":        cu_velo,
            # deception gap off best fastball: bigger = harder to time off-speed
            "fb_ch_gap":      fb_velo - ch_velo,
            "fb_fs_gap":      fb_velo - fs_velo,
        })
    
    logs = (
        df.groupby("game_pk")
        .apply(agg_game, include_groups=False)
        .reset_index(drop=True)
        .sort_values("game_date")
    )

    print("games aggregated:", len(logs))
    print("game_date dtype:", logs["game_date"].dtype)
    print("sample dates:", logs["game_date"].head())
    print("end_date_str:", end_date_str)

    # normalize game_date to datetime for safe comparison
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs[logs["game_date"] < pd.Timestamp(end_date_str)]
    print("after date filter:", len(logs))
    return logs.tail(n_starts)

# Returns pitchers features as dictionary
def get_pitcher_features(pitcher_id, date, n_starts=5):
    logs = get_pitcher_game_logs(pitcher_id, date, n_starts)

    if logs.empty:
        return {}
    
    numeric_cols = logs.select_dtypes(include="number").columns
    means = logs[numeric_cols].mean()
    return {f"{col}_last{n_starts}": round(val, 4) for col, val in means.items()}


if __name__ == "__main__":
    # Paul Skenes
    logs = get_pitcher_game_logs(694973, pd.Timestamp("2024-09-01"))
    print(logs)
