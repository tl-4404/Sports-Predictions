import os
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "cache", "training_data.csv")

# only keep columns actually used by feature functions — drops ~90% of Statcast columns
KEEP_COLS = [
    "game_pk", "game_date", "pitcher", "batter",
    "home_team", "inning_topbot", "at_bat_number", "events", "description",
    "pitch_type", "release_speed", "zone", "p_throws",
]

# outs recorded per plate-appearance event (multi-out events counted fully;
# caught stealing / pickoffs aren't in `events` so outs are still ~1% low,
# which is why serving-time grading uses official IP from the box score)
EVENT_OUTS = {
    "strikeout": 1, "field_out": 1, "force_out": 1, "sac_fly": 1,
    "sac_bunt": 1, "fielders_choice_out": 1, "other_out": 1,
    "grounded_into_double_play": 2, "strikeout_double_play": 2,
    "sac_fly_double_play": 2, "double_play": 2, "triple_play": 3,
}

# Find BOTH starters: first pitcher in each offensive half. Outs reconstructed
# from events are approximate; official box-score labels remain preferable.
def get_starter_stats(game_df):
    rows = []
    for half in ("Top", "Bot"):
        half_df = game_df[game_df["inning_topbot"] == half]
        if half_df.empty:
            continue
        starter_id = half_df.sort_values("at_bat_number").iloc[0]["pitcher"]
        starter_df = half_df[half_df["pitcher"] == starter_id]
        actual_ks = starter_df["events"].isin(("strikeout", "strikeout_double_play")).sum()
        actual_outs = starter_df["events"].map(EVENT_OUTS).fillna(0).sum()
        rows.append((starter_id, int(actual_ks), int(actual_outs)))
    return rows

# reconstruct the opposing lineup from batters who faced the starter
# batters are ordered by first appearance (approximates lineup order)
# returns list of tuples (lineup_position, batter_id)
def get_lineup_from_game(game_df, starter_id):
    starter_df = game_df[game_df["pitcher"] == starter_id].sort_values("at_bat_number")
    batters_in_order = (
        starter_df
        .drop_duplicates(subset="batter", keep="first")["batter"]
        .tolist()
    )
    # cap at 9 — occasionally a 10th batter appears due to extra innings / substitutions
    return [(i + 1, int(batter_id)) for i, batter_id in enumerate(batters_in_order[:9])]

# build one training row per starting pitcher appearance across the given seasons
# processes one season at a time to avoid OOM — keeps only current + lookback season in RAM
# saves result to cache/training_data.csv and returns the DataFrame
def build_training_data(seasons):
    import pybaseball
    from pybaseball import statcast
    from data_pitcher import get_pitcher_features
    from data_lineup import get_lineup_features
    pybaseball.cache.enable()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # wipe existing file so re-runs don't append duplicate seasons
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    for season in seasons:
        lookback = season - 1
        print(f"\n=== Season {season} (lookback: {lookback}) ===")

        # load lookback + current season — drop everything else after processing
        print(f"  fetching {lookback}...")
        lb_df = statcast(f"{lookback}-03-20", f"{lookback}-10-01")[KEEP_COLS]
        print(f"  fetching {season}...")
        cur_df = statcast(f"{season}-03-20", f"{season}-10-01")[KEEP_COLS]

        if cur_df.empty:
            print(f"  no data for {season}, skipping.")
            continue

        # feature lookback window = lookback season + current season to date
        lookup_df = pd.concat([lb_df, cur_df], ignore_index=True)
        lookup_df["game_date"] = pd.to_datetime(lookup_df["game_date"])
        cur_df["game_date"]    = pd.to_datetime(cur_df["game_date"])
        del lb_df  # free memory

        # pre-group by ID once — O(1) lookups per game instead of O(n) scans
        pitcher_lookup = {pid: grp for pid, grp in lookup_df.groupby("pitcher")}
        batter_lookup  = {bid: grp for bid, grp in lookup_df.groupby("batter")}

        games = cur_df.groupby("game_pk")
        print(f"  {len(games)} games to process...")

        rows = []
        for game_pk, game_df in games:
            game_date = game_df["game_date"].iloc[0]
            home_team = game_df["home_team"].iloc[0]

            for starter_id, actual_ks, actual_outs in get_starter_stats(game_df):
                lineup = get_lineup_from_game(game_df, starter_id)
                pitcher_feats = get_pitcher_features(
                    starter_id, game_date,
                    df=pitcher_lookup.get(starter_id, pd.DataFrame()))
                if not pitcher_feats:
                    continue
                p_throws = "L" if pitcher_feats["p_throws"] == 0 else "R"
                lineup_feats = get_lineup_features(lineup, game_date, p_throws=p_throws,
                                                   batter_lookup=batter_lookup)
                # Dated park/weather features require historical snapshots.
                # Omit today's park cache and silently reconstructed weather.
                rows.append({"game_pk": game_pk, "game_date": game_date,
                             "pitcher_id": starter_id, "home_team": home_team,
                             "actual_ks": actual_ks, "actual_outs": actual_outs,
                             **pitcher_feats, **lineup_feats})

        del lookup_df, cur_df  # free memory before next season

        print(f"  {len(rows)} rows from {season}")

        # append this season to CSV and clear from memory
        season_df = pd.DataFrame(rows)
        write_header = not os.path.exists(OUTPUT_CSV)
        season_df.to_csv(OUTPUT_CSV, mode="a", index=False, header=write_header)
        print(f"  checkpoint saved → {OUTPUT_CSV}")

    training_df = pd.read_csv(OUTPUT_CSV)
    print(f"\nSaved {len(training_df)} rows → {OUTPUT_CSV}")
    return training_df


if __name__ == "__main__":
    df = build_training_data((2022, 2023, 2024))
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df[["game_date", "pitcher_id", "actual_ks", "actual_outs"]].head(10))

