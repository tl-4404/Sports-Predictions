import time
import os
from requests.exceptions import ReadTimeout
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder, LeagueDashTeamStats
# sleep time between API calls
SLEEP = 0.7
# api parameters
SEASONS = ["2023-24"]   
SEASON_TYPE = "Regular Season"
PER_MODE = "PerGame"
MEASURE_TYPES = [
    "Base",
    "Advanced",
    "Misc",
    "Four Factors",
    "Opponent"
]
# cache paths
CACHE_ROOT = "cache"
SNAPSHOT_DIR = os.path.join(CACHE_ROOT, "team_snapshots")
PROGRESS_FILE = os.path.join(CACHE_ROOT, "last_date.txt")
# create cache
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

# loads last date of cache
def load_last_date():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return pd.to_datetime(f.read().strip())
    return None

# save last date cached
def save_last_date(d):
    with open(PROGRESS_FILE, "w") as f:
        f.write(pd.to_datetime(d).strftime("%Y-%m-%d"))

# api call to get team stats
def call_leaguedash_stats(season, date_api, measure_type, season_type = SEASON_TYPE, per_mode = PER_MODE):
    try:
        stats = LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense=measure_type,   
            per_mode_detailed=per_mode,
            date_to_nullable=date_api,
            timeout=5
        )
        return stats.get_data_frames()[0]
    # if api timeout, raise error
    except ReadTimeout:
        print(f"ReadTimeout on {measure_type} {date_api}")
        return None 

# get all teams' stats on specific date
def fetch_team_stats(season, date_api):
    # cache path
    cache_path = os.path.join(SNAPSHOT_DIR, f"{season}_{date_api.replace('/','-')}.csv")
    # check to see if exists
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    print(f"fetching team stats on {date_api}!")
    dfs = {}
    # get all measure types' stats
    for mt in MEASURE_TYPES:
        df_mt = call_leaguedash_stats(season, date_api, mt)
        if df_mt is None or df_mt.empty:
            print(f"[WARN] Failed/empty {mt} on {date_api}")
            return None
        dfs[mt] = df_mt
        time.sleep(SLEEP)
    # concat all measure type's for each team into one pd
    base = dfs["Base"].copy()
    for mt in MEASURE_TYPES:
        if mt == "Base":
            continue

        suffix = "_" + mt.lower().replace(" ", "")
        df_other = dfs[mt].copy().drop(columns=["TEAM_NAME", "TEAM_ABBREVIATION"], errors="ignore")
        base = base.merge(df_other, on="TEAM_ID", how="left", suffixes=("", suffix))

    base.to_csv(cache_path, index=False) 
    return base

# build an entire season's team data
def build_training_data_for_season(season, season_type = SEASON_TYPE):
    # get all games for the season
    gamefinder = LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type
    )
    # get games df
    games = gamefinder.get_data_frames()[0]
    # converts and sorts by date
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE")

    rows = []
    # get last cached date
    last_done = load_last_date()
    # get data for all games by day
    for game_date, df_day in games.groupby("GAME_DATE"):
        # check for if date is already processed
        if last_done is not None and game_date <= last_done:
            continue
        # lag date to avoid data leaking to model
        lagged_date = (game_date - pd.Timedelta(days=1))
        date_api = lagged_date.strftime("%m/%d/%Y")
        # fetch team stats for lagged date
        try:
            team_snapshot = fetch_team_stats(season, date_api)
        except Exception as e:
            print(f"[ERROR] Failed to build snapshot for {date_api}: {e}. Skipping {date_api}.")
            continue
        # check for stats (if api returned stats)
        if team_snapshot is None or team_snapshot.empty:
            print(f"[WARN] Empty snapshot for {date_api}, skipping this date.")
            continue
        # set index for stats as "TEAM_ID"
        team_stats_by_id = team_snapshot.set_index("TEAM_ID")
        # get each individual game per day
        for game_id, df_game in df_day.groupby("GAME_ID"):
            # check for exactly 2 teams per game
            if df_game.shape[0] != 2:
                continue
            # mask to seperate home/away team
            home_mask = df_game["MATCHUP"].str.contains(r" vs\. ", case=False, regex=True)
            away_mask = df_game["MATCHUP"].str.contains(r" @ ", case=False, regex=True)
            # check to have one home and one away team
            if not home_mask.any() or not away_mask.any():
                continue
            # seperate home/away team
            home_row = df_game[home_mask].iloc[0]
            away_row = df_game[away_mask].iloc[0]
            home_team_id = int(home_row["TEAM_ID"])
            away_team_id = int(away_row["TEAM_ID"])
            home_win = 1 if home_row["WL"] == "W" else 0
            # check to see if stats exist for team
            if (home_team_id not in team_stats_by_id.index) or (away_team_id not in team_stats_by_id.index):
                print(f"[WARN] Missing team stats for GAME_ID {game_id} on {date_api}, skipping game.")
                continue
            # get team stats
            home_feats = team_stats_by_id.loc[[home_team_id]].add_prefix("home_")
            away_feats = team_stats_by_id.loc[[away_team_id]].add_prefix("away_")
            # concat home and away stats to a one row pd df
            row_df = pd.concat([
                    home_feats.reset_index(drop=True),
                    away_feats.reset_index(drop=True),
                ],
                axis=1,
            )
            # set game indetifiers
            row_df["GAME_ID"] = game_id
            row_df["GAME_DATE"] = game_date
            row_df["home_team"] = home_row["TEAM_ABBREVIATION"]
            row_df["away_team"] = away_row["TEAM_ABBREVIATION"]
            row_df["home_win"] = home_win

            rows.append(row_df)
            # update cached date
            save_last_date(game_date)
            time.sleep(0.5) 
    # check for no games added
    if not rows:
        raise ValueError(f"No training rows built for season {season} (check API or params).")
    # concat all game rows to one df & cleans up columns
    training_df = pd.concat(rows, ignore_index=True)
    training_df.columns = training_df.columns.str.strip()

    print(f"[INFO] Built training set for {season} with {len(training_df)} games.")
    return training_df
        
def main():
    all_seasons = []

    for season in SEASONS:
        print(f"Building training data for season {season}...")
        # build season
        df_season = build_training_data_for_season(season)
        all_seasons.append(df_season)

        out_file = f"nba_training_{season.replace('-', '')}.csv"
        df_season.to_csv(out_file, index=False)
        print(f"Saved {out_file} with {len(df_season)} rows.")
    # check for more than 1 season
    if len(all_seasons) > 1:
        full_df = pd.concat(all_seasons, ignore_index=True)
        full_df.to_csv("nba_training_all_seasons_full_simple_loop.csv", index=False)
        print(f"Saved nba_training_all_seasons_full_simple_loop.csv with {len(full_df)} rows.")


if __name__ == "__main__":
    main()

