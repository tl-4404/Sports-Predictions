import os
from datetime import date
import time
from requests.exceptions import ReadTimeout
import pandas as pd
from nba_api.stats.endpoints import ScoreboardV2, LeagueDashTeamStats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# sleep time between API calls
SLEEP = 0.3
# api parameters
SEASON = ["2025-26"]   
SEASON_TYPE = "Regular Season"
PER_MODE = "PerGame"
MEASURE_TYPES = [
    "Base",
    "Advanced",
    "Misc",
    "Four Factors",
    "Opponent"
]

# get all games for day, d
def get_games_for_date(d):
    # if no specific date passed, use roday
    if d is None:
        d = date.today()
    # set date parameter to correct format
    if isinstance(d, str):
        date_api = d
    else:
        date_api = d.strftime("%m/%d/%Y")
    season = SEASON
    # get games
    sb = ScoreboardV2(game_date=date_api, day_offset=0, league_id="00")
    # convert json to pd dataframs & get the game header table
    header = sb.get_data_frames()[0]
    # relevant cols
    cols = [
        "GAME_ID",
        "HOME_TEAM_ID",
        "VISITOR_TEAM_ID",
        "HOME_TEAM_ABBREVIATION",
        "VISITOR_TEAM_ABBREVIATION",
        "GAME_STATUS_TEXT",
    ]
    # only keep relevant cols from game header table
    cols = [c for c in cols if c in header.columns]
    games = header[cols].copy()

    return games, date_api, season

# api call to get team stats
def call_leaguedash_stats(season, date_api, measure_type, season_type = SEASON_TYPE, per_mode = PER_MODE):
    # api call & return a pd data frame of stats
    try:
        stats = LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense=measure_type,   
            per_mode_detailed=per_mode,
            date_to_nullable=date_api,
            timeout=10
        )
        return stats.get_data_frames()[0]
    # if api timeout, raise error
    except ReadTimeout:
        print(f"ReadTimeout on {measure_type} {date_api}")
        return None 

# get all teams' stats on specific date
def fetch_team_stats(season, date_api):
    print(f"fetching team stats on {date_api}!")
    dfs = {}
    # get all measure types' stats
    for mt in MEASURE_TYPES:
        df_mt = call_leaguedash_stats(season, date_api, mt)
        dfs[mt] = df_mt
        time.sleep(SLEEP)
    # if any measure type timed out, abort
    if any(v is None for v in dfs.values()):
        print("Skipping snapshot — one or more measure types failed to fetch.")
        return None

    # concat all measure type's for each team into one pd
    base = dfs["Base"].copy()
    for mt in MEASURE_TYPES:
        if mt == "Base":
            continue

        suffix = "_" + mt.lower().replace(" ", "")
        df_other = dfs[mt].copy()

        drop_cols = ["TEAM_NAME", "TEAM_ABBREVIATION"]
        df_other = df_other.drop(columns=drop_cols, errors="ignore")

        base = base.merge(
            df_other,
            on="TEAM_ID",
            how="left",
            suffixes=("", suffix),
        )

    return base

# builds each game played on date's, d, data
def build_feature_rows_for_date(d):
    # get each game for date
    games, date_api, season = get_games_for_date(d)
    # check for day with no games
    if games.empty:
        print(f"No games found for {date_api}.")
        return pd.DataFrame()

    print(f"Fetching team stats on {date_api} for season {season}...")
    # get a snapshot of all teams stats
    snapshot = fetch_team_stats(season, date_api)
    if snapshot is None:
        return pd.DataFrame()
    # set the index of snapshot to each teams' id, TEAM_ID
    snapshot_by_team = snapshot.set_index("TEAM_ID")

    rows = []

    for _, g in games.iterrows():
        home_id = g["HOME_TEAM_ID"]
        away_id = g["VISITOR_TEAM_ID"]
        # check for if team stats available for the day
        if home_id not in snapshot_by_team.index or away_id not in snapshot_by_team.index:
            print(f"[WARN] Missing team stats for GAME_ID {g['GAME_ID']}. Skipping.")
            continue
        # fetch home & away stats, seperate by adding suffix to cols
        # reset index, index is no longer team_id
        home_feats = snapshot_by_team.loc[[home_id]].add_prefix("home_").reset_index(drop=True)
        away_feats = snapshot_by_team.loc[[away_id]].add_prefix("away_").reset_index(drop=True)
        # concat home and away to one pd dataframe
        row = pd.concat([home_feats, away_feats], axis=1)
        # add col GAME_ID
        row["GAME_ID"] = g["GAME_ID"]
        # check for team abr & game status
        if "HOME_TEAM_ABBREVIATION" in g.index:
            row["home_team"] = g["HOME_TEAM_ABBREVIATION"]
        if "VISITOR_TEAM_ABBREVIATION" in g.index:
            row["away_team"] = g["VISITOR_TEAM_ABBREVIATION"]
        if "GAME_STATUS_TEXT" in g.index:
            row["GAME_STATUS_TEXT"] = g["GAME_STATUS_TEXT"]

        rows.append(row)
    # check for empty games
    if not rows:
        print("No feature rows built.")
        return pd.DataFrame()
    # return all games
    return pd.concat(rows, ignore_index=True)

def main():
    # get all games today
    df_today = build_feature_rows_for_date(date.today())

    if df_today.empty:
        return

    print("Today's feature rows:")
    print(df_today.head())
    # create file with current games features
    out_file = os.path.join(BASE_DIR, "current_games_features.csv")
    df_today.to_csv(out_file, index=False)
    print(f"Saved {len(df_today)} games to {out_file}")


if __name__ == "__main__":
    main()