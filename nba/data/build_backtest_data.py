"""Build pregame NBA moneyline datasets from season-level team game logs.

One LeagueGameFinder request per season, rather than five requests per game day.
Historical reconstruction, not a record of forecasts actually made at the time.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import pandas as pd
from nba.providers import nba_api

CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache"
STATS = ("PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "OREB", "DREB", "REB",
         "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS")
REQUIRED = {"GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION",
            "MATCHUP", "WL", "PTS"}


def validate_season(value):
    if not re.fullmatch(r"\d{4}-\d{2}", value) or int(value[-2:]) != (int(value[:4]) + 1) % 100:
        raise argparse.ArgumentTypeError("Season must look like 2025-26")
    return value


def fetch_game_logs(season, cache_root=CACHE_ROOT, refresh=False):
    """Cache raw responses. Refresh explicitly for an unfinished season."""
    path = Path(cache_root) / "game_logs" / f"{season}_regular.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"GAME_ID": str, "TEAM_ID": str})
    for attempt in range(3):
        try:
            frame = nba_api.fetch_game_logs(season, team_only=True)
            if frame.empty:
                raise ValueError(f"No completed game logs returned for {season}")
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            return frame
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def build_dataset(logs, season, window=10, min_games=5):
    """Emit each day's features before ingesting any results from that day.

    History resets each season. First games have insufficient history and are
    omitted by default; --min-games 0 retains them with missing feature values.
    """
    if window < 1 or min_games < 0:
        raise ValueError("window must be positive and min_games nonnegative")
    missing = REQUIRED - set(logs.columns)
    if missing:
        raise ValueError(f"Missing game log columns: {sorted(missing)}")
    games = logs.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="raise").dt.normalize()
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)
    games["TEAM_ID"] = games["TEAM_ID"].astype(str)
    if games.duplicated(["GAME_ID", "TEAM_ID"]).any():
        raise ValueError("Duplicate team/game rows in raw logs")
    stats = [c for c in STATS if c in games.columns]
    for c in stats:
        games[c] = pd.to_numeric(games[c], errors="raise")
    histories, output = {}, []
    for game_date, day in games.sort_values(["GAME_DATE", "GAME_ID"]).groupby("GAME_DATE"):
        completed = []
        for game_id, pair in day.groupby("GAME_ID"):
            if len(pair) != 2 or set(pair["WL"]) != {"W", "L"} or pair["PTS"].isna().any():
                continue
            home = pair[pair["MATCHUP"].str.contains(" vs. ", regex=False, na=False)]
            away = pair[pair["MATCHUP"].str.contains(" @ ", regex=False, na=False)]
            if len(home) != 1 or len(away) != 1:
                continue
            home, away = home.iloc[0], away.iloc[0]
            if home["TEAM_ID"] == away["TEAM_ID"]:
                raise ValueError(f"Same home and away team for {game_id}")
            if home["PTS"] == away["PTS"] or (home["PTS"] > away["PTS"]) != (home["WL"] == "W"):
                raise ValueError(f"Inconsistent score and result for {game_id}")
            completed.extend([home, away])
            prior_home = histories.get(home["TEAM_ID"], [])
            prior_away = histories.get(away["TEAM_ID"], [])
            if min(len(prior_home), len(prior_away)) < min_games:
                continue
            row = {
                "GAME_ID": game_id, "GAME_DATE": game_date.date().isoformat(),
                "season": season, "season_type": "Regular Season",
                "feature_cutoff": (game_date - pd.Timedelta(days=1)).date().isoformat(),
                "home_team_id": home["TEAM_ID"], "away_team_id": away["TEAM_ID"],
                "home_team": home["TEAM_ABBREVIATION"], "away_team": away["TEAM_ABBREVIATION"],
                "home_win": int(home["WL"] == "W"),
                "target_home_points": home["PTS"], "target_away_points": away["PTS"],
            }
            for side, history in (("home", prior_home), ("away", prior_away)):
                row[f"{side}_prior_games"] = len(history)
                rest = (game_date - history[-1]["GAME_DATE"]).days - 1 if history else float("nan")
                row[f"{side}_rest_days"] = rest
                row[f"{side}_back_to_back"] = int(rest == 0) if history else float("nan")
                row[f"{side}_win_pct"] = sum(g["WL"] == "W" for g in history) / len(history) if history else float("nan")
                for stat in stats:
                    row[f"{side}_season_{stat}"] = pd.Series([g[stat] for g in history], dtype=float).mean()
                    row[f"{side}_last{window}_{stat}"] = pd.Series([g[stat] for g in history[-window:]], dtype=float).mean()
            # Only paired pregame features; identifiers and outcomes never enter diffs.
            for name in list(row):
                if name.startswith("home_") and f"away_{name[5:]}" in row and name not in {"home_team", "home_team_id"}:
                    row[f"diff_{name[5:]}"] = row[name] - row[f"away_{name[5:]}"]
            output.append(row)
        for result in completed:
            histories.setdefault(result["TEAM_ID"], []).append(result)
    return pd.DataFrame(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", required=True, type=validate_season)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-games", type=int, default=5)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--refresh", action="store_true", help="Re-fetch cached season logs")
    parser.add_argument("--raw-csv", type=Path, help="Offline LeagueGameFinder CSV; one season only")
    parser.add_argument("--output", type=Path, default=CACHE_ROOT / "backtest" / "nba_games.csv")
    args = parser.parse_args(argv)
    if args.window < 1 or args.min_games < 0:
        parser.error("--window must be positive and --min-games nonnegative")
    if args.raw_csv and len(args.seasons) != 1:
        parser.error("--raw-csv requires exactly one season")
    if len(set(args.seasons)) != len(args.seasons):
        parser.error("--seasons contains duplicates")
    frames, counts = [], {}
    for season in sorted(args.seasons):
        raw = (pd.read_csv(args.raw_csv, dtype={"GAME_ID": str, "TEAM_ID": str})
               if args.raw_csv else fetch_game_logs(season, args.cache_root, args.refresh))
        frame = build_dataset(raw, season, args.window, args.min_games)
        frames.append(frame)
        counts[season] = {"raw_team_rows": len(raw), "dataset_games": len(frame)}
        print(f"{season}: {len(raw)} team logs -> {len(frame)} eligible games")
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise ValueError("No eligible games. Check source logs and --min-games.")
    result = result.sort_values(["GAME_DATE", "GAME_ID"])
    if result["GAME_ID"].duplicated().any():
        raise ValueError("Duplicate game IDs across seasons")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": "nba_api LeagueGameFinder" if not args.raw_csv else str(args.raw_csv.resolve()),
        "seasons": counts, "window": args.window, "min_games": args.min_games,
        "feature_columns": [c for c in result if c.startswith("diff_")],
        "target": "home_win", "historical_reconstruction": True,
        "includes_odds": False, "includes_historical_injuries": False,
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(result)} games to {args.output}")


if __name__ == "__main__":
    main()
