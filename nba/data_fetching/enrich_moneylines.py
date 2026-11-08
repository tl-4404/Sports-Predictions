"""Enrich chronological moneyline history with official NBA per-game measures.

Run: python -m nba.data_fetching.enrich_moneylines --seasons 2021-22 2022-23 2023-24 2024-25 2025-26
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from nba.data_fetching.build_backtest_data import (
    CACHE_ROOT, STATS, audit_dataset, build_dataset, fetch_game_logs, validate_season,
)
from nba.data_fetching.providers import nba_api


# Explicit allowlists keep ranks, identifiers, and current-game targets out of X.
# Overlapping measures use the Advanced table once rather than duplicate columns.
MEASURES = {
    "Base": ("PFD",),
    "Advanced": (
        "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO", "AST_RATIO",
        "OREB_PCT", "DREB_PCT", "REB_PCT", "TM_TOV_PCT", "EFG_PCT", "TS_PCT",
        "E_PACE", "PACE", "POSS", "PIE",
    ),
    "Four Factors": (
        "FTA_RATE", "OPP_EFG_PCT", "OPP_FTA_RATE", "OPP_TOV_PCT", "OPP_OREB_PCT",
    ),
    "Misc": (
        "PTS_OFF_TOV", "PTS_2ND_CHANCE", "PTS_FB", "PTS_PAINT",
        "OPP_PTS_OFF_TOV", "OPP_PTS_2ND_CHANCE", "OPP_PTS_FB", "OPP_PTS_PAINT",
    ),
    "Opponent": (
        "OPP_FG_PCT", "OPP_FG3_PCT", "OPP_FT_PCT", "OPP_OREB", "OPP_DREB",
        "OPP_AST", "OPP_TOV", "OPP_STL", "OPP_BLK", "OPP_BLKA", "OPP_PF",
        "OPP_PFD", "OPP_PTS",
    ),
}
ALL_STATS = tuple(dict.fromkeys(STATS + tuple(
    stat for fields in MEASURES.values() for stat in fields
)))
DEFAULT_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")


def normalize_keys(frame):
    frame = frame.copy()
    if not {"GAME_ID", "TEAM_ID", "GAME_DATE"}.issubset(frame):
        raise ValueError("Measure logs require GAME_ID, TEAM_ID, GAME_DATE")
    frame["GAME_ID"] = frame["GAME_ID"].astype(str).str.zfill(10)
    frame["TEAM_ID"] = frame["TEAM_ID"].astype(str)
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"], errors="raise").dt.normalize()
    if frame.duplicated(["GAME_ID", "TEAM_ID"]).any():
        raise ValueError("Duplicate measure rows for a team/game")
    return frame


def fetch_measure_logs(season, measure, cache_root=CACHE_ROOT, refresh=False):
    path = Path(cache_root) / "game_logs" / "measures" / (
        f"{season}_{measure.lower().replace(' ', '_')}.csv"
    )
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"GAME_ID": str, "TEAM_ID": str})
    for attempt in range(3):
        try:
            frame = nba_api.fetch_team_game_measures(season, measure)
            if frame.empty:
                raise ValueError(f"Empty {measure} response for {season}")
            # Validate schema and identity before accepting a new raw cache.
            normalize_keys(frame)
            missing = set(MEASURES[measure]) - set(frame)
            if missing:
                raise ValueError(f"{measure} missing fields: {sorted(missing)}")
            if "SEASON_YEAR" in frame and not frame["SEASON_YEAR"].astype(str).eq(season).all():
                raise ValueError(f"{measure} response does not match season {season}")
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            return frame
        except Exception as exc:
            if attempt == 2:
                raise
            print(f"{season} {measure}: retry {attempt + 2}/3 ({exc})", flush=True)
            time.sleep(2 ** attempt)


def merge_measures(base, measures):
    """Require complete one-to-one coverage and matching dates for every source."""
    merged = normalize_keys(base)
    baseline_keys = pd.MultiIndex.from_frame(merged[["GAME_ID", "TEAM_ID"]])
    for measure, fields in MEASURES.items():
        source = normalize_keys(measures[measure])
        source_keys = pd.MultiIndex.from_frame(source[["GAME_ID", "TEAM_ID"]])
        missing_keys = baseline_keys.difference(source_keys)
        extra_keys = source_keys.difference(baseline_keys)
        if len(missing_keys) or len(extra_keys):
            raise ValueError(
                f"{measure} coverage mismatch: {len(missing_keys)} missing, "
                f"{len(extra_keys)} extra team/game rows"
            )
        missing_fields = set(fields) - set(source)
        if missing_fields:
            raise ValueError(f"{measure} missing fields: {sorted(missing_fields)}")
        source = source[["GAME_ID", "TEAM_ID", "GAME_DATE", *fields]].rename(
            columns={"GAME_DATE": "source_game_date"}
        )
        merged = merged.merge(source, on=["GAME_ID", "TEAM_ID"], validate="one_to_one")
        if not merged["GAME_DATE"].eq(merged["source_game_date"]).all():
            raise ValueError(f"{measure} game dates do not match base logs")
        merged = merged.drop(columns="source_game_date")
    for field in ALL_STATS:
        merged[field] = pd.to_numeric(merged[field], errors="raise")
    # A game percentage is undefined when no attempts occurred (e.g. Boston
    # attempted zero free throws on 2024-04-09). Preserve that missing rate;
    # the historical mean excludes it rather than inventing a 0% shooting game.
    for field in ALL_STATS:
        structural_missing = pd.Series(False, index=merged.index)
        denominator = {"FG_PCT": "FGA", "FG3_PCT": "FG3A", "FT_PCT": "FTA"}.get(field)
        if denominator in merged:
            structural_missing = merged[field].isna() & merged[denominator].eq(0)
        invalid = ~np.isfinite(merged[field]) & ~structural_missing
        if invalid.any():
            raise ValueError(f"Raw enrichment contains missing or infinite {field}")
    return merged


def legacy_coverage(features):
    """Map the saved NN feature names to the new season-window equivalents."""
    from nba.paths import SAVED_MODELS_DIR
    path = SAVED_MODELS_DIR / "feature_columns.csv"
    if not path.exists():
        return {"available": False}
    names = [name.strip() for name in path.read_text().splitlines() if name.strip()]
    mapping = {
        name: ("diff_win_pct" if name == "diff_W_PCT"
               else f"diff_season_{name.removeprefix('diff_')}")
        for name in names
    }
    return {
        "available": True, "original_count": len(names),
        "mapping": mapping,
        "missing": [name for name, target in mapping.items() if target not in features],
        "note": "Equivalent stat concepts; per-game averages differ from legacy pooled snapshots.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS, type=validate_season)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=CACHE_ROOT / "backtest" / "nba_games_enriched.csv")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-games", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.window < 1 or args.min_games < 0 or not 1 <= args.workers <= 5:
        parser.error("Require window > 0, min-games >= 0, and workers between 1 and 5")
    if len(set(args.seasons)) != len(args.seasons):
        parser.error("Duplicate seasons")
    frames, seasons = [], {}
    for season in sorted(args.seasons):
        base = fetch_game_logs(season, args.cache_root, args.refresh)
        measures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(fetch_measure_logs, season, measure, args.cache_root,
                                args.refresh): measure for measure in MEASURES}
            for job in as_completed(jobs):
                measure = jobs[job]
                measures[measure] = job.result()
                print(f"{season} {measure}: {len(measures[measure])} raw rows", flush=True)
        merged = merge_measures(base, measures)
        frame = build_dataset(merged, season, args.window, args.min_games, ALL_STATS)
        frames.append(frame)
        seasons[season] = {
            "raw_team_rows": len(base), "games": len(frame),
            "exclusion_reasons": frame.attrs["exclusions"],
            "audit": audit_dataset(frame),
            "source_sha256": {
                name: hashlib.sha256(value.to_csv(index=False).encode()).hexdigest()
                for name, value in {"GameFinder": base, **measures}.items()
            },
        }
        print(f"{season}: {len(frame)} enriched pregame games", flush=True)
    result = pd.concat(frames, ignore_index=True).sort_values(["GAME_DATE", "GAME_ID"])
    audit = audit_dataset(result)
    features = [column for column in result if column.startswith("diff_")]
    if args.min_games > 0 and not np.isfinite(result[features].to_numpy(dtype=float)).all():
        raise ValueError("Enriched pregame features contain missing or infinite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    metadata = {
        "schema_version": "nba_moneyline_enriched_v2", "market": "full_game_moneyline",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": "NBA Stats LeagueGameFinder + TeamGameLogs",
        "seasons": seasons, "window": args.window, "min_games": args.min_games,
        "target": "home_win", "feature_columns": features, "statistics": list(ALL_STATS),
        "source_groups": {name: list(fields) for name, fields in MEASURES.items()},
        "aggregation": "Arithmetic mean of earlier per-game values in each window; not pooled season rates",
        "historical_reconstruction": True, "includes_odds": False,
        "includes_historical_injuries": False, "audit": audit,
        "legacy_nn_coverage": legacy_coverage(features),
        "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(result)} games, {len(features)} features -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
