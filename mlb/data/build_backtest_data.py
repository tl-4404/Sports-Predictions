"""Build leakage-safe historical inputs for the Monte Carlo backtest.

The builder walks games chronologically. For every target start it snapshots
only state accumulated from earlier games, then ingests the completed game.
Use a full prior-season warmup when producing a serious evaluation dataset.

Example:
    python -m mlb.data.build_backtest_data 2025-03-27 2025-09-28 \
      --warmup-start 2024-03-20 --out mlb/cache/backtest_2025.json
    python -m mlb.training.backtest mlb/cache/backtest_2025.json --sims 10000
"""
import argparse
import json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests

from mlb import db
from mlb.data import game_artifacts
from mlb.prediction import factors, simulator
from mlb.providers import mlb_api

BASE = "https://statsapi.mlb.com/api/v1"
LIVE_BASE = "https://statsapi.mlb.com/api/v1.1"
K_EVENTS = {"strikeout", "strikeout_double_play"}
BB_EVENTS = {"walk", "intent_walk", "hit_by_pitch"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
_http_cache = {}


def _get(path, **params):
    key = (path, tuple(sorted(params.items())))
    if key in _http_cache:
        return _http_cache[key]
    response = requests.get(f"{BASE}{path}", params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    _http_cache[key] = data
    return data


def _get_live(path):
    key = ("live", path)
    if key in _http_cache:
        return _http_cache[key]
    response = requests.get(f"{LIVE_BASE}{path}", timeout=30)
    response.raise_for_status()
    data = response.json()
    _http_cache[key] = data
    return data


def _final_game_pks(game_date):
    data = _get("/schedule", sportId=1, date=game_date)
    games = [
        game
        for day in data.get("dates", [])
        for game in day.get("games", [])
        if game.get("status", {}).get("codedGameState") == "F"
        and game.get("gameType") in {"R", "P", "F", "D", "L", "W"}
    ]
    games.sort(key=lambda game: (game.get("gameDate", ""), game["gamePk"]))
    return [game["gamePk"] for game in games]


def _empty_totals():
    return {"k": 0, "bb": 0, "hit": 0, "n": 0}


class PregameRateState:
    """Chronological pitcher/batter rates and starter workload history."""

    def __init__(self):
        self.pitchers = defaultdict(_empty_totals)
        self.batter_splits = defaultdict(_empty_totals)
        self.batters = defaultdict(_empty_totals)
        self.recent_pitch_counts = defaultdict(lambda: deque(maxlen=5))

    def pitcher_profile(self, pitcher_id, season):
        current = self.pitchers[(season, pitcher_id)]
        previous = self.pitchers[(season - 1, pitcher_id)]
        prior = mlb_api._credible_rates(previous, mlb_api.PITCHER_RATE_THRESHOLD)
        rates = mlb_api._threshold_rates(
            current, prior, mlb_api.PITCHER_RATE_THRESHOLD
        )
        return {
            **rates,
            "batters_faced": current["n"],
            "prior_batters_faced": previous["n"],
            "recent_pitch_counts": list(self.recent_pitch_counts[pitcher_id]),
        }

    def batter_rates(self, batter_id, pitcher_hand, season):
        current = self.batter_splits[(season, batter_id, pitcher_hand)]
        previous = self.batter_splits[(season - 1, batter_id, pitcher_hand)]
        if not previous["n"]:
            previous = self.batters[(season - 1, batter_id)]
        if current["n"]:
            prior = mlb_api._credible_rates(
                previous, mlb_api.BATTER_SPLIT_RATE_THRESHOLD
            )
            return mlb_api._threshold_rates(
                current, prior, mlb_api.BATTER_SPLIT_RATE_THRESHOLD
            )
        if previous["n"]:
            return mlb_api._credible_rates(
                previous, mlb_api.BATTER_SPLIT_RATE_THRESHOLD
            )
        current_overall = self.batters[(season, batter_id)]
        previous_overall = self.batters[(season - 1, batter_id)]
        prior = mlb_api._credible_rates(
            previous_overall, mlb_api.BATTER_OVERALL_RATE_THRESHOLD
        )
        return mlb_api._threshold_rates(
            current_overall, prior, mlb_api.BATTER_OVERALL_RATE_THRESHOLD
        )

    def ingest_game(self, plays, box, season):
        for play in plays.get("allPlays", []):
            matchup = play.get("matchup", {})
            pitcher_id = (matchup.get("pitcher") or {}).get("id")
            batter_id = (matchup.get("batter") or {}).get("id")
            if not pitcher_id or not batter_id:
                continue
            hand = (matchup.get("pitchHand") or {}).get("code", "R")
            event = (play.get("result") or {}).get("eventType", "")
            for totals in (
                self.pitchers[(season, pitcher_id)],
                self.batter_splits[(season, batter_id, hand)],
                self.batters[(season, batter_id)],
            ):
                totals["n"] += 1
                totals["k"] += int(event in K_EVENTS)
                totals["bb"] += int(event in BB_EVENTS)
                totals["hit"] += int(event in HIT_EVENTS)

        for side in ("home", "away"):
            team = box.get("teams", {}).get(side, {})
            pitcher_ids = team.get("pitchers", [])
            if not pitcher_ids:
                continue
            starter_id = pitcher_ids[0]
            player = team.get("players", {}).get(f"ID{starter_id}", {})
            stat = player.get("stats", {}).get("pitching", {})
            pitches = stat.get("numberOfPitches") or stat.get("pitchesThrown")
            if pitches:
                self.recent_pitch_counts[starter_id].append(int(pitches))


def _starter_hand(plays, pitcher_id):
    for play in plays.get("allPlays", []):
        matchup = play.get("matchup", {})
        if (matchup.get("pitcher") or {}).get("id") == pitcher_id:
            return (matchup.get("pitchHand") or {}).get("code", "R")
    return "R"


def _fetch_game_payload(game_pk):
    """Prefer one live-feed request; retain endpoint fallbacks for old games."""
    feed = _get_live(f"/game/{game_pk}/feed/live")
    live = feed.get("liveData", {})
    box = live.get("boxscore")
    plays = live.get("plays")
    if not box:
        box = _get(f"/game/{game_pk}/boxscore")
    if not plays:
        plays = _get(f"/game/{game_pk}/playByPlay")
    return box, plays


def _projected_lines(row, n_sims, seed):
    """Model-derived evaluation thresholds; never inspect the actual result."""
    if n_sims <= 0:
        return []
    ks, outs = simulator.simulate_starter(
        row["pitcher"], row["lineup"],
        recent_pitch_counts=row["recent_pitch_counts"],
        k_park=row["k_park"], k_ump=row["k_ump"],
        k_form=row.get("k_form", 1.0), k_arsenal=row.get("k_arsenal", 1.0),
        defense_context=row.get("defense_context"),
        third_time_penalty=row.get(
            "third_time_penalty", simulator.THIRD_TIME_PENALTY
        ),
        n_sims=n_sims, seed=seed,
    )
    lines = []
    for market, mean in (
        ("pitcher_strikeouts", float(ks.mean())),
        ("pitcher_outs", float(outs.mean())),
    ):
        center = round(mean - 0.5) + 0.5
        for offset in (-2, -1, 0, 1, 2):
            line = center + offset
            if line > 0:
                lines.append({"market": market, "line": line})
    return lines


def build_game(game_pk, game_date, state, artifact_conn,
               emit=True, line_sims=500, cached_context_factors=False,
               payload=None):
    """Snapshot both starters pregame, then ingest the game into history."""
    box, plays = payload or _fetch_game_payload(game_pk)
    season = date.fromisoformat(game_date).year
    home_name = (box.get("teams", {}).get("home", {}).get("team", {})
                 or {}).get("name", "")
    away_name = (box.get("teams", {}).get("away", {}).get("team", {})
                 or {}).get("name", "")
    hp_ump = next((
        (official.get("official") or {}).get("fullName")
        for official in box.get("officials", [])
        if official.get("officialType") == "Home Plate"
    ), None)
    k_park = factors.park_k_factor(home_name) if cached_context_factors else 1.0
    k_ump = factors.ump_k_factor(hp_ump) if cached_context_factors else 1.0

    rows = []
    if emit:
        for side, opponent in (("home", "away"), ("away", "home")):
            team = box.get("teams", {}).get(side, {})
            pitcher_ids = team.get("pitchers", [])
            if not pitcher_ids:
                continue
            starter_id = pitcher_ids[0]
            player = team.get("players", {}).get(f"ID{starter_id}", {})
            stat = player.get("stats", {}).get("pitching", {})
            pitcher_name = (player.get("person") or {}).get("fullName")
            hand = _starter_hand(plays, starter_id)
            profile = state.pitcher_profile(starter_id, season)
            lineup_ids = box.get("teams", {}).get(opponent, {}).get(
                "battingOrder", []
            )[:9]
            lineup = [
                state.batter_rates(batter_id, hand, season)
                for batter_id in lineup_ids
            ]
            while len(lineup) < 9:
                lineup.append(dict(mlb_api.LEAGUE))
            fielding_team = home_name if side == "home" else away_name
            context = db.defense_context(artifact_conn, starter_id, fielding_team)
            k_arsenal = db.arsenal_matchup_factor(
                artifact_conn, starter_id, lineup_ids
            )
            row = {
                "game_pk": game_pk,
                "game_date": game_date,
                "pitcher_id": starter_id,
                "pitcher_name": pitcher_name,
                "pitcher_hand": hand,
                "fielding_team": fielding_team,
                "opponent_team": away_name if side == "home" else home_name,
                "lineup_ids": lineup_ids,
                "pitcher": {key: profile[key] for key in ("k", "bb", "hit")},
                "pitcher_sample": {
                    "current_bf": profile["batters_faced"],
                    "prior_bf": profile["prior_batters_faced"],
                },
                "lineup": lineup,
                "recent_pitch_counts": profile["recent_pitch_counts"],
                "k_park": k_park,
                "k_ump": k_ump,
                "k_form": 1.0,
                "k_arsenal": k_arsenal,
                "defense_context": context,
                "third_time_penalty": simulator.THIRD_TIME_PENALTY,
                "actual_ks": int(stat.get("strikeOuts", 0)),
                "actual_outs": mlb_api.innings_to_outs(
                    stat.get("inningsPitched", "0.0")
                ),
                "line_source": "pregame_model_projection",
            }
            row["lines"] = _projected_lines(
                row, line_sims, seed=game_pk + starter_id
            )
            rows.append(row)

    artifacts = game_artifacts.parse_game_artifacts(
        plays, game_pk, game_date, home_name, away_name
    )
    db.replace_game_artifacts(artifact_conn, *artifacts)
    state.ingest_game(plays, box, season)
    return rows


def build_range(start, end, out_path, warmup_start=None, line_sims=500,
                cached_context_factors=False, skip_errors=False, workers=8):
    target_start = date.fromisoformat(start)
    target_end = date.fromisoformat(end)
    history_start = date.fromisoformat(warmup_start) if warmup_start else date(
        target_start.year - 1, 3, 1
    )
    if history_start > target_start or target_end < target_start:
        raise ValueError("Require warmup_start <= start <= end")

    state = PregameRateState()
    artifact_conn = db.connect(":memory:")
    db.ensure_schema(artifact_conn)
    dataset = []
    failures = []
    day = history_start
    while day <= target_end:
        day_string = day.isoformat()
        game_pks = _final_game_pks(day_string)
        starts_before = len(dataset)
        payloads = {}
        fetch_errors = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(_fetch_game_payload, game_pk): game_pk
                for game_pk in game_pks
            }
            for future in as_completed(futures):
                game_pk = futures[future]
                try:
                    payloads[game_pk] = future.result()
                except Exception as exc:
                    fetch_errors[game_pk] = exc

        for game_pk in game_pks:
            try:
                if game_pk in fetch_errors:
                    raise fetch_errors[game_pk]
                dataset.extend(build_game(
                    game_pk, day_string, state, artifact_conn,
                    emit=day >= target_start, line_sims=line_sims,
                    cached_context_factors=cached_context_factors,
                    payload=payloads[game_pk],
                ))
            except Exception as exc:
                failures.append({"game_pk": game_pk, "date": day_string,
                                 "error": str(exc)})
                if not skip_errors:
                    artifact_conn.close()
                    raise
                print(f"  game {game_pk}: skipped ({exc})")
        phase = "target" if day >= target_start else "warmup"
        print(f"{day_string} [{phase}]: {len(game_pks)} finals, "
              f"{len(dataset) - starts_before} starts emitted")
        day += timedelta(days=1)

    artifact_conn.close()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2))
    metadata = {
        "schema_version": 2,
        "created_for": "mlb_monte_carlo_backtest",
        "model_version": simulator.MODEL_VERSION,
        "target_start": start,
        "target_end": end,
        "warmup_start": history_start.isoformat(),
        "starts": len(dataset),
        "line_sims": line_sims,
        "line_source": "pregame_model_projection",
        "cached_context_factors": cached_context_factors,
        "third_time_penalty": simulator.THIRD_TIME_PENALTY,
        "base_dp_probability": simulator.DP_PROB,
        "max_pitches": simulator.MAX_PITCHES,
        "rate_thresholds": {
            "pitcher_bf": mlb_api.PITCHER_RATE_THRESHOLD,
            "batter_split_pa": mlb_api.BATTER_SPLIT_RATE_THRESHOLD,
            "batter_overall_pa": mlb_api.BATTER_OVERALL_RATE_THRESHOLD,
        },
        "fetch_workers": workers,
        "failures": failures,
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"wrote {len(dataset)} starts -> {output}")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build chronological MLB Monte Carlo backtest data"
    )
    parser.add_argument("start", help="first target game date (YYYY-MM-DD)")
    parser.add_argument("end", help="last target game date (YYYY-MM-DD)")
    parser.add_argument("--warmup-start",
                        help="history ingestion start; defaults to prior Mar 1")
    parser.add_argument("--out", default="mlb/cache/backtest.json")
    parser.add_argument("--line-sims", type=int, default=500,
                        help="independent sims used to choose synthetic lines")
    parser.add_argument("--cached-context-factors", action="store_true",
                        help="use park/ump caches only if built from pre-period data")
    parser.add_argument("--skip-errors", action="store_true",
                        help="record failed games instead of failing the build")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel game-feed downloads per date")
    args = parser.parse_args()
    build_range(
        args.start, args.end, args.out,
        warmup_start=args.warmup_start,
        line_sims=args.line_sims,
        cached_context_factors=args.cached_context_factors,
        skip_errors=args.skip_errors,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
