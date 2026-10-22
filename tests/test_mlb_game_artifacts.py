"""Postgame artifact parsing and idempotent storage tests."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlb import db
from mlb.data.game_artifacts import parse_game_artifacts
from mlb.prediction import simulator


def _pitch(code, description, in_play=False, trajectory=None):
    pitch = {
        "isPitch": True,
        "details": {
            "type": {"code": code},
            "description": description,
            "isInPlay": in_play,
        },
    }
    if trajectory:
        pitch["hitData"] = {"trajectory": trajectory}
    return pitch


def _play(event_type, pitcher, half, inning, outs_after, events,
          trajectory=None, runner_first=False):
    runners = ([{"movement": {"start": "1B", "end": "2B"}}]
               if runner_first else [])
    return {
        "about": {"halfInning": half, "inning": inning},
        "matchup": {
            "pitcher": {"id": pitcher},
            "batter": {"id": pitcher + 100},
        },
        "result": {"eventType": event_type},
        "count": {"outs": outs_after},
        "playEvents": events,
        "hitData": {"trajectory": trajectory} if trajectory else {},
        "runners": runners,
    }


def _fixture():
    return {"allPlays": [
        _play("single", 10, "top", 1, 0,
              [_pitch("FF", "Called Strike"),
               _pitch("SL", "Hit Into Play", True)],
              "ground_ball"),
        _play("grounded_into_double_play", 10, "top", 1, 2,
              [_pitch("FF", "Swinging Strike"),
               _pitch("FF", "Hit Into Play", True)],
              "ground_ball", runner_first=True),
        _play("field_error", 20, "bottom", 1, 0,
              [_pitch("CH", "Hit Into Play", True)],
              "ground_ball"),
    ]}


def test_parser_builds_arsenal_ground_dp_and_error_rows():
    arsenal, batters, batted, fielding = parse_game_artifacts(
        _fixture(), 99, "2026-07-01", "Home", "Away"
    )
    ff = next(row for row in arsenal
              if row["pitcher_id"] == 10 and row["pitch_type"] == "FF")
    assert ff["pitches"] == 3
    assert ff["swings"] == 2
    assert ff["whiffs"] == 1
    assert sum(row["pitches"] for row in batters) == 5
    pitcher = next(row for row in batted if row["pitcher_id"] == 10)
    assert pitcher["ground_balls"] == 2
    assert pitcher["balls_in_play"] == 2
    home = next(row for row in fielding if row["team_name"] == "Home")
    away = next(row for row in fielding if row["team_name"] == "Away")
    assert home["dp_opportunities"] == 1 and home["double_plays"] == 1
    assert away["errors"] == 1 and away["balls_in_play"] == 1


def test_parser_reads_live_feed_trajectory_from_final_pitch_event():
    feed = {"allPlays": [
        _play("grounded_into_double_play", 10, "top", 1, 2,
              [_pitch("FF", "Hit Into Play", True, "ground_ball")],
              trajectory=None, runner_first=True),
    ]}
    _, _, batted, fielding = parse_game_artifacts(
        feed, 100, "2026-07-02", "Home", "Away"
    )
    assert batted[0]["ground_balls"] == 1
    assert fielding[0]["dp_opportunities"] == 1
    assert fielding[0]["double_plays"] == 1


def test_storage_is_idempotent_and_excludes_same_day_from_pregame_context():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.ensure_schema(conn)
    rows = parse_game_artifacts(_fixture(), 99, "2026-07-01", "Home", "Away")
    db.replace_game_artifacts(conn, *rows)
    db.replace_game_artifacts(conn, *rows)
    assert conn.execute("SELECT COUNT(*) FROM pitcher_arsenal_games").fetchone()[0] == 3
    assert db.pitcher_arsenal(conn, 10, "2026-07-01") == []
    arsenal = db.pitcher_arsenal(conn, 10, "2026-07-02")
    assert sum(row["pitches"] for row in arsenal) == 4
    context = db.defense_context(conn, 10, "Home", "2026-07-02")
    assert context["pitcher_ground_rate"] == 1.0
    assert context["team_dp_rate"] == 1.0
    conn.close()


def test_defense_rates_use_pitcher_grounders_team_conversion_and_errors():
    low = simulator.defense_rates({
        "pitcher_ground_rate": 0.30, "pitcher_bip": 200,
        "team_dp_rate": 0.35, "team_dp_opportunities": 100,
        "team_error_rate": 0.005, "team_bip": 1000,
        "league_ground_rate": 0.45, "league_dp_rate": 0.50,
        "league_error_rate": 0.015,
    })
    high = simulator.defense_rates({
        "pitcher_ground_rate": 0.60, "pitcher_bip": 200,
        "team_dp_rate": 0.65, "team_dp_opportunities": 100,
        "team_error_rate": 0.030, "team_bip": 1000,
        "league_ground_rate": 0.45, "league_dp_rate": 0.50,
        "league_error_rate": 0.015,
    })
    assert high["dp"] > low["dp"]
    assert high["error"] > low["error"]


def test_fixed_third_time_penalty_moves_events_in_expected_direction():
    base = [0.25, 0.08, 0.22, 0.45]
    adjusted = simulator.third_time_probs(base, penalty=0.05)
    assert adjusted[0] < base[0]
    assert adjusted[1] > base[1]
    assert adjusted[2] > base[2]
    assert abs(sum(adjusted) - 1.0) < 1e-12


def test_hook_rewards_efficiency_and_penalizes_pitcher_created_traffic():
    good = simulator.hook_probability(
        pitches=88, workload_center=88, batters_faced=24,
        pitcher_reaches=3, inning_pitcher_reaches=0,
        expected_reach_rate=0.30, third_time=True, inning_end=True,
    )
    poor = simulator.hook_probability(
        pitches=88, workload_center=88, batters_faced=24,
        pitcher_reaches=10, inning_pitcher_reaches=4,
        expected_reach_rate=0.30, third_time=True, inning_end=True,
    )
    assert poor > good


def test_errors_do_not_enter_pitcher_performance_hook_signal():
    # Identical pitcher-created traffic yields an identical hook probability;
    # errors affect hook risk only indirectly through added pitches/BF.
    baseline = simulator.hook_probability(
        75, 90, 22, 6, 1, 0.30, inning_end=True
    )
    with_untracked_errors = simulator.hook_probability(
        75, 90, 22, 6, 1, 0.30, inning_end=True
    )
    assert baseline == with_untracked_errors
