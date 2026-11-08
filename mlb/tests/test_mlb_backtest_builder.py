"""Leakage and chronology checks for the real MLB backtest builder."""
import copy
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlb import db
from mlb.data_fetching import build_backtest_data as builder
from mlb.data_fetching.providers import mlb_api


def _pitch(code="FF", description="Swinging Strike", in_play=False):
    return {
        "isPitch": True,
        "details": {
            "type": {"code": code}, "code": "S",
            "description": description, "isInPlay": in_play,
        },
    }


def _play(pitcher, batter, half, event, outs):
    return {
        "about": {"inning": 1, "halfInning": half},
        "matchup": {
            "pitcher": {"id": pitcher}, "batter": {"id": batter},
            "pitchHand": {"code": "R"},
        },
        "result": {"eventType": event},
        "count": {"outs": outs},
        "playEvents": [_pitch()],
        "runners": [],
    }


def _box():
    def team(name, starter, batter, strikeouts, pitches):
        return {
            "team": {"name": name},
            "pitchers": [starter],
            "battingOrder": [batter],
            "players": {
                f"ID{starter}": {
                    "person": {"fullName": f"Pitcher {starter}"},
                    "stats": {"pitching": {
                        "strikeOuts": strikeouts,
                        "inningsPitched": "1.0",
                        "numberOfPitches": pitches,
                    }},
                }
            },
        }
    return {
        "officials": [],
        "teams": {
            "home": team("Home", 10, 101, 1, 15),
            "away": team("Away", 20, 201, 0, 18),
        },
    }


def _plays():
    return {"allPlays": [
        _play(10, 201, "top", "strikeout", 1),
        _play(20, 101, "bottom", "walk", 0),
    ]}


def test_build_game_snapshots_before_ingesting_current_game():
    state = builder.PregameRateState()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.ensure_schema(conn)
    original_fetch = builder._fetch_game_payload
    builder._fetch_game_payload = lambda game_pk: (_box(), _plays())
    try:
        rows = builder.build_game(
            99, "2025-04-01", state, conn, emit=True, line_sims=0
        )
    finally:
        builder._fetch_game_payload = original_fetch
        conn.close()

    assert len(rows) == 2
    assert all(row["pitcher_sample"]["current_bf"] == 0 for row in rows)
    assert rows[0]["pitcher"] == mlb_api.LEAGUE
    assert state.pitcher_profile(10, 2025)["batters_faced"] == 1
    assert list(state.recent_pitch_counts[10]) == [15]


def test_model_derived_lines_do_not_depend_on_actual_result():
    row = {
        "pitcher": dict(mlb_api.LEAGUE),
        "lineup": [dict(mlb_api.LEAGUE) for _ in range(9)],
        "recent_pitch_counts": [88, 92, 90],
        "k_park": 1.0, "k_ump": 1.0, "k_form": 1.0,
        "k_arsenal": 1.0, "defense_context": None,
        "third_time_penalty": 0.05,
        "actual_ks": 0, "actual_outs": 3,
    }
    changed_actual = copy.deepcopy(row)
    changed_actual["actual_ks"] = 14
    changed_actual["actual_outs"] = 27
    assert builder._projected_lines(row, 300, seed=7) == builder._projected_lines(
        changed_actual, 300, seed=7
    )


def test_previous_season_becomes_small_sample_prior():
    state = builder.PregameRateState()
    state.ingest_game(_plays(), _box(), 2024)
    profile = state.pitcher_profile(10, 2025)
    assert profile["batters_faced"] == 0
    assert profile["prior_batters_faced"] == 1
    assert profile["k"] > mlb_api.LEAGUE["k"]
