"""Behavioral regressions for the October audit, with no external requests."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mlb"))
import backtest
import betting_math
import build_backtest_data
import build_training
import calibration
import data_today
import db
import grade
import integrity
import predict
import prizepicks
import simulator
from training_validation import chronological_features


def prediction(**overrides):
    now = datetime.now(timezone.utc)
    row = dict(run_id="r1", created_at=now.isoformat(), game_date=now.date().isoformat(),
               game_pk=1, player_id=10, player_name="Same Pitcher", market="pitcher_strikeouts",
               line=5.5, source="judge", p_over=0.63, raw_p_over=0.61, p_push=0.0,
               mean=6.2, book="Book", over_odds=-110, under_odds=-110,
               market_p_over=0.5, edge_over=0.13, edge_under=-0.13,
               bet_side="over", kelly_frac=0.02, rationale=None,
               model_version=integrity.MODEL_VERSION, model_status="active",
               game_start_time=(now + timedelta(hours=2)).isoformat(),
               line_as_of=now.isoformat(), lineup_confirmed=1, role_verified=1,
               price_verified=1)
    row.update(overrides)
    return row


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "audit.db")
    connection = db.connect()
    db.ensure_schema(connection)
    yield connection
    connection.close()


def test_eligible_row_and_positive_price_ev(conn):
    row = prediction()
    assert integrity.publication_reasons(row) == []
    pid = db.insert_prediction(conn, row)
    assert conn.execute("SELECT bet_side FROM prop_predictions WHERE id=?", (pid,)).fetchone()[0] == "over"
    pid = db.insert_prediction(conn, prediction(kelly_frac=0.0))
    stored = conn.execute("SELECT * FROM prop_predictions WHERE id=?", (pid,)).fetchone()
    assert stored["bet_side"] is None and "no_positive_price_ev" in stored["eligibility_reason"]
    # A large no-vig disagreement still loses money at an expensive offered price.
    pid = db.insert_prediction(conn, prediction(over_odds=-250, kelly_frac=.02))
    stored = conn.execute("SELECT * FROM prop_predictions WHERE id=?", (pid,)).fetchone()
    assert stored["bet_side"] is None and "no_positive_price_ev" in stored["eligibility_reason"]
    assert betting_math.kelly_fraction(.7, 0) == 0
    assert betting_math.american_to_prob(float("inf")) is None


@pytest.mark.parametrize("field,value,reason", [
    ("model_status", "shadow", "shadow_model"),
    ("model_version", None, "legacy_or_unversioned"),
    ("game_pk", None, "missing_game_or_player_identity"),
    ("lineup_confirmed", 0, "lineup_unconfirmed"),
    ("role_verified", 0, "role_or_workload_unverified"),
    ("game_start_time", "2020-01-01T20:00:00Z", "not_current_pregame"),
    ("line_as_of", "2020-01-01T19:00:00Z", "stale_or_missing_price"),
    ("line_as_of", "2090-01-01T19:00:00Z", "stale_or_missing_price"),
    ("line_as_of", "2026-01-01T19:00:00", "stale_or_missing_price"),
    ("book", None, "unpriced"),
    ("price_verified", 0, "fixture_or_unverified_price_source"),
])
def test_every_lane_obeys_publication_gate(conn, field, value, reason):
    pid = db.insert_prediction(conn, prediction(**{field: value}))
    row = conn.execute("SELECT * FROM prop_predictions WHERE id=?", (pid,)).fetchone()
    assert row["bet_side"] is None and row["kelly_frac"] == 0
    assert reason in row["eligibility_reason"]


def test_ledgers_reject_rewrites_and_migration_is_repeatable(conn):
    db.ensure_schema(conn)
    pid = db.insert_prediction(conn, prediction())
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE prop_predictions SET p_over=0.99 WHERE id=?", (pid,))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM prop_predictions WHERE id=?", (pid,))


def test_ui_rechecks_expiration_and_raw_llm_cannot_publish(conn):
    import app
    today = datetime.now(timezone.utc).date().isoformat()
    db.insert_prediction(conn, prediction(source="llm"))
    assert app.load_mlb_props(conn, today, bets_only=True) == []
    db.insert_prediction(conn, prediction(source="judge"))
    assert len(app.load_mlb_props(conn, today, bets_only=True)) == 1
    import mlb.integrity as ui_integrity
    original = ui_integrity.publication_reasons
    # Aging happens without a rerun, so the UI must fail the old recommendation.
    ui_integrity.publication_reasons = lambda r: original(r, now=datetime.now(timezone.utc) + timedelta(hours=3))
    try:
        assert app.load_mlb_props(conn, today, bets_only=True) == []
    finally:
        ui_integrity.publication_reasons = original


def test_lineup_requires_nine_unique_batters(monkeypatch):
    monkeypatch.setattr(data_today, "get_batter_rates", lambda *a, **kw: dict(data_today.LEAGUE))
    assert predict._lineup_rates([1], "R")[1] is False
    assert predict._lineup_rates([1] * 9, "R")[1] is False
    assert predict._lineup_rates(list(range(1, 10)), "R")[1] is True


def test_target_and_future_games_excluded_from_rates_and_workload(monkeypatch):
    splits = []
    for pk, day, ks, pitches in [(1, "2025-04-01", 2, 60), (2, "2025-04-10", 12, 110), (3, "2025-04-15", 15, 115)]:
        splits.append({"game": {"gamePk": pk}, "date": day,
                       "stat": {"battersFaced": 20, "strikeOuts": ks, "gamesStarted": 1,
                                "numberOfPitches": pitches, "hits": 4, "baseOnBalls": 1}})
    monkeypatch.setattr(data_today, "_get", lambda *a, **kw: {"stats": [{"splits": splits + [splits[0]]}]})
    profile = data_today.get_pitcher_profile(1, as_of_date="2025-04-10")
    assert profile["batters_faced"] == 20
    assert profile["recent_pitch_counts"] == [60]
    assert profile["k"] == pytest.approx((2 + 50 * data_today.LEAGUE["k"]) / 70)


def test_historical_platoon_query_cannot_fetch_future_season_totals(monkeypatch):
    seen = []
    monkeypatch.setattr(data_today, "_get", lambda *a, **kw: seen.append(kw) or {"stats": []})
    assert data_today.get_batter_rates(1, "L", as_of_date="2025-04-10") == data_today.LEAGUE
    assert seen and all(p["season"] == 2024 for p in seen)


def test_backtest_grid_and_clv_are_outcome_independent():
    lines = build_backtest_data._lines_for()
    assert len(lines) == 21 and len({(r["market"], r["line"]) for r in lines}) == 21
    assert betting_math.probability_clv(.50, .55, "over") == pytest.approx(.05)
    assert betting_math.probability_clv(.50, .55, "under") == pytest.approx(-.05)
    assert betting_math.probability_clv(None, .55, "over") is None
    rec = backtest._score_line(np.array([6] * 65 + [4] * 35), 6,
        dict(market="pitcher_strikeouts", line=5.5, over_odds=-110, under_odds=-110, prices_verified=True,
             closing_over_odds=-150, closing_under_odds=130))
    assert rec["clv"] == pytest.approx(betting_math.consensus_p_over([(-150, 130)]) - .5)


def test_low_pitch_cap_and_plus_four_batters_are_real_numerical_scenarios():
    rng = np.random.default_rng(2)
    caps = [simulator.sample_pitch_limit([25, 30, 35], rng, pitch_cap=40) for _ in range(500)]
    assert np.mean(caps) < 35 and max(caps) <= 40
    arr = np.array([3, 4, 5, 6, 7])
    base = simulator.prob_over(arr, 5.5)[0]
    stress = simulator.extra_batters_k_probability(arr, 5.5, .25)
    assert stress > base
    assert simulator.extra_batters_k_probability(arr, 5.5, .25, extra_batters=0) == base


def test_missing_stress_test_cannot_authorize_k_under():
    import llm_saved_pipeline as policy
    row = dict(key=("pitcher_strikeouts", 5.5), action="bet", p_over=.36, market_p=.5)
    assert policy._select_primary_key([row]) is None
    assert policy._select_primary_key([{**row, "sim": {"stress_p_over": .48}}]) is None
    assert policy._select_primary_key([{**row, "sim": {"stress_p_over": .40}}]) == row["key"]


def test_grading_retries_nonfinal_and_failed_boxes_and_distinguishes_games(conn, monkeypatch):
    day = "2025-07-01"
    for pk in (1, 2):
        db.insert_prediction(conn, prediction(game_date=day, game_pk=pk))
    games = [{"game_pk": pk, "status": "Preview"} for pk in (1, 2)]
    monkeypatch.setattr(data_today, "get_schedule", lambda gd: games)
    assert grade.grade_date(day) == 0
    for g in games:
        g["status"] = "Final"
    monkeypatch.setattr(data_today, "get_starter_actuals", lambda pk: (_ for _ in ()).throw(RuntimeError("transient")))
    assert grade.grade_date(day) == 0
    assert conn.execute("SELECT COUNT(*) FROM prop_results").fetchone()[0] == 0
    monkeypatch.setattr(data_today, "get_starter_actuals", lambda pk: {
        10: {"name": "Same Pitcher", "ks": 7 if pk == 1 else 3, "outs": 18},
        20: {"name": "Other", "ks": 2, "outs": 15}})
    assert grade.grade_date(day) == 2
    result = conn.execute("SELECT outcome FROM prop_results ORDER BY prediction_id").fetchall()
    assert [r[0] for r in result] == ["over", "under"]
    assert grade.grade_date(day) == 0


def test_close_excludes_live_prices_and_disappeared_thresholds(conn):
    row = prediction(game_date="2025-07-01")
    line = {"game_date": row["game_date"], "game_pk": 1, "player_id": 10,
            "player_name": row["player_name"], "event_id": "e", "market": row["market"],
            "line": 5.5, "book": "A", "over_odds": -110, "under_odds": -110,
            "is_live": True}
    db.insert_lines(conn, [
        {**line, "as_of": "2025-07-01T18:00:00Z"},
        {**line, "as_of": "2025-07-01T18:59:00Z", "over_odds": -150, "under_odds": 130},
        {**line, "as_of": "2025-07-01T19:10:00Z", "over_odds": 200, "under_odds": -250},
    ])
    close = grade.closing_probability(conn, row, "2025-07-01T19:00:00Z")
    assert close == pytest.approx(betting_math.consensus_p_over([(-150, 130)]))
    db.insert_lines(conn, [{**line, "line": 6.5, "as_of": "2025-07-01T18:59:30Z"}])
    assert grade.closing_probability(conn, row, "2025-07-01T19:00:00Z") is None


def test_calibration_uses_first_raw_pregame_forecast_and_no_alt_multiplicity(conn):
    day = "2025-07-01"
    for t, line, raw, adjusted in [("16:00", 5.5, .7, .6), ("16:01", 6.5, .5, .4),
                                    ("16:10", 5.5, .8, .65), ("20:00", 5.5, .99, .9)]:
        pid = db.insert_prediction(conn, prediction(game_date=day, created_at=f"{day}T{t}:00Z",
                     game_start_time=f"{day}T19:00:00Z", line=line, raw_p_over=raw, p_over=adjusted))
        conn.execute("INSERT INTO prop_results (prediction_id,graded_at,actual,outcome) VALUES (?,?,?,?)",
                     (pid, "2025-07-02T12:00:00Z", 7, "over"))
    conn.commit()
    rows = calibration._eligible_rows(conn, day)
    assert len(rows) == 1 and rows[0]["p_over"] == .7


def test_calibration_monotonic_and_small_or_future_fits_unused(conn):
    rows = ([{"p_over": .51, "outcome": "over"}] * 90
            + [{"p_over": .61, "outcome": "under"}] * 90)
    buckets = calibration._build_buckets(rows)
    assert buckets[0]["calibrated_p"] <= buckets[1]["calibrated_p"]
    assert calibration.calibrated_probability(conn, .61, "pitcher_strikeouts", "sim", "2025-07-01") == .61
    conn.execute("""INSERT INTO calibration_runs
      (run_id,created_at,through_date,market,source,sample_size,mode,metadata_json)
      VALUES ('future','2026-01-01T00:00:00Z','2025-06-01','pitcher_strikeouts','sim',500,'full','{}')""")
    assert calibration.calibrated_probability(conn, .61, "pitcher_strikeouts", "sim", "2025-07-01") == .61


def test_pickem_keeps_push_mass_and_requires_real_entry_terms(conn):
    prizepicks.ensure_schema(conn)
    prizepicks._log_pick(conn, "r1", "2025-07-01", {"game_pk": 1},
                        {"id": 10, "name": "Pitcher"},
                        {"line": 6, "market": "pitcher_strikeouts", "odds_type": "standard"},
                        np.array([6, 6, 6, 7, 5]))
    row = conn.execute("SELECT * FROM prizepicks_picks").fetchone()
    assert row["p_pick"] == pytest.approx(.2) and row["p_push"] == pytest.approx(.6)
    assert row["p_over_conditional"] == .5 and not row["playable"]
    assert row["ev_2"] is None and row["ev_3"] is None
    assert prizepicks.entry_ev({"WW": .4, "WP": .2, "LW": .4},
                              {"WW": 3, "WP": 1, "LW": 0}) == pytest.approx(.4)
    with pytest.raises(ValueError, match="settlement"):
        prizepicks.entry_ev({"WW": .4, "WP": .2, "LW": .4}, {"WW": 3, "LW": 0})


def test_training_medians_use_past_and_dates_do_not_cross_split():
    days = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-03"]))
    x = pd.DataFrame({"a": [1., np.nan, 1000., 2000.], "b": [np.nan, np.nan, 5., 6.]})
    filled, medians, train, validation = chronological_features(x, days, .67)
    assert medians["a"] == 1 and medians["b"] == 0
    assert filled.iloc[1]["a"] == 1
    assert max(days.iloc[train]) < min(days.iloc[validation])


def test_training_includes_both_starting_pitchers():
    frame = pd.DataFrame([
        {"inning_topbot": "Top", "at_bat_number": 1, "pitcher": 10, "events": "strikeout"},
        {"inning_topbot": "Top", "at_bat_number": 2, "pitcher": 10, "events": "field_out"},
        {"inning_topbot": "Bot", "at_bat_number": 3, "pitcher": 20, "events": "strikeout"},
        {"inning_topbot": "Bot", "at_bat_number": 4, "pitcher": 20, "events": "grounded_into_double_play"},
    ])
    assert build_training.get_starter_stats(frame) == [(10, 1, 2), (20, 1, 3)]


def test_usage_evidence_must_match_game_date_identity_and_be_reviewed():
    import usage_evidence
    now = datetime.now(timezone.utc)
    game = {"game_pk": 1, "game_date": now.date().isoformat(),
            "game_start_time": (now + timedelta(hours=2)).isoformat()}
    evidence = {"game_pk": 1, "player_id": 10, "game_date": game["game_date"],
                "observed_at": (now - timedelta(hours=1)).isoformat(),
                "reviewed_at": now.isoformat(), "reviewed": True,
                "expected_role": "opener", "pitch_cap": 35,
                "source_url": "https://www.mlb.com/example-manager-report",
                "evidence": "Manager described an opening assignment and 35-pitch limit."}
    assert usage_evidence.match([evidence], game, {"id": 10}, now) == evidence
    for changed in ({"player_id": 11}, {"game_pk": 2}, {"reviewed": False},
                    {"pitch_cap": -5}, {"observed_at": (now - timedelta(days=2)).isoformat()}):
        assert usage_evidence.match([{**evidence, **changed}], game, {"id": 10}, now) is None
    assert usage_evidence.match([None, "broken", {}], game, {"id": 10}, now) is None


def test_undated_and_future_factors_cannot_enter_historical_forecasts(monkeypatch):
    import factors
    monkeypatch.setattr(factors, "_park_cache", {"NYM": {"k_factor": 1.15}})
    assert factors.park_k_factor("New York Mets", "2025-07-01") == 1.0
    monkeypatch.setattr(factors, "_park_cache", {"NYM": {"k_factor": 1.15,
        "created_at": "2025-06-01T00:00:00Z", "through_date": "2025-05-31"}})
    assert factors.park_k_factor("New York Mets", "2025-07-01") == 1.15
    assert factors.park_k_factor("New York Mets", "2025-05-01") == 1.0


def test_unconditional_pickem_side_selection_with_many_pushes(conn):
    prizepicks.ensure_schema(conn)
    prizepicks._log_pick(conn, "r", "2025-07-01", {"game_pk": 1},
                        {"id": 10, "name": "P"},
                        {"line": 6, "market": "pitcher_strikeouts", "odds_type": "standard"},
                        np.array([6] * 6 + [7] * 3 + [5]))
    row = conn.execute("SELECT * FROM prizepicks_picks").fetchone()
    assert row["pick_side"] == "more" and row["p_pick"] == pytest.approx(.3)


def test_push_stake_in_roi_denominator_but_not_decided_hit_rate():
    lines = {"market": "pitcher_strikeouts", "line": 6., "over_odds": -110,
             "under_odds": -110, "prices_verified": True}
    arr = np.array([7] * 70 + [5] * 30)
    pushed = backtest._score_line(arr, 6, lines)
    won = backtest._score_line(arr, 7, lines)
    summary = backtest.betting_summary([pushed, won])
    assert summary["n_bets"] == 2 and summary["n_decided"] == 1 and summary["n_pushes"] == 1
    assert summary["hit_rate"] == 1
    assert summary["roi_flat"] == pytest.approx((100 / 110) / 2)


def test_missing_archive_times_and_postgame_close_stay_unverified(tmp_path):
    import json
    record = dict(game_date="2025-07-01", game_pk=1, player_id=10, player_name="P",
                  market="pitcher_strikeouts", line=5.5, over_odds=-110, under_odds=-110,
                  closing_over_odds=-150, closing_under_odds=130)
    path = tmp_path / "prices.json"
    path.write_text(json.dumps([record]))
    archive = backtest.load_odds_archive(str(path))
    assert not archive[("2025-07-01", 1, 10)][0]["prices_verified"]
    record.update(as_of="2025-07-01T18:00:00Z", decision_time="2025-07-01T18:05:00Z",
                  game_start_time="2025-07-01T19:00:00Z", closing_as_of="2025-07-01T19:10:00Z")
    path.write_text(json.dumps([record]))
    line = backtest.load_odds_archive(str(path))[("2025-07-01", 1, 10)][0]
    assert line["prices_verified"] and line["closing_over_odds"] is None


def test_nonfinite_probabilities_are_rejected(conn):
    for bad in (float("nan"), float("inf"), -0.01, 1.01):
        with pytest.raises(ValueError, match="invalid p_over"):
            db.insert_prediction(conn, prediction(p_over=bad))


def test_shadow_skill_projection_cannot_read_the_forecast_season():
    import skill_baseline
    seasons = {year: {10: {"bf": 200, "gs": 10, "k": 40 + year % 3,
                           "bb": 20, "hit": 45}} for year in range(2020, 2026)}
    before = skill_baseline.project(seasons, 2025, 10, "k", .5, 200)
    seasons[2025][10]["k"] = 150
    assert skill_baseline.project(seasons, 2025, 10, "k", .5, 200) == before
