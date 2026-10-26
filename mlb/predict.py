"""Run the simulator over today's slate and log prop predictions.

Flow per probable starter:
  schedule (probables + confirmed lineups) -> per-PA rates -> Monte Carlo
  -> calibrated P(over) at current posted lines -> de-vigged market consensus
  -> conservative edge gate -> prop_predictions.

Synthetic no-book rows remain observable but are never eligible bets.
"""
import os
import sys
from datetime import date, datetime, timezone

import calibration
import data_today
import db
import form
import integrity
import simulator
import usage_evidence
from betting_math import consensus_p_over, kelly_fraction
from factors import park_k_factor, ump_k_factor, _norm_name

# A 3% point-estimate discrepancy was too permissive for a noisy, incomplete
# model. Six percentage points is the default minimum after calibration.
MIN_EDGE = float(os.getenv("MLB_MIN_EDGE", "0.06"))
N_SIMS = int(os.getenv("MLB_N_SIMS", "20000"))


def _norm(name):
    return _norm_name(name)


def _lineup_rates(batter_ids, pitcher_hand, as_of_date=None):
    """Per-PA rates for the nine batters; league average when unposted."""
    if not batter_ids:
        return [dict(data_today.LEAGUE) for _ in range(9)], False
    lineup = [
        data_today.get_batter_rates(batter_id, pitcher_hand, as_of_date=as_of_date)
        for batter_id in batter_ids[:9]
    ]
    while len(lineup) < 9:
        lineup.append(dict(data_today.LEAGUE))
    return lineup, len(batter_ids) == 9 and len(set(batter_ids)) == 9


def simulate_pitcher(
    pitcher,
    opp_lineup_ids,
    k_park=1.0,
    k_ump=1.0,
    n_sims=N_SIMS,
    as_of_date=None,
):
    """Simulate one starter end-to-end from serving-time data."""
    profile = data_today.get_pitcher_profile(pitcher["id"], as_of_date=as_of_date)
    lineup, confirmed = _lineup_rates(opp_lineup_ids, pitcher["hand"], as_of_date)
    # Hand-set CSW/velocity multipliers are an unvalidated challenger. A dated
    # snapshot and out-of-sample fit are required before they alter the baseline.
    k_rate, k_form = profile["k"], 1.0
    ks, outs = simulator.simulate_starter(
        {"k": k_rate, "bb": profile["bb"], "hit": profile["hit"]},
        lineup,
        recent_pitch_counts=profile["recent_pitch_counts"],
        k_park=k_park,
        k_ump=k_ump,
        k_form=k_form,
        n_sims=n_sims,
        pitch_cap=pitcher.get("pitch_cap"),
    )
    return ks, outs, {
        "confirmed": confirmed,
        "k_form": k_form,
        "recent_pitch_counts": profile["recent_pitch_counts"],
        "role_verified": bool(pitcher.get("role_verified")),
        "stress_k_probability": float(simulator.build_matchup_probs(
            {"k": k_rate, "bb": profile["bb"], "hit": profile["hit"]},
            lineup, k_park, k_ump, k_form)[:, 0].max()),
    }


def predict_date(game_date=None, n_sims=N_SIMS):
    game_date = game_date or date.today().isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    conn = db.connect()
    db.ensure_schema(conn)

    lines_by_player = {}
    for row in db.latest_lines(conn, game_date):
        key = (_norm(row["player_name"]), row["market"])
        lines_by_player.setdefault(key, []).append(row)

    games = data_today.get_schedule(game_date)
    name_counts = {}
    for game in games:
        for side in ("home", "away"):
            starter = game.get(f"{side}_pitcher")
            if starter:
                name_counts[_norm(starter["name"])] = name_counts.get(_norm(starter["name"]), 0) + 1
    print(f"[predict] {game_date}: {len(games)} games, run {run_id}")
    n_logged = 0

    for game in games:
        if game.get("status") != "Preview":
            continue
        k_park = park_k_factor(game["home_team"], as_of_date=game_date)
        ump_name = data_today.get_home_plate_umpire(game["game_pk"])
        k_ump = ump_k_factor(ump_name, as_of_date=game_date)

        for side in ("home", "away"):
            pitcher = game[f"{side}_pitcher"]
            if not pitcher:
                continue
            pitcher = usage_evidence.attach(game, pitcher)
            opp_lineup_ids = (
                game["away_lineup"] if side == "home" else game["home_lineup"]
            )
            try:
                ks, outs, meta = simulate_pitcher(
                    pitcher,
                    opp_lineup_ids,
                    k_park=k_park,
                    k_ump=k_ump,
                    n_sims=n_sims,
                    as_of_date=game_date,
                )
            except Exception as exc:
                print(f"[predict] {pitcher['name']}: failed ({exc})")
                continue

            samples = {"pitcher_strikeouts": ks, "pitcher_outs": outs}
            for market, arr in samples.items():
                rows = lines_by_player.get((_norm(pitcher["name"]), market), [])
                rows = [r for r in rows
                        if (r["game_pk"] == game["game_pk"] or
                            (r["game_pk"] is None and name_counts[_norm(pitcher["name"])] == 1))
                        and r["player_id"] in (None, pitcher["id"])]
                n_logged += _log_market(
                    conn,
                    run_id,
                    game_date,
                    game,
                    pitcher,
                    market,
                    arr,
                    rows,
                    allow_bet=bool(meta["confirmed"]),
                    role_verified=meta["role_verified"],
                    stress_k_probability=meta["stress_k_probability"],
                )
            print(
                f"[predict] {pitcher['name']} ({side}, "
                f"lineup={'posted' if meta['confirmed'] else 'league-avg HOLD'}, "
                f"ump={ump_name or 'TBD'} k={k_ump:.2f}, "
                f"form k={meta['k_form']:.2f}): "
                f"K mean {ks.mean():.2f}, outs mean {outs.mean():.2f}"
            )

    print(f"[predict] logged {n_logged} predictions")
    conn.close()
    return n_logged


def _calibrated_sides(conn, raw_p_over, market, game_date):
    p_over = calibration.calibrated_probability(
        conn, raw_p_over, market, "sim", as_of_date=game_date
    )
    return p_over, 1.0 - p_over


def _log_market(
    conn,
    run_id,
    game_date,
    game,
    pitcher,
    market,
    arr,
    line_rows,
    allow_bet=True,
    role_verified=False,
    stress_k_probability=None,
):
    logged = 0
    base = {
        "run_id": run_id,
        "created_at": db.utcnow(),
        "game_date": game_date,
        "game_pk": game["game_pk"],
        "player_name": pitcher["name"],
        "player_id": pitcher["id"],
        "market": market,
        "source": "sim",
        "rationale": None,
        "model_version": integrity.MODEL_VERSION,
        "model_status": "shadow",
        "game_start_time": game.get("game_start_time"),
        "lineup_confirmed": int(allow_bet),
        "role_verified": int(role_verified),
        "role_evidence_json": pitcher.get("role_evidence_json"),
    }

    if not line_rows:
        mean = float(arr.mean())
        line = round(mean - 0.5) + 0.5
        raw_p_over, _, p_push = simulator.prob_over(arr, line)
        db.insert_prediction(conn, {
            **base,
            "line": line,
            "p_over": raw_p_over,
            "raw_p_over": raw_p_over,
            "p_push": p_push,
            "mean": mean,
            "book": None,
            "over_odds": None,
            "under_odds": None,
            "market_p_over": None,
            "edge_over": None,
            "edge_under": None,
            "bet_side": None,
            "kelly_frac": 0.0,
        })
        return 1

    by_line = {}
    for row in line_rows:
        by_line.setdefault(row["line"], []).append(row)

    for line, books in by_line.items():
        decision_time = integrity.timestamp(base["created_at"])
        fresh_live = [book for book in books if dict(book).get("is_live")
                      and integrity.timestamp(book["as_of"])
                      and 0 <= (decision_time - integrity.timestamp(book["as_of"])).total_seconds()
                      <= integrity.MAX_LINE_AGE_MINUTES * 60]
        if fresh_live:
            books = fresh_live  # fixture/stale books cannot contaminate a live consensus
        raw_p_over, _, p_push = simulator.prob_over(arr, line)
        p_over, calibration_run_id = calibration.calibrated_probability(
            conn, raw_p_over, market, "sim", as_of_date=game_date,
            return_snapshot=True,
        )
        p_under = 1.0 - p_over
        stress_p = (simulator.extra_batters_k_probability(arr, line, stress_k_probability)
                    if market == "pitcher_strikeouts" and stress_k_probability is not None else None)
        market_p = consensus_p_over(
            [(book["over_odds"], book["under_odds"]) for book in books]
        )
        edge_over = p_over - market_p if market_p is not None else None
        edge_under = (
            p_under - (1.0 - market_p) if market_p is not None else None
        )

        bet_side, kelly, best = None, 0.0, None
        if allow_bet and edge_over is not None:
            if edge_over >= MIN_EDGE and edge_over >= (edge_under or 0.0):
                bet_side = "over"
                best = _best_price(books, "over_odds")
                kelly = kelly_fraction(p_over, best and best["over_odds"])
            elif edge_under is not None and edge_under >= MIN_EDGE:
                bet_side = "under"
                best = _best_price(books, "under_odds")
                kelly = kelly_fraction(p_under, best and best["under_odds"])
        if best is None:
            best = books[0]

        rationale = None
        if not allow_bet:
            rationale = "HOLD: opposing lineup not confirmed"
        elif abs(p_over - raw_p_over) >= 0.005:
            rationale = (
                f"calibrated from raw P(over) {raw_p_over:.3f} "
                f"to {p_over:.3f}"
            )

        db.insert_prediction(conn, {
            **base,
            "line": line,
            "p_over": p_over,
            "raw_p_over": raw_p_over,
            "p_push": p_push,
            "line_as_of": best["as_of"],
            "stress_p_over": stress_p,
            "price_verified": bool(dict(best).get("is_live")),
            "calibration_run_id": calibration_run_id,
            "mean": float(arr.mean()),
            "book": best["book"],
            "over_odds": best["over_odds"],
            "under_odds": best["under_odds"],
            "market_p_over": market_p,
            "edge_over": edge_over,
            "edge_under": edge_under,
            "bet_side": bet_side,
            "kelly_frac": kelly,
            "rationale": rationale,
        })
        logged += 1
    return logged


def _best_price(books, side_col):
    """Book offering the highest-payout American price on a side."""
    from betting_math import valid_american
    priced = [book for book in books if valid_american(book[side_col])]
    if not priced:
        return None
    from betting_math import american_to_decimal
    return max(priced, key=lambda book: american_to_decimal(book[side_col]))


if __name__ == "__main__":
    predict_date(sys.argv[1] if len(sys.argv) > 1 else None)
