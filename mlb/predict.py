"""Run the simulator over today's slate and log prop predictions.

Flow per probable starter:
  schedule (probables + confirmed lineups) -> per-PA rates -> Monte Carlo
  -> calibrated P(over) at current posted lines -> de-vigged market consensus
  -> conservative edge gate -> prop_predictions.

Synthetic no-book rows remain observable but are never eligible bets.
"""
import json
import os
import sys
from datetime import datetime, timezone

import calibration
import data_today
import db
import decision
import form
import simulator
from betting_math import consensus_p_over
from factors import park_k_factor, ump_k_factor
from timeutils import slate_today

# A 3% point-estimate discrepancy was too permissive for a noisy, incomplete
# model. Six percentage points is the default minimum after calibration.
MIN_EDGE = decision.MIN_EDGE
N_SIMS = int(os.getenv("MLB_N_SIMS", "20000"))


def _norm(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


def _lineup_rates(batter_ids, pitcher_hand):
    """Per-PA rates for the nine batters; league average when unposted."""
    confirmed = decision.complete_lineup(batter_ids)
    if not batter_ids:
        return [dict(data_today.LEAGUE) for _ in range(9)], False
    lineup = [
        data_today.get_batter_rates(batter_id, pitcher_hand)
        for batter_id in batter_ids[:9]
    ]
    while len(lineup) < 9:
        lineup.append(dict(data_today.LEAGUE))
    return lineup, confirmed


def simulate_pitcher(
    pitcher,
    opp_lineup_ids,
    k_park=1.0,
    k_ump=1.0,
    n_sims=N_SIMS,
    game_date=None,
):
    """Simulate one starter end-to-end from serving-time data."""
    try:
        profile = data_today.get_pitcher_profile(
            pitcher["id"], as_of_date=game_date
        )
    except TypeError:
        profile = data_today.get_pitcher_profile(pitcher["id"])
    lineup, confirmed = _lineup_rates(opp_lineup_ids, pitcher["hand"])
    k_rate = form.blended_k_rate(pitcher["id"], profile["k"])
    k_form = form.velo_k_factor(pitcher["id"])
    ks, outs = simulator.simulate_starter(
        {"k": k_rate, "bb": profile["bb"], "hit": profile["hit"]},
        lineup,
        recent_pitch_counts=profile["recent_pitch_counts"],
        recent_starts=profile.get("recent_starts"),
        k_park=k_park,
        k_ump=k_ump,
        k_form=k_form,
        n_sims=n_sims,
    )
    workload = profile.get("workload") or {
        "confirmed": False,
        "reasons": ["dated recent-start workload evidence unavailable"],
    }
    quality_reasons = []
    if not confirmed:
        quality_reasons.append(
            "opposing starting lineup is not complete (nine hitters)"
        )
    quality_reasons.extend(workload.get("reasons") or [])
    return ks, outs, {
        "confirmed": confirmed,
        "workload": workload,
        "quality_reasons": quality_reasons,
        "k_form": k_form,
        "recent_pitch_counts": profile["recent_pitch_counts"],
    }


def predict_date(game_date=None, n_sims=N_SIMS):
    game_date = game_date or slate_today()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = db.connect()
    db.ensure_schema(conn)

    lines_by_player = {}
    for row in db.active_lines(conn, game_date):
        key = (_norm(row["player_name"]), row["market"])
        lines_by_player.setdefault(key, []).append(row)

    games = data_today.get_schedule(game_date)
    print(f"[predict] {game_date}: {len(games)} games, run {run_id}")
    n_logged = 0

    for game in games:
        game_reasons = data_today.game_gate_reasons(game, game_date)
        k_park = park_k_factor(game["home_team"])
        ump_name = data_today.get_home_plate_umpire(game["game_pk"])
        k_ump = ump_k_factor(ump_name)

        for side in ("home", "away"):
            pitcher = game[f"{side}_pitcher"]
            if not pitcher:
                continue
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
                    game_date=game_date,
                )
            except Exception as exc:
                print(f"[predict] {pitcher['name']}: failed ({exc})")
                continue

            samples = {"pitcher_strikeouts": ks, "pitcher_outs": outs}
            for market, arr in samples.items():
                rows = lines_by_player.get((_norm(pitcher["name"]), market), [])
                n_logged += _log_market(
                    conn,
                    run_id,
                    game_date,
                    game,
                    pitcher,
                    market,
                    arr,
                    rows,
                    quality_reasons=game_reasons + meta["quality_reasons"],
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
        conn,
        raw_p_over,
        market,
        "sim",
        as_of_date=game_date,
        policy_version=decision.POLICY_VERSION,
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
    quality_reasons=(),
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
        "policy_version": decision.POLICY_VERSION,
    }

    if not line_rows:
        mean = float(arr.mean())
        line = round(mean - 0.5) + 0.5
        raw_p_over, _, p_push = simulator.prob_over(arr, line)
        reasons = list(quality_reasons) + ["no live two-way market anchor"]
        db.insert_prediction(conn, {
            **base,
            "line": line,
            "p_over": raw_p_over,
            "raw_p_over": raw_p_over,
            "decision_p_over": raw_p_over,
            "p_push": p_push,
            "mean": mean,
            "book": None,
            "over_odds": None,
            "under_odds": None,
            "market_p_over": None,
            "edge_over": None,
            "edge_under": None,
            "candidate_side": None,
            "bet_side": None,
            "kelly_frac": 0.0,
            "rationale": json.dumps({
                "rationale": "; ".join(reasons),
                "hold_reasons": reasons,
            }),
            "quality_json": json.dumps({"hold_reasons": reasons}),
        })
        return 1

    by_line = {}
    for row in line_rows:
        by_line.setdefault(row["line"], []).append(row)

    for line, books in by_line.items():
        raw_p_over, _, p_push = simulator.prob_over(arr, line)
        p_over, p_under = _calibrated_sides(
            conn, raw_p_over, market, game_date
        )
        market_p = consensus_p_over(
            [(book["over_odds"], book["under_odds"]) for book in books]
        )
        priced_books = {
            book["book"]
            for book in books
            if book["over_odds"] is not None and book["under_odds"] is not None
        }
        validation = calibration.validation_status(
            conn,
            market,
            "sim",
            as_of_date=game_date,
            policy_version=decision.POLICY_VERSION,
        )
        call = decision.decide(
            p_over,
            market_p,
            book_count=len(priced_books),
            validation=validation,
            quality_reasons=quality_reasons,
        )

        candidate_side, bet_side, kelly, best = call.bet_side, None, 0.0, None
        if call.bet_side == "over":
            best = _best_price(books, "over_odds")
        elif call.bet_side == "under":
            best = _best_price(books, "under_odds")
        if best is None:
            best = books[0]

        rationale_parts = list(call.hold_reasons)
        if candidate_side:
            rationale_parts.append(
                "candidate only: awaiting independent LLM safety-judge confirmation"
            )
        if abs(p_over - raw_p_over) >= 0.005:
            rationale_parts.append(
                f"calibrated from raw P(over) {raw_p_over:.3f} "
                f"to {p_over:.3f}"
            )
        rationale = json.dumps({
            "rationale": "; ".join(rationale_parts),
            "hold_reasons": list(call.hold_reasons),
            "validation": validation,
            "model_p_over": p_over,
            "decision_p_over": call.decision_p_over,
        })

        db.insert_prediction(conn, {
            **base,
            "line": line,
            "p_over": p_over,
            "raw_p_over": raw_p_over,
            "decision_p_over": call.decision_p_over,
            "p_push": p_push,
            "mean": float(arr.mean()),
            "book": best["book"],
            "over_odds": best["over_odds"],
            "under_odds": best["under_odds"],
            "market_p_over": market_p,
            "edge_over": call.edge_over,
            "edge_under": call.edge_under,
            "candidate_side": candidate_side,
            "bet_side": bet_side,
            "kelly_frac": kelly,
            "rationale": rationale,
            "quality_json": json.dumps({
                "hold_reasons": list(call.hold_reasons),
                "validation": validation,
                "book_count": len(priced_books),
            }),
        })
        logged += 1
    return logged


def _best_price(books, side_col):
    """Book offering the highest-payout American price on a side."""
    priced = [book for book in books if book[side_col] is not None]
    if not priced:
        return None
    from betting_math import american_to_decimal
    return max(priced, key=lambda book: american_to_decimal(book[side_col]))


if __name__ == "__main__":
    predict_date(sys.argv[1] if len(sys.argv) > 1 else None)
