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

from mlb import db
from mlb.prediction import form, simulator
from mlb.prediction.betting_math import consensus_p_over, kelly_fraction
from mlb.prediction.factors import park_k_factor, ump_k_factor
from mlb.providers import mlb_api
from mlb.training import calibration

# A 3% point-estimate discrepancy was too permissive for a noisy, incomplete
# model. Six percentage points is the default minimum after calibration.
MIN_EDGE = float(os.getenv("MLB_MIN_EDGE", "0.06"))
N_SIMS = int(os.getenv("MLB_N_SIMS", "20000"))


def _norm(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


def _lineup_rates(batter_ids, pitcher_hand):
    """Per-PA rates for the nine batters; league average when unposted."""
    if not batter_ids:
        return [dict(mlb_api.LEAGUE) for _ in range(9)], False
    lineup = [
        mlb_api.get_batter_rates(batter_id, pitcher_hand)
        for batter_id in batter_ids[:9]
    ]
    while len(lineup) < 9:
        lineup.append(dict(mlb_api.LEAGUE))
    return lineup, True


def simulate_pitcher(
    pitcher,
    opp_lineup_ids,
    k_park=1.0,
    k_ump=1.0,
    defense_context=None,
    k_arsenal=1.0,
    n_sims=N_SIMS,
):
    """Simulate one starter end-to-end from serving-time data."""
    profile = mlb_api.get_pitcher_profile(pitcher["id"])
    lineup, confirmed = _lineup_rates(opp_lineup_ids, pitcher["hand"])
    k_rate = form.blended_k_rate(pitcher["id"], profile["k"])
    k_form = form.velo_k_factor(pitcher["id"])
    ks, outs = simulator.simulate_starter(
        {"k": k_rate, "bb": profile["bb"], "hit": profile["hit"]},
        lineup,
        recent_pitch_counts=profile["recent_pitch_counts"],
        k_park=k_park,
        k_ump=k_ump,
        k_form=k_form,
        k_arsenal=k_arsenal,
        defense_context=defense_context,
        n_sims=n_sims,
    )
    return ks, outs, {
        "confirmed": confirmed,
        "k_form": k_form,
        "k_arsenal": k_arsenal,
        "recent_pitch_counts": profile["recent_pitch_counts"],
    }


def predict_date(game_date=None, n_sims=N_SIMS):
    game_date = game_date or date.today().isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = db.connect()
    db.ensure_schema(conn)

    lines_by_player = {}
    for row in db.latest_lines(conn, game_date):
        key = (_norm(row["player_name"]), row["market"])
        lines_by_player.setdefault(key, []).append(row)

    games = mlb_api.get_schedule(game_date)
    print(f"[predict] {game_date}: {len(games)} games, run {run_id}")
    n_logged = 0

    for game in games:
        k_park = park_k_factor(game["home_team"])
        ump_name = mlb_api.get_home_plate_umpire(game["game_pk"])
        k_ump = ump_k_factor(ump_name)

        for side in ("home", "away"):
            pitcher = game[f"{side}_pitcher"]
            if not pitcher:
                continue
            opp_lineup_ids = (
                game["away_lineup"] if side == "home" else game["home_lineup"]
            )
            try:
                fielding_team = game[f"{side}_team"]
                defense_context = db.defense_context(
                    conn, pitcher["id"], fielding_team, game_date
                )
                k_arsenal = db.arsenal_matchup_factor(
                    conn, pitcher["id"], opp_lineup_ids, game_date
                )
                ks, outs, meta = simulate_pitcher(
                    pitcher,
                    opp_lineup_ids,
                    k_park=k_park,
                    k_ump=k_ump,
                    defense_context=defense_context,
                    k_arsenal=k_arsenal,
                    n_sims=n_sims,
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
                    allow_bet=bool(meta["confirmed"]),
                )
            print(
                f"[predict] {pitcher['name']} ({side}, "
                f"lineup={'posted' if meta['confirmed'] else 'league-avg HOLD'}, "
                f"ump={ump_name or 'TBD'} k={k_ump:.2f}, "
                f"form k={meta['k_form']:.2f}, "
                f"arsenal k={meta['k_arsenal']:.2f}): "
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
    }

    if not line_rows:
        mean = float(arr.mean())
        line = round(mean - 0.5) + 0.5
        raw_p_over, _, _ = simulator.prob_over(arr, line)
        db.insert_prediction(conn, {
            **base,
            "line": line,
            "p_over": raw_p_over,
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
        raw_p_over, _, _ = simulator.prob_over(arr, line)
        p_over, p_under = _calibrated_sides(
            conn, raw_p_over, market, game_date
        )
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
    priced = [book for book in books if book[side_col] is not None]
    if not priced:
        return None
    from mlb.prediction.betting_math import american_to_decimal
    return max(priced, key=lambda book: american_to_decimal(book[side_col]))


if __name__ == "__main__":
    predict_date(sys.argv[1] if len(sys.argv) > 1 else None)
