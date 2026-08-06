"""Run the merged v4.2/v5 numerical forecast lane over a slate.

Every exact live line is logged before the result is known. The simulator is a
shadow/candidate lane; only the final judge can promote one primary market per
pitcher after validation and exposure controls.
"""
from __future__ import annotations

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

MIN_EDGE = decision.MIN_EDGE
N_SIMS = int(os.getenv("MLB_N_SIMS", "20000"))


def _norm(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


def _lineup_rates(batter_ids, pitcher_hand):
    confirmed = decision.complete_lineup(batter_ids)
    ids = list(batter_ids or [])[:9]
    lineup = [data_today.get_batter_rates(batter_id, pitcher_hand) for batter_id in ids]
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
    try:
        profile = data_today.get_pitcher_profile(pitcher["id"], as_of_date=game_date)
    except TypeError:
        profile = data_today.get_pitcher_profile(pitcher["id"])
    lineup, confirmed = _lineup_rates(opp_lineup_ids, pitcher["hand"])
    k_rate = form.blended_k_rate(pitcher["id"], profile["k"])
    k_form = form.velo_k_factor(pitcher["id"])
    workload = profile.get("workload") or {
        "confirmed": False,
        "reasons": ["dated recent-start workload evidence unavailable"],
    }
    detailed = simulator.simulate_starter_detailed(
        {"k": k_rate, "bb": profile["bb"], "hit": profile["hit"]},
        lineup,
        recent_pitch_counts=profile.get("recent_pitch_counts"),
        recent_starts=profile.get("recent_starts"),
        verified_planned_limit=profile.get("verified_planned_limit"),
        manager_hook_factor=float(profile.get("manager_hook_factor") or 1.0),
        bullpen_pressure=float(profile.get("bullpen_pressure") or 0.0),
        score_environment=float(profile.get("score_environment") or 1.0),
        pitches_per_pa_factor=float(profile.get("pitches_per_pa_factor") or 1.0),
        k_park=k_park,
        k_ump=k_ump,
        k_form=k_form,
        n_sims=n_sims,
    )
    reasons = []
    if not confirmed:
        reasons.append("opposing starting lineup is projected or incomplete")
    reasons.extend(workload.get("reasons") or [])
    return detailed["ks"], detailed["outs"], {
        "confirmed": confirmed,
        "lineup_status": "confirmed" if confirmed else "projected",
        "workload": workload,
        "workload_mix": detailed["workload"],
        "quality_reasons": reasons,
        "k_form": k_form,
        "recent_pitch_counts": profile.get("recent_pitch_counts") or [],
        "batters_faced": detailed["batters_faced"],
        "pitches": detailed["pitches"],
    }


def predict_date(game_date=None, n_sims=N_SIMS):
    game_date = game_date or slate_today()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = db.connect(); db.ensure_schema(conn)
    active = list(db.active_lines(conn, game_date))
    board_as_of = max((row["as_of"] for row in active), default=None)
    db.record_run(
        conn, run_id, game_date, decision.POLICY_VERSION,
        mode="shadow" if decision.SHADOW_MODE else "production",
        source="sim", board_as_of=board_as_of, line_count=len(active),
        status="started", metadata={"n_sims": n_sims},
    )

    lines_by_player = {}
    for row in active:
        lines_by_player.setdefault((_norm(row["player_name"]), row["market"]), []).append(row)

    games = data_today.get_schedule(game_date)
    print(f"[predict] {game_date}: {len(games)} games, run {run_id}")
    logged = 0
    for game in games:
        game_reasons = data_today.game_gate_reasons(game, game_date)
        k_park = park_k_factor(game["home_team"])
        ump_name = data_today.get_home_plate_umpire(game["game_pk"])
        k_ump = ump_k_factor(ump_name)
        for side in ("home", "away"):
            pitcher = game.get(f"{side}_pitcher")
            if not pitcher:
                continue
            opp_ids = game["away_lineup"] if side == "home" else game["home_lineup"]
            try:
                ks, outs, meta = simulate_pitcher(
                    pitcher, opp_ids, k_park=k_park, k_ump=k_ump,
                    n_sims=n_sims, game_date=game_date,
                )
            except Exception as exc:
                print(f"[predict] {pitcher['name']}: failed ({exc})")
                continue
            for market, arr in (("pitcher_strikeouts", ks), ("pitcher_outs", outs)):
                rows = lines_by_player.get((_norm(pitcher["name"]), market), [])
                logged += _log_market(
                    conn, run_id, game_date, game, pitcher, market, arr, rows,
                    quality_reasons=game_reasons + meta["quality_reasons"],
                    batters_faced=meta["batters_faced"], meta=meta,
                )
            print(
                f"[predict] {pitcher['name']} lineup={meta['lineup_status']} "
                f"workload={'verified' if meta['workload'].get('confirmed') else 'HOLD'} "
                f"K={ks.mean():.2f} outs={outs.mean():.2f} BF={meta['batters_faced'].mean():.2f}"
            )
    db.record_run(
        conn, run_id, game_date, decision.POLICY_VERSION,
        mode="shadow" if decision.SHADOW_MODE else "production",
        source="sim", board_as_of=board_as_of, line_count=len(active),
        status="complete", metadata={"n_sims": n_sims, "predictions": logged},
    )
    conn.close()
    print(f"[predict] logged {logged} immutable decision snapshots")
    return logged


def _calibrated_sides(conn, raw_p_over, market, game_date):
    p_over = calibration.calibrated_probability(
        conn, raw_p_over, market, "sim", as_of_date=game_date,
        policy_version=decision.POLICY_VERSION,
    )
    return p_over, 1.0 - p_over


def _log_market(
    conn, run_id, game_date, game, pitcher, market, arr, line_rows,
    quality_reasons=(), batters_faced=None, meta=None,
):
    meta = meta or {}
    base = {
        "run_id": run_id, "created_at": db.utcnow(), "game_date": game_date,
        "game_pk": game.get("game_pk"), "player_name": pitcher["name"],
        "player_id": pitcher.get("id"), "market": market, "source": "sim",
        "policy_version": decision.POLICY_VERSION,
        "event_start": game.get("start_time"),
        "lineup_status": meta.get("lineup_status", "unknown"),
        "starter_verified": 1 if pitcher.get("id") else 0,
        "workload_verified": 1 if (meta.get("workload") or {}).get("confirmed") else 0,
        "provenance": "model",
    }

    if not line_rows:
        mean = float(arr.mean())
        line = round(mean - 0.5) + 0.5
        raw_p, _, p_push = simulator.prob_over(arr, line)
        reasons = list(quality_reasons) + ["no live exact line; synthetic row is observability-only"]
        db.insert_prediction(conn, {
            **base, "line": line, "p_over": raw_p, "raw_p_over": raw_p,
            "decision_p_over": raw_p, "p_push": p_push, "mean": mean,
            "book": None, "over_odds": None, "under_odds": None,
            "market_p_over": None, "edge_over": None, "edge_under": None,
            "candidate_side": None, "bet_side": None, "kelly_frac": 0.0,
            "rationale": json.dumps({"rationale": "; ".join(reasons),
                                     "hold_reasons": reasons}),
            "quality_json": json.dumps({"hold_reasons": reasons}),
            "decision_status": "pass", "official": 0, "provenance": "synthetic",
            "line_verified": 0, "line_tier": "synthetic",
            "suppression_reason": "no_live_exact_line",
            "rule_trace_json": json.dumps(["exact_line:fail"]),
        })
        return 1

    by_line = {}
    for row in line_rows:
        by_line.setdefault(float(row["line"]), []).append(row)

    logged = 0
    for line, books in by_line.items():
        raw_p_over, _, p_push = simulator.prob_over(arr, line)
        p_over, p_under = _calibrated_sides(conn, raw_p_over, market, game_date)
        market_p = consensus_p_over([(b["over_odds"], b["under_odds"]) for b in books])
        priced_books = {b["book"] for b in books
                        if b["over_odds"] is not None and b["under_odds"] is not None}
        line_as_of = max((b["as_of"] for b in books if b["as_of"]), default=None)
        line_verified = bool(books) and all(bool(b["verified"]) for b in books)
        standard_tier = all((b["line_tier"] or "standard") == "standard" for b in books)
        validation = calibration.validation_status(
            conn, market, "sim", as_of_date=game_date,
            policy_version=decision.POLICY_VERSION,
        )
        reasons = list(quality_reasons)
        stress = None
        under_basis = None
        if market == "pitcher_strikeouts" and batters_faced is not None:
            stress = simulator.k_under_stress(arr, batters_faced, line)
            under_basis = stress["basis"]
            # This is no longer a blanket +4 BF veto. It is a mechanical stress
            # lane that raises uncertainty and labels workload-dependent unders.
            if p_under >= p_over and stress["plus_4"] < 0.55:
                reasons.append(
                    f"K under is workload-dependent: P(under) base {stress['base']:.1%}, "
                    f"+2 BF {stress['plus_2']:.1%}, +4 BF {stress['plus_4']:.1%}"
                )

        uncertainty = 0.0
        if meta.get("lineup_status") != "confirmed": uncertainty += 0.35
        if not (meta.get("workload") or {}).get("confirmed"): uncertainty += 0.35
        if not line_verified: uncertainty += 0.30
        uncertainty = min(1.0, uncertainty)
        call = decision.decide(
            p_over, market_p, book_count=len(priced_books), validation=validation,
            quality_reasons=reasons, allow_auto=False, market=market,
            line_verified=line_verified, standard_tier=standard_tier,
            lineup_confirmed=meta.get("lineup_status") == "confirmed",
            starter_verified=bool(pitcher.get("id")),
            workload_verified=bool((meta.get("workload") or {}).get("confirmed")),
            uncertainty=uncertainty,
        )
        candidate_side = "over" if p_over >= market_p else "under" if market_p is not None else None
        best = _best_price(books, candidate_side) or books[0]
        side_market = None
        if market_p is not None and candidate_side:
            side_market = market_p if candidate_side == "over" else 1.0 - market_p
        rationale = {
            "rationale": "; ".join(call.hold_reasons),
            "hold_reasons": list(call.hold_reasons),
            "validation": validation,
            "workload_mix": meta.get("workload_mix"),
            "stress_test": stress,
            "decision_status": call.status,
            "suppression_reason": call.suppression_reason,
        }
        db.insert_prediction(conn, {
            **base, "line": line, "p_over": p_over, "raw_p_over": raw_p_over,
            "decision_p_over": call.decision_p_over, "p_push": p_push,
            "mean": float(arr.mean()), "book": best["book"],
            "over_odds": best["over_odds"], "under_odds": best["under_odds"],
            "market_p_over": market_p, "edge_over": call.edge_over,
            "edge_under": call.edge_under, "candidate_side": candidate_side,
            "bet_side": None, "kelly_frac": 0.0,
            "rationale": json.dumps(rationale, sort_keys=True, default=str),
            "quality_json": json.dumps({"hold_reasons": list(call.hold_reasons),
                                         "validation": validation,
                                         "book_count": len(priced_books)}, sort_keys=True),
            "decision_status": call.status, "official": 0,
            "line_as_of": line_as_of, "line_verified": 1 if line_verified else 0,
            "line_tier": "standard" if standard_tier else "alternate",
            "calibration_status": "platt" if validation["sample_size"] >= decision.MIN_VALIDATION_ROWS else "raw",
            "suppression_reason": call.suppression_reason,
            "rule_trace_json": json.dumps(call.rule_trace),
            "uncertainty_score": uncertainty,
            "stress_p_under_plus2": stress.get("plus_2") if stress else None,
            "stress_p_under_plus4": stress.get("plus_4") if stress else None,
            "under_basis": under_basis,
            "entry_market_p_side": side_market,
        })
        logged += 1
    return logged


def _best_price(books, side):
    if side not in {"over", "under"}:
        return None
    col = f"{side}_odds"
    priced = [book for book in books if book[col] is not None]
    if not priced:
        return None
    from betting_math import american_to_decimal
    return max(priced, key=lambda book: american_to_decimal(book[col]))


if __name__ == "__main__":
    predict_date(sys.argv[1] if len(sys.argv) > 1 else None)
