"""Grade immutable MLB prop decisions and compute true probability CLV.

Pushes, voids, and refunds use one canonical settlement state so they cannot be
counted both as losses and as returned stakes. Every forecast is graded for model
evaluation; ROI includes only official, integrity-passing rows.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import data_today
import db
from betting_math import american_to_decimal, consensus_p_over
from timeutils import slate_today


def _norm(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


def grade_date(game_date=None):
    game_date = game_date or (
        date.fromisoformat(slate_today()) - timedelta(days=1)
    ).isoformat()
    conn = db.connect(); db.ensure_schema(conn)
    predictions = db.ungraded_predictions(conn, game_date)
    if not predictions:
        print(f"[grade] nothing ungraded for {game_date}")
        conn.close(); return 0

    games = data_today.get_schedule(game_date)
    game_meta = {g["game_pk"]: g for g in games}
    actuals_by_game, failed_games = {}, set()
    for game in games:
        try:
            actuals_by_game[game["game_pk"]] = data_today.get_starter_actuals(game["game_pk"]) or {}
        except Exception as exc:
            print(f"[grade] boxscore {game['game_pk']} failed: {exc}")
            failed_games.add(game["game_pk"])

    graded = 0
    for pred in predictions:
        if pred["game_pk"] in failed_games or pred["game_pk"] not in actuals_by_game:
            continue
        actuals = actuals_by_game[pred["game_pk"]]
        result = actuals.get(pred["player_id"])
        if result is None:
            result = next((row for row in actuals.values()
                           if _norm(row.get("name")) == _norm(pred["player_name"])), None)
        stat = "ks" if pred["market"] == "pitcher_strikeouts" else "outs"
        actual = result.get(stat) if result else None

        if actual is None:
            outcome = "void"
        elif actual > pred["line"]:
            outcome = "over"
        elif actual < pred["line"]:
            outcome = "under"
        else:
            outcome = "push"

        paper_side = _paper_side(pred)
        settlement, refund_reason, won, profit, roi_eligible = _settle(pred, outcome, paper_side)

        start_time = (game_meta.get(pred["game_pk"]) or {}).get("start_time") or pred["event_start"]
        close_rows = []
        if start_time:
            close_rows = db.closing_lines(
                conn, game_date, pred["player_name"], pred["market"],
                pred["line"], start_time,
            )
        close_p_over = consensus_p_over(
            [(row["over_odds"], row["under_odds"]) for row in close_rows]
        )
        entry_side = None
        if pred["market_p_over"] is not None and paper_side:
            entry_side = float(pred["market_p_over"])
            if paper_side == "under":
                entry_side = 1.0 - entry_side
        close_side = None
        if close_p_over is not None and paper_side:
            close_side = close_p_over if paper_side == "over" else 1.0 - close_p_over
        clv = close_side - entry_side if close_side is not None and entry_side is not None else None
        clv_available = int(clv is not None)

        conn.execute(
            """INSERT INTO prop_results
               (prediction_id,graded_at,actual,outcome,settlement_status,refund_reason,
                bet_won,profit_units,closing_p_over,entry_p_side,closing_p_side,clv,
                clv_available,roi_eligible)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pred["id"], db.utcnow(), actual, outcome, settlement, refund_reason,
             won, profit, close_p_over, entry_side, close_side, clv,
             clv_available, roi_eligible),
        )
        graded += 1
    conn.commit()
    _print_summary(conn, game_date)
    conn.close()
    return graded


def _paper_side(pred):
    if pred["bet_side"] in {"over", "under"}:
        return pred["bet_side"]
    if pred["candidate_side"] in {"over", "under"}:
        return pred["candidate_side"]
    if pred["market_p_over"] is None:
        return None
    raw = pred["raw_p_over"] if pred["raw_p_over"] is not None else pred["p_over"]
    return "over" if float(raw) >= float(pred["market_p_over"]) else "under"


def _settle(pred, outcome, paper_side):
    official = bool(pred["official"]) and pred["decision_status"] == "official"
    stake = float(pred["kelly_frac"] or 0.0)
    if outcome == "void":
        return "refund", "pitcher_did_not_start_or_no_official_stat", None, 0.0, 0
    if outcome == "push":
        return "refund", "whole_number_push", None, 0.0, 0
    if not official or pred["bet_side"] not in {"over", "under"} or stake <= 0:
        return "graded", None, None, 0.0, 0

    won = 1 if outcome == pred["bet_side"] else 0
    odds = pred["over_odds"] if pred["bet_side"] == "over" else pred["under_odds"]
    if odds is None:
        return "graded", "official_row_missing_price", won, 0.0, 0
    decimal = american_to_decimal(odds)
    profit = stake * (decimal - 1.0) if won else -stake
    return ("win" if won else "loss"), None, won, profit, 1


def _print_summary(conn, game_date):
    rows = conn.execute(
        """WITH first_snapshot AS (
             SELECT p.*,
                    ROW_NUMBER() OVER (
                      PARTITION BY p.game_date,COALESCE(p.game_pk,-1),
                                   LOWER(p.player_name),p.market,p.source,
                                   COALESCE(p.policy_version,'legacy')
                      ORDER BY p.created_at ASC,p.id ASC
                    ) first_rank
             FROM prop_predictions p WHERE p.game_date=?
           )
           SELECT p.source,p.policy_version,COUNT(*) forecasts,
                  SUM(CASE WHEN r.roi_eligible=1 THEN 1 ELSE 0 END) official_bets,
                  SUM(CASE WHEN r.settlement_status='win' THEN 1 ELSE 0 END) wins,
                  SUM(CASE WHEN r.settlement_status='refund' THEN 1 ELSE 0 END) refunds,
                  ROUND(SUM(COALESCE(r.profit_units,0)),4) pnl,
                  ROUND(AVG(CASE WHEN r.clv_available=1 THEN r.clv END),4) avg_clv
           FROM first_snapshot p JOIN prop_results r ON r.prediction_id=p.id
           WHERE p.first_rank=1 GROUP BY p.source,p.policy_version""",
        (game_date,),
    ).fetchall()
    for row in rows:
        print(
            f"[grade] {game_date} {row['source']}/{row['policy_version']}: "
            f"{row['forecasts']} forecasts, {row['official_bets'] or 0} official, "
            f"{row['wins'] or 0} wins, {row['refunds'] or 0} refunds, "
            f"pnl={row['pnl']}, avg true CLV={row['avg_clv']}"
        )


if __name__ == "__main__":
    grade_date(sys.argv[1] if len(sys.argv) > 1 else None)
