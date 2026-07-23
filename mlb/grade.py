"""Grade yesterday's prop predictions against box scores.

For every ungraded prediction on a date:
  * pull the starter's actual Ks / outs from the MLB StatsAPI box score
  * settle over/under/push (void if the pitcher didn't start)
  * compute profit at the logged Kelly stake
  * compute CLV: closing de-vigged probability minus entry de-vigged
    probability on the selected (or paper-tracked) side, using only the last
    snapshot before first pitch

Run: python mlb/grade.py [YYYY-MM-DD]   (defaults to yesterday)
"""
import sys
from datetime import date, timedelta

import db
import data_today
from betting_math import american_to_decimal, devig_two_way, consensus_p_over
from timeutils import slate_today


def _norm(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


def grade_date(game_date=None):
    game_date = game_date or (
        date.fromisoformat(slate_today()) - timedelta(days=1)
    ).isoformat()
    conn = db.connect()
    db.ensure_schema(conn)
    preds = db.ungraded_predictions(conn, game_date)
    if not preds:
        print(f"[grade] nothing ungraded for {game_date}")
        conn.close()
        return 0

    games = data_today.get_schedule(game_date)
    game_meta = {g["game_pk"]: g for g in games}
    actuals_by_game = {}
    failed_games = set()
    for g in games:
        try:
            actuals_by_game[g["game_pk"]] = (
                data_today.get_starter_actuals(g["game_pk"]) or {}
            )
        except Exception as e:  # game postponed, boxscore missing, etc.
            print(f"[grade] boxscore {g['game_pk']} failed: {e}")
            failed_games.add(g["game_pk"])

    graded = 0
    for p in preds:
        if p["game_pk"] in failed_games or p["game_pk"] not in actuals_by_game:
            continue
        game_actuals = actuals_by_game[p["game_pk"]]
        res = game_actuals.get(p["player_id"])
        if res is None:
            res = next(
                (
                    row
                    for row in game_actuals.values()
                    if _norm(row.get("name")) == _norm(p["player_name"])
                ),
                None,
            )
        stat = "ks" if p["market"] == "pitcher_strikeouts" else "outs"
        actual = res[stat] if res else None

        if actual is None:
            outcome, bet_won, profit = "void", None, 0.0
        elif actual > p["line"]:
            outcome = "over"
        elif actual < p["line"]:
            outcome = "under"
        else:
            outcome = "push"

        if actual is not None:
            bet_won, profit = _settle(p, outcome)

        start_time = (game_meta.get(p["game_pk"]) or {}).get("start_time")
        close_rows = (
            db.closing_lines(
                conn,
                game_date,
                p["player_name"],
                p["market"],
                p["line"],
                start_time,
            )
            if start_time
            else []
        )
        close_p = consensus_p_over(
            [(row["over_odds"], row["under_odds"]) for row in close_rows]
        )
        clv = None
        if close_p is not None and p["market_p_over"] is not None:
            raw_p = (
                p["raw_p_over"]
                if "raw_p_over" in p.keys() and p["raw_p_over"] is not None
                else p["p_over"]
            )
            paper_side = (
                p["bet_side"]
                if p["bet_side"] in ("over", "under")
                else "over" if raw_p >= p["market_p_over"] else "under"
            )
            entry_side = (
                p["market_p_over"]
                if paper_side == "over"
                else 1.0 - p["market_p_over"]
            )
            close_side = close_p if paper_side == "over" else 1.0 - close_p
            # Positive means the market moved toward the side at a worse price.
            clv = close_side - entry_side

        conn.execute(
            """INSERT INTO prop_results
               (prediction_id, graded_at, actual, outcome, bet_won,
                profit_units, closing_p_over, clv)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (p["id"], db.utcnow(), actual, outcome, bet_won, profit,
             close_p, clv),
        )
        graded += 1
    conn.commit()

    _print_summary(conn, game_date)
    conn.close()
    return graded


def _settle(pred, outcome):
    """Returns (bet_won, profit_units) at the logged kelly stake."""
    side = pred["bet_side"]
    stake = pred["kelly_frac"] or 0.0
    if side not in ("over", "under") or stake <= 0:
        return None, 0.0
    if outcome == "push":
        return None, 0.0
    won = 1 if outcome == side else 0
    odds = pred["over_odds"] if side == "over" else pred["under_odds"]
    if odds is None:
        return won, 0.0
    dec = american_to_decimal(odds)
    return won, stake * (dec - 1.0) if won else -stake


def _print_summary(conn, game_date):
    rows = conn.execute(
        """WITH ranked AS (
             SELECT p.*,
                    ROW_NUMBER() OVER (
                      PARTITION BY p.game_date, COALESCE(p.game_pk,-1),
                                   LOWER(p.player_name), p.market, p.line,
                                   p.source, COALESCE(p.policy_version,'legacy')
                      ORDER BY p.created_at DESC, p.id DESC
                    ) AS recency_rank
             FROM prop_predictions p
             WHERE p.game_date = ?
           )
           SELECT p.source, COALESCE(p.policy_version,'legacy') policy_version,
                  COUNT(*) n,
                  SUM(CASE WHEN r.bet_won IS NOT NULL THEN 1 ELSE 0 END) bets,
                  SUM(COALESCE(r.bet_won,0)) wins,
                  ROUND(SUM(COALESCE(r.profit_units,0)), 4) pnl,
                  ROUND(AVG(r.clv), 4) avg_clv
           FROM ranked p JOIN prop_results r ON r.prediction_id = p.id
           WHERE p.recency_rank = 1
           GROUP BY p.source, COALESCE(p.policy_version,'legacy')""",
        (game_date,),
    ).fetchall()
    for r in rows:
        print(
              f"[grade] {game_date} {r['source']}/{r['policy_version']}: "
              f"{r['n']} preds, "
              f"{r['bets'] or 0} bets, {r['wins'] or 0} wins, "
              f"pnl={r['pnl']}, avg CLV={r['avg_clv']}")


if __name__ == "__main__":
    grade_date(sys.argv[1] if len(sys.argv) > 1 else None)
