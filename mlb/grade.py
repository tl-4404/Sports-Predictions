"""Grade yesterday's prop predictions against box scores.

For every ungraded prediction on a date:
  * pull the starter's actual Ks / outs from the MLB StatsAPI box score
  * settle over/under/push (void if the pitcher didn't start)
  * compute profit at the logged Kelly stake
  * compute same-contract fair probability movement from entry to pregame close

Run: python mlb/grade.py [YYYY-MM-DD]   (defaults to yesterday)
"""
import sys
from datetime import date, timedelta

import db
import data_today
from betting_math import american_to_decimal, consensus_p_over, probability_clv


def _norm(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


def grade_date(game_date=None):
    game_date = game_date or (date.today() - timedelta(days=1)).isoformat()
    conn = db.connect()
    db.ensure_schema(conn)
    preds = db.ungraded_predictions(conn, game_date)
    if not preds:
        print(f"[grade] nothing ungraded for {game_date}")
        return 0

    # Missing/nonfinal/API-failed results remain pending and can be retried.
    actuals = {}
    games = {g["game_pk"]: g for g in data_today.get_schedule(game_date)}
    for g in games.values():
        if g.get("status") != "Final":
            continue
        try:
            result = data_today.get_starter_actuals(g["game_pk"])
            if len(result) == 2:
                actuals[g["game_pk"]] = result
        except Exception as e:  # game postponed, boxscore missing, etc.
            print(f"[grade] boxscore {g['game_pk']} failed: {e}")

    graded = 0
    for p in preds:
        game = games.get(p["game_pk"])
        if p["game_pk"] not in actuals or not p["player_id"]:
            continue
        res = actuals[p["game_pk"]].get(p["player_id"])
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

        close_p = closing_probability(conn, p, game.get("game_start_time"))
        clv = probability_clv(p["market_p_over"], close_p, p["bet_side"])

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


def closing_probability(conn, pred, start_time):
    """Last pre-start complete board per book; never substitute a moved line."""
    if not start_time:
        return None
    rows = conn.execute(
        """SELECT pl.over_odds, pl.under_odds FROM prop_lines pl
           WHERE pl.game_date=? AND pl.game_pk=? AND pl.player_id=?
             AND pl.market=? AND pl.line=?
             AND pl.is_live=1
             AND julianday(?) - julianday(pl.as_of) <= 15.0 / (24 * 60)
             AND julianday(pl.as_of) = (
               SELECT MAX(julianday(q.as_of)) FROM prop_lines q
               WHERE q.game_date=pl.game_date AND q.book=pl.book
                 AND julianday(q.as_of) < julianday(?))""",
        (pred["game_date"], pred["game_pk"], pred["player_id"],
         pred["market"], pred["line"], start_time, start_time),
    ).fetchall()
    return consensus_p_over([(r["over_odds"], r["under_odds"]) for r in rows])


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
        """SELECT p.source, COUNT(*) n,
                  SUM(CASE WHEN r.bet_won IS NOT NULL THEN 1 ELSE 0 END) bets,
                  SUM(COALESCE(r.bet_won,0)) wins,
                  ROUND(SUM(COALESCE(r.profit_units,0)), 4) pnl,
                  ROUND(AVG(r.clv), 4) avg_clv
           FROM prop_predictions p JOIN prop_results r ON r.prediction_id = p.id
           WHERE p.game_date = ?
           GROUP BY p.source""",
        (game_date,),
    ).fetchall()
    for r in rows:
        print(f"[grade] {game_date} {r['source']}: {r['n']} preds, "
              f"{r['bets'] or 0} bets, {r['wins'] or 0} wins, "
              f"pnl={r['pnl']}, avg CLV={r['avg_clv']}")


if __name__ == "__main__":
    grade_date(sys.argv[1] if len(sys.argv) > 1 else None)
