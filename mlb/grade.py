"""Grade yesterday's prop predictions against box scores.

For every ungraded prediction on a date:
  * pull the starter's actual Ks / outs from the MLB StatsAPI box score
  * settle over/under/push (void if the pitcher didn't start)
  * compute profit at the logged Kelly stake
  * compute CLV: our bet-side fair prob minus the de-vigged consensus at the
    LAST line snapshot before the game — the honest "did we beat the close"

Run: python -m mlb.grade [YYYY-MM-DD]   (defaults to yesterday)
"""
import sys
from datetime import date, timedelta

from mlb import db
from mlb.data_fetching import game_artifacts
from shared.betting.math import american_to_decimal, consensus_p_over, devig_two_way
from mlb.data_fetching.providers import mlb_api


def _norm(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


def grade_date(game_date=None):
    game_date = game_date or (date.today() - timedelta(days=1)).isoformat()
    conn = db.connect()
    db.ensure_schema(conn)
    preds = db.ungraded_predictions(conn, game_date)

    # actual results, keyed by normalized starter name
    actuals = {}
    for g in mlb_api.get_schedule(game_date):
        try:
            for pid, res in mlb_api.get_starter_actuals(g["game_pk"]).items():
                actuals[_norm(res["name"])] = res
        except Exception as e:  # game postponed, boxscore missing, etc.
            print(f"[grade] boxscore {g['game_pk']} failed: {e}")
        try:
            artifacts = game_artifacts.fetch_game_artifacts(
                g["game_pk"], game_date, g["home_team"], g["away_team"]
            )
            db.replace_game_artifacts(conn, *artifacts)
        except Exception as e:
            # Artifact refresh must not prevent bets from being settled. Missing
            # games remain eligible for a later idempotent grading rerun.
            print(f"[grade] artifacts {g['game_pk']} failed: {e}")

    if not preds:
        print(f"[grade] nothing ungraded for {game_date}; artifacts refreshed")
        conn.close()
        return 0

    # closing consensus per (player, market, line)
    closing = {}
    for row in db.latest_lines(conn, game_date):
        key = (_norm(row["player_name"]), row["market"], row["line"])
        closing.setdefault(key, []).append((row["over_odds"], row["under_odds"]))

    graded = 0
    for p in preds:
        res = actuals.get(_norm(p["player_name"]))
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

        close_p = consensus_p_over(closing.get(
            (_norm(p["player_name"]), p["market"], p["line"]), []))
        clv = None
        if close_p is not None and p["bet_side"] in ("over", "under"):
            ours = p["p_over"] if p["bet_side"] == "over" else 1 - p["p_over"]
            close_side = close_p if p["bet_side"] == "over" else 1 - close_p
            clv = ours - close_side  # >0: we thought our side better than close did

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
