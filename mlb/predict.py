"""Run the simulator over today's slate and log prop predictions.

Flow per probable starter:
  schedule (probables + confirmed lineups)  →  pitcher/batter per-PA rates
  →  PA-level Monte Carlo  →  P(over) per posted line  →  de-vigged market
  consensus  →  edge / bet side / fractional-Kelly stake  →  prop_predictions

When no odds rows exist for a player (stub mode with no fixture), we still
log the simulated distribution against a synthetic line near the mean so the
pipeline stays observable end-to-end — those rows carry no book/edge/bet.

Run: python mlb/predict.py [YYYY-MM-DD]
"""
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import db
import data_today
import simulator
from betting_math import consensus_p_over, kelly_fraction

BASE_DIR = Path(__file__).resolve().parent
PARK_FACTORS_PATH = BASE_DIR / "cache" / "park_factors.json"

MIN_EDGE = 0.03      # minimum edge vs de-vigged consensus before we call a bet
N_SIMS = 10000

# full team name (StatsAPI) → statcast/park-factor abbreviation
TEAM_ABBR = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "OAK",
    "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "Seattle Mariners": "SEA", "San Francisco Giants": "SF",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def _norm(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


def _park_k_factor(home_team_name):
    if not PARK_FACTORS_PATH.exists():
        return 1.0
    with open(PARK_FACTORS_PATH) as f:
        factors = json.load(f)
    abbr = TEAM_ABBR.get(home_team_name, "")
    return (factors.get(abbr) or {}).get("k_factor", 1.0)


def _lineup_rates(batter_ids, pitcher_hand):
    """Per-PA rates for the 9 batters; league-average when lineup unposted."""
    if not batter_ids:
        return [dict(data_today.LEAGUE) for _ in range(9)], False
    lineup = [data_today.get_batter_rates(b, pitcher_hand)
              for b in batter_ids[:9]]
    while len(lineup) < 9:
        lineup.append(dict(data_today.LEAGUE))
    return lineup, True


def predict_date(game_date=None, n_sims=N_SIMS):
    game_date = game_date or date.today().isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = db.connect()
    db.ensure_schema(conn)

    # posted lines grouped by (norm_player, market)
    lines_by_player = {}
    for row in db.latest_lines(conn, game_date):
        key = (_norm(row["player_name"]), row["market"])
        lines_by_player.setdefault(key, []).append(row)

    games = data_today.get_schedule(game_date)
    print(f"[predict] {game_date}: {len(games)} games, run {run_id}")
    n_logged = 0

    for g in games:
        for side in ("home", "away"):
            pitcher = g[f"{side}_pitcher"]
            if not pitcher:
                continue
            opp_lineup_ids = g["away_lineup"] if side == "home" else g["home_lineup"]
            try:
                profile = data_today.get_pitcher_profile(pitcher["id"])
                lineup, confirmed = _lineup_rates(opp_lineup_ids, pitcher["hand"])
                k_park = _park_k_factor(g["home_team"])
                ks, outs = simulator.simulate_starter(
                    {"k": profile["k"], "bb": profile["bb"], "hit": profile["hit"]},
                    lineup,
                    recent_pitch_counts=profile["recent_pitch_counts"],
                    k_park=k_park,
                    n_sims=n_sims,
                )
            except Exception as e:
                print(f"[predict] {pitcher['name']}: failed ({e})")
                continue

            samples = {"pitcher_strikeouts": ks, "pitcher_outs": outs}
            for market, arr in samples.items():
                rows = lines_by_player.get((_norm(pitcher["name"]), market), [])
                n_logged += _log_market(
                    conn, run_id, game_date, g, pitcher, market, arr, rows)
            print(f"[predict] {pitcher['name']} ({side}, "
                  f"lineup={'posted' if confirmed else 'league-avg'}): "
                  f"K mean {ks.mean():.2f}, outs mean {outs.mean():.2f}")

    print(f"[predict] logged {n_logged} predictions")
    conn.close()
    return n_logged


def _log_market(conn, run_id, game_date, game, pitcher, market, arr, line_rows):
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
        # no odds available — log the distribution vs a synthetic .5 line
        mean = float(arr.mean())
        line = round(mean - 0.5) + 0.5
        p_over, _, _ = simulator.prob_over(arr, line)
        db.insert_prediction(conn, {
            **base, "line": line, "p_over": p_over, "mean": mean,
            "book": None, "over_odds": None, "under_odds": None,
            "market_p_over": None, "edge_over": None, "edge_under": None,
            "bet_side": None, "kelly_frac": 0.0,
        })
        return 1

    # group books by line, one prediction row per distinct line
    by_line = {}
    for r in line_rows:
        by_line.setdefault(r["line"], []).append(r)

    for line, books in by_line.items():
        p_over, p_under, _ = simulator.prob_over(arr, line)
        market_p = consensus_p_over(
            [(b["over_odds"], b["under_odds"]) for b in books])
        edge_over = p_over - market_p if market_p is not None else None
        edge_under = p_under - (1 - market_p) if market_p is not None else None

        bet_side, kelly, best = None, 0.0, None
        if edge_over is not None:
            if edge_over >= MIN_EDGE and edge_over >= (edge_under or 0):
                bet_side = "over"
                best = _best_price(books, "over_odds")
                kelly = kelly_fraction(p_over, best and best["over_odds"])
            elif edge_under is not None and edge_under >= MIN_EDGE:
                bet_side = "under"
                best = _best_price(books, "under_odds")
                kelly = kelly_fraction(p_under, best and best["under_odds"])
        if best is None:
            best = books[0]

        db.insert_prediction(conn, {
            **base, "line": line, "p_over": p_over, "mean": float(arr.mean()),
            "book": best["book"],
            "over_odds": best["over_odds"], "under_odds": best["under_odds"],
            "market_p_over": market_p,
            "edge_over": edge_over, "edge_under": edge_under,
            "bet_side": bet_side, "kelly_frac": kelly,
        })
        logged += 1
    return logged


def _best_price(books, side_col):
    """Book offering the best (highest-payout) american price on a side."""
    priced = [b for b in books if b[side_col] is not None]
    if not priced:
        return None
    from betting_math import american_to_decimal
    return max(priced, key=lambda b: american_to_decimal(b[side_col]))


if __name__ == "__main__":
    predict_date(sys.argv[1] if len(sys.argv) > 1 else None)
