"""PrizePicks lane for the MLB pitcher-props pipeline.

PrizePicks is a DFS pick'em product, NOT a sportsbook. It posts projections and
users choose More/Less, with payout terms determined by the lineup. This lane:

  * evaluates the simulator distribution at PrizePicks' exact current line;
  * keeps standard, goblin, and demon lines distinct using ``odds_type``; and
  * automatically evaluates only STANDARD projections because goblin/demon
    payouts are lineup-specific and cannot safely use the standard break-even.

The sportsbook lane remains the sharp-consensus/CLV anchor. PrizePicks output
lands in separate append-only tables so it never pollutes two-way odds history.

Live fetching is opt-in (PRIZEPICKS_LIVE=1) and uses an undocumented endpoint.
Set PRIZEPICKS_STATE_CODE to the jurisdiction where the board will be used.
Verify the current board and payout shown in the app before submitting a lineup.

Run: python -m mlb.models.prediction.prizepicks [YYYY-MM-DD]
"""
import os
from datetime import date, datetime, timezone

try:
    from dotenv import load_dotenv
    from shared.paths import PROJECT_ROOT
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from mlb import db
from mlb.models.prediction import simulator
from mlb.models.prediction.daily import N_SIMS, simulate_pitcher
from mlb.models.prediction.factors import park_k_factor, ump_k_factor
from mlb.data_fetching.prizepicks import get_fetcher
from mlb.data_fetching.providers import mlb_api


# Standard PrizePicks power-play multipliers. Only the 2-pick rate is used for
# the automatic play gate; other entries are retained for reporting/tests.
POWER_PAYOUTS = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 37.5}
TARGET_LEGS = 2
EVALUATED_ODDS_TYPE = "standard"
PLAYABLE_MARGIN = float(os.getenv("PRIZEPICKS_PLAYABLE_MARGIN", "0.04"))
LINE_MAX_AGE_MINUTES = int(os.getenv("PRIZEPICKS_LINE_MAX_AGE_MINUTES", "30"))


def power_breakeven(n_legs, payouts=None):
    """Per-leg win probability at which an all-hit power play breaks even."""
    payouts = payouts or POWER_PAYOUTS
    return (1.0 / payouts[n_legs]) ** (1.0 / n_legs)


def power_ev(p, n_legs, payouts=None):
    """Net EV per unit stake, assuming independent equal-probability legs."""
    payouts = payouts or POWER_PAYOUTS
    return payouts[n_legs] * (p ** n_legs) - 1.0


def _ensure_column(conn, table, column, declaration):
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def ensure_schema(conn):
    # PrizePicks uses the shared simulation artifacts as well as its own tables.
    db.ensure_schema(conn)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS prizepicks_lines (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      game_date     TEXT NOT NULL,
      player_name   TEXT NOT NULL,
      player_id     INTEGER,
      market        TEXT NOT NULL,
      line          REAL NOT NULL,
      odds_type     TEXT NOT NULL DEFAULT 'standard',
      stat_type     TEXT,
      projection_id TEXT,
      as_of         TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_pp_lines_lookup
      ON prizepicks_lines(game_date, market, player_name);

    CREATE TABLE IF NOT EXISTS prizepicks_picks (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id        TEXT NOT NULL,
      created_at    TEXT NOT NULL,
      game_date     TEXT NOT NULL,
      game_pk       INTEGER,
      player_name   TEXT NOT NULL,
      player_id     INTEGER,
      market        TEXT NOT NULL,
      line          REAL NOT NULL,
      odds_type     TEXT NOT NULL DEFAULT 'standard',
      p_over        REAL NOT NULL,
      pick_side     TEXT NOT NULL,
      p_pick        REAL NOT NULL,
      breakeven     REAL NOT NULL,
      edge          REAL NOT NULL,
      ev_2          REAL,
      ev_3          REAL,
      playable      INTEGER NOT NULL,
      mean          REAL
    );
    CREATE INDEX IF NOT EXISTS idx_pp_picks_lookup
      ON prizepicks_picks(game_date, market, player_name);

    CREATE TABLE IF NOT EXISTS prizepicks_results (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      pick_id       INTEGER NOT NULL REFERENCES prizepicks_picks(id),
      graded_at     TEXT NOT NULL,
      actual        REAL,
      outcome       TEXT,
      pick_won      INTEGER
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_pp_results_pick
      ON prizepicks_results(pick_id);
    """)
    _ensure_column(conn, "prizepicks_lines", "odds_type",
                   "TEXT NOT NULL DEFAULT 'standard'")
    _ensure_column(conn, "prizepicks_picks", "odds_type",
                   "TEXT NOT NULL DEFAULT 'standard'")
    conn.commit()


def _insert_lines(conn, rows):
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO prizepicks_lines
           (game_date, player_name, player_id, market, line, odds_type,
            stat_type, projection_id, as_of)
           VALUES (:game_date, :player_name, :player_id, :market, :line,
                   :odds_type, :stat_type, :projection_id, :as_of)""",
        rows,
    )
    conn.commit()
    return len(rows)


def _latest_lines(conn, game_date, odds_type=EVALUATED_ODDS_TYPE):
    """Return one odds tier from the newest complete PrizePicks snapshot.

    Automatic selection is intentionally limited to ``standard`` projections.
    Goblin and demon rows remain stored for audit, but their lineup-specific
    payout adjustments make the standard 3x break-even invalid.
    """
    return conn.execute(
        """SELECT * FROM prizepicks_lines
           WHERE game_date = ?
             AND as_of = (
               SELECT MAX(as_of) FROM prizepicks_lines WHERE game_date = ?)
             AND odds_type = ?""",
        (game_date, game_date, odds_type)).fetchall()


def _latest_snapshot_time(conn, game_date):
    row = conn.execute(
        "SELECT MAX(as_of) AS as_of FROM prizepicks_lines WHERE game_date=?",
        (game_date,),
    ).fetchone()
    return row["as_of"] if row else None


def _snapshot_is_fresh(conn, game_date):
    """Require a recent live-board snapshot for today's real-money run."""
    if game_date != date.today().isoformat():
        return True
    stamp = _latest_snapshot_time(conn, game_date)
    if not stamp:
        return False
    try:
        observed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    age_minutes = (datetime.now(timezone.utc) - observed).total_seconds() / 60.0
    return 0 <= age_minutes <= LINE_MAX_AGE_MINUTES


def _norm(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


def refresh(game_date=None):
    """Fetch today's PrizePicks projections and append them to line history."""
    conn = db.connect()
    ensure_schema(conn)
    fetcher = get_fetcher()
    try:
        rows = fetcher.fetch(game_date)
    except Exception as exc:
        print(f"[prizepicks] refresh failed: {exc}")
        conn.close()
        return 0
    n = _insert_lines(conn, rows)
    print(f"[prizepicks] {type(fetcher).__name__}: stored {n} projection snapshots")
    conn.close()
    return n


def evaluate_date(game_date=None, n_sims=N_SIMS):
    """Score every currently posted standard PrizePicks line."""
    game_date = game_date or date.today().isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = db.connect()
    ensure_schema(conn)

    if not _snapshot_is_fresh(conn, game_date):
        print(
            f"[prizepicks] refusing to evaluate {game_date}: "
            f"no projection snapshot within {LINE_MAX_AGE_MINUTES} minutes"
        )
        conn.close()
        return 0

    lines_by_player = {}
    for row in _latest_lines(conn, game_date):
        lines_by_player.setdefault(_norm(row["player_name"]), []).append(row)
    if not lines_by_player:
        print("[prizepicks] no current standard projections — run refresh first")
        conn.close()
        return 0

    n_logged = 0
    for game in mlb_api.get_schedule(game_date):
        k_park = park_k_factor(game["home_team"])
        ump_name = mlb_api.get_home_plate_umpire(game["game_pk"])
        k_ump = ump_k_factor(ump_name)
        for side in ("home", "away"):
            pitcher = game[f"{side}_pitcher"]
            if not pitcher:
                continue
            line_rows = lines_by_player.get(_norm(pitcher["name"]))
            if not line_rows:
                continue
            opp_lineup_ids = (
                game["away_lineup"] if side == "home" else game["home_lineup"]
            )
            try:
                defense_context = db.defense_context(
                    conn, pitcher["id"], game[f"{side}_team"], game_date
                )
                k_arsenal = db.arsenal_matchup_factor(
                    conn, pitcher["id"], opp_lineup_ids, game_date
                )
                ks, outs, meta = simulate_pitcher(
                    pitcher, opp_lineup_ids, k_park=k_park, k_ump=k_ump,
                    defense_context=defense_context,
                    k_arsenal=k_arsenal,
                    n_sims=n_sims)
            except Exception as exc:
                print(f"[prizepicks] {pitcher['name']}: sim failed ({exc})")
                continue
            arrays = {"pitcher_strikeouts": ks, "pitcher_outs": outs}
            for row in line_rows:
                arr = arrays.get(row["market"])
                if arr is None:
                    continue
                n_logged += _log_pick(
                    conn, run_id, game_date, game, pitcher, row, arr,
                    allow_play=bool(meta.get("confirmed")),
                )
    print(f"[prizepicks] logged {n_logged} standard picks")
    conn.close()
    return n_logged


def _log_pick(
    conn, run_id, game_date, game, pitcher, line_row, arr, allow_play=True
):
    line = line_row["line"]
    p_over, p_under, _ = simulator.prob_over(arr, line)
    if p_over >= 0.5:
        pick_side, p_pick = "more", p_over
    else:
        pick_side, p_pick = "less", p_under
    breakeven = power_breakeven(TARGET_LEGS)
    edge = p_pick - breakeven
    playable = bool(allow_play and edge >= PLAYABLE_MARGIN)
    conn.execute(
        """INSERT INTO prizepicks_picks
           (run_id, created_at, game_date, game_pk, player_name, player_id,
            market, line, odds_type, p_over, pick_side, p_pick, breakeven, edge,
            ev_2, ev_3, playable, mean)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, db.utcnow(), game_date, game["game_pk"], pitcher["name"],
         pitcher.get("id"), line_row["market"], line, line_row["odds_type"],
         p_over, pick_side, p_pick, breakeven, edge, power_ev(p_pick, 2),
         power_ev(p_pick, 3), 1 if playable else 0, float(arr.mean())))
    conn.commit()
    return 1


def grade_date(game_date=None):
    """Grade ungraded PrizePicks picks against box-score actuals."""
    game_date = game_date or date.today().isoformat()
    conn = db.connect()
    ensure_schema(conn)
    picks = conn.execute(
        """SELECT p.* FROM prizepicks_picks p
           LEFT JOIN prizepicks_results r ON r.pick_id = p.id
           WHERE p.game_date = ? AND r.id IS NULL""",
        (game_date,)).fetchall()
    if not picks:
        print(f"[prizepicks] nothing ungraded for {game_date}")
        conn.close()
        return 0

    actuals_by_game = {}
    graded = 0
    for pick in picks:
        game_pk = pick["game_pk"]
        if game_pk not in actuals_by_game:
            try:
                actuals_by_game[game_pk] = (
                    mlb_api.get_starter_actuals(game_pk) or {}
                )
            except Exception as exc:
                print(f"[prizepicks] actuals for game {game_pk} failed ({exc})")
                actuals_by_game[game_pk] = {}
        stat = "ks" if pick["market"] == "pitcher_strikeouts" else "outs"
        box = actuals_by_game[game_pk].get(pick["player_id"])
        actual = box.get(stat) if box else None

        if actual is None:
            outcome, won = "void", None
        elif actual > pick["line"]:
            outcome, won = "more", 1 if pick["pick_side"] == "more" else 0
        elif actual < pick["line"]:
            outcome, won = "less", 1 if pick["pick_side"] == "less" else 0
        else:
            outcome, won = "push", None
        conn.execute(
            """INSERT INTO prizepicks_results
               (pick_id, graded_at, actual, outcome, pick_won)
               VALUES (?,?,?,?,?)""",
            (pick["id"], db.utcnow(), actual, outcome, won))
        conn.commit()
        graded += 1
    print(f"[prizepicks] graded {graded} picks")
    conn.close()
    return graded


def run_all(game_date=None):
    """Refresh and evaluate only when a current snapshot was stored."""
    n = refresh(game_date)
    if n <= 0:
        print("[prizepicks] refresh returned no projections; evaluation aborted")
        return 0
    return evaluate_date(game_date)


def print_board(game_date=None):
    game_date = game_date or date.today().isoformat()
    conn = db.connect()
    ensure_schema(conn)
    run = conn.execute(
        "SELECT MAX(run_id) AS run_id FROM prizepicks_picks WHERE game_date=?",
        (game_date,)).fetchone()["run_id"]
    if not run:
        print(f"[prizepicks] no picks for {game_date}")
        conn.close()
        return
    rows = conn.execute(
        """SELECT * FROM prizepicks_picks WHERE game_date=? AND run_id=?
           ORDER BY edge DESC""", (game_date, run)).fetchall()
    market_names = {
        "pitcher_strikeouts": "Strikeouts",
        "pitcher_outs": "Pitcher Outs",
    }
    breakeven = power_breakeven(TARGET_LEGS)
    print(f"\n===== PrizePicks STANDARD Board — {game_date}  "
          f"(break-even {breakeven:.1%} for a {TARGET_LEGS}-pick power) =====")
    header = (f"{'Pitcher':<15}{'Stat':<13}{'PP Line':>8}  {'Pick':<6}"
              f"{'P(pick)':>8}{'Edge':>8}  {'Play?':<6}")
    print(header + "\n" + "-" * len(header))
    for row in rows:
        flag = "PLAY" if row["playable"] else "—"
        print(f"{row['player_name']:<15}"
              f"{market_names.get(row['market'], row['market']):<13}"
              f"{row['line']:>8}  {row['pick_side'].upper():<6}"
              f"{row['p_pick']:>8.1%}{row['edge']:>+8.1%}  {flag:<6}")
    conn.close()


if __name__ == "__main__":
    import sys
    selected_date = sys.argv[1] if len(sys.argv) > 1 else None
    run_all(selected_date)
    print_board(selected_date)
