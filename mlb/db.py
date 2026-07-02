"""SQLite schema + helpers for the MLB props pipeline.

Three tables, all append-only:
  prop_lines       — every odds snapshot we ever fetch (line history, per book)
  prop_predictions — every probability any pipeline (sim / llm / judge) produces
  prop_results     — graded outcomes, filled in the morning after

Everything lives in the same sportsaid.db the Flask app reads.
"""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "sportsaid.db"

# markets we predict; keys match The Odds API market names
MARKETS = ("pitcher_strikeouts", "pitcher_outs")


def connect(db_path=None):
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def ensure_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS prop_lines (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      game_date     TEXT NOT NULL,          -- YYYY-MM-DD (local game date)
      game_pk       INTEGER,                -- MLB StatsAPI gamePk when known
      event_id      TEXT,                   -- odds provider event id
      player_name   TEXT NOT NULL,
      player_id     INTEGER,                -- MLBAM id when resolved
      market        TEXT NOT NULL,          -- pitcher_strikeouts | pitcher_outs
      book          TEXT NOT NULL,
      line          REAL NOT NULL,
      over_odds     INTEGER,                -- american
      under_odds    INTEGER,
      as_of         TEXT NOT NULL           -- snapshot time, append-only
    );
    CREATE INDEX IF NOT EXISTS idx_prop_lines_lookup
      ON prop_lines(game_date, market, player_name);

    CREATE TABLE IF NOT EXISTS prop_predictions (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id        TEXT NOT NULL,          -- one pipeline invocation
      created_at    TEXT NOT NULL,
      game_date     TEXT NOT NULL,
      game_pk       INTEGER,
      player_name   TEXT NOT NULL,
      player_id     INTEGER,
      market        TEXT NOT NULL,
      line          REAL NOT NULL,
      source        TEXT NOT NULL,          -- 'sim' | 'llm' | 'judge'
      p_over        REAL NOT NULL,          -- our probability the OVER hits
      mean          REAL,                   -- distribution mean (Ks or outs)
      book          TEXT,                   -- book whose price we evaluated
      over_odds     INTEGER,
      under_odds    INTEGER,
      market_p_over REAL,                   -- de-vigged market prob at bet time
      edge_over     REAL,                   -- p_over - market_p_over
      edge_under    REAL,
      bet_side      TEXT,                   -- 'over' | 'under' | NULL (no bet)
      kelly_frac    REAL,                   -- fractional-kelly suggested stake (of bankroll)
      rationale     TEXT                    -- llm/judge reasoning, NULL for sim
    );
    CREATE INDEX IF NOT EXISTS idx_prop_pred_lookup
      ON prop_predictions(game_date, market, player_name, source);

    CREATE TABLE IF NOT EXISTS prop_results (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      prediction_id INTEGER NOT NULL REFERENCES prop_predictions(id),
      graded_at     TEXT NOT NULL,
      actual        REAL,                   -- actual Ks or outs; NULL = void (no start)
      outcome       TEXT,                   -- 'over' | 'under' | 'push' | 'void'
      bet_won       INTEGER,                -- 1/0/NULL (NULL when no bet or push/void)
      profit_units  REAL,                   -- at the logged kelly stake, in bankroll units
      closing_p_over REAL,                  -- de-vigged consensus at last snapshot before pitch
      clv           REAL                    -- (our bet-side prob edge vs close); >0 = beat the close
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_prop_results_pred
      ON prop_results(prediction_id);
    """)
    conn.commit()


def insert_lines(conn, rows):
    """rows: iterable of dicts matching prop_lines columns (minus id)."""
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO prop_lines
           (game_date, game_pk, event_id, player_name, player_id, market,
            book, line, over_odds, under_odds, as_of)
           VALUES (:game_date, :game_pk, :event_id, :player_name, :player_id,
                   :market, :book, :line, :over_odds, :under_odds, :as_of)""",
        rows,
    )
    conn.commit()
    return len(rows)


def insert_prediction(conn, row):
    cur = conn.execute(
        """INSERT INTO prop_predictions
           (run_id, created_at, game_date, game_pk, player_name, player_id,
            market, line, source, p_over, mean, book, over_odds, under_odds,
            market_p_over, edge_over, edge_under, bet_side, kelly_frac, rationale)
           VALUES (:run_id, :created_at, :game_date, :game_pk, :player_name,
                   :player_id, :market, :line, :source, :p_over, :mean, :book,
                   :over_odds, :under_odds, :market_p_over, :edge_over,
                   :edge_under, :bet_side, :kelly_frac, :rationale)""",
        row,
    )
    conn.commit()
    return cur.lastrowid


def latest_lines(conn, game_date, market=None):
    """Most recent snapshot per (player, market, book, line) for a date."""
    q = """
      SELECT * FROM prop_lines pl
      WHERE pl.game_date = ?
        AND pl.as_of = (
          SELECT MAX(as_of) FROM prop_lines
          WHERE game_date = pl.game_date AND player_name = pl.player_name
            AND market = pl.market AND book = pl.book AND line = pl.line
        )
    """
    args = [game_date]
    if market:
        q += " AND pl.market = ?"
        args.append(market)
    return conn.execute(q, args).fetchall()


def ungraded_predictions(conn, game_date):
    return conn.execute(
        """SELECT p.* FROM prop_predictions p
           LEFT JOIN prop_results r ON r.prediction_id = p.id
           WHERE p.game_date = ? AND r.id IS NULL""",
        (game_date,),
    ).fetchall()


if __name__ == "__main__":
    conn = connect()
    ensure_schema(conn)
    print(f"Schema ensured in {DB_PATH}")
