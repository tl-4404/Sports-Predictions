"""SQLite schema + helpers for the MLB props pipeline.

Core tables are append-only:
  prop_lines       — every odds snapshot we ever fetch (line history, per book)
  prop_predictions — every probability any pipeline (sim / llm / judge) produces
  prop_results     — graded outcomes, filled in the morning after
  calibration_runs — immutable fitted calibration snapshots
  calibration_buckets — empirical probability mapping for each snapshot

Everything lives in the same sportsaid.db the Flask app reads.
"""
import os
import sqlite3
import integrity
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "sportsaid.db"
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
      game_date     TEXT NOT NULL,
      game_pk       INTEGER,
      event_id      TEXT,
      player_name   TEXT NOT NULL,
      player_id     INTEGER,
      market        TEXT NOT NULL,
      book          TEXT NOT NULL,
      line          REAL NOT NULL,
      over_odds     INTEGER,
      under_odds    INTEGER,
      as_of         TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_prop_lines_lookup
      ON prop_lines(game_date, market, player_name);

    CREATE TABLE IF NOT EXISTS prop_predictions (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id        TEXT NOT NULL,
      created_at    TEXT NOT NULL,
      game_date     TEXT NOT NULL,
      game_pk       INTEGER,
      player_name   TEXT NOT NULL,
      player_id     INTEGER,
      market        TEXT NOT NULL,
      line          REAL NOT NULL,
      source        TEXT NOT NULL,
      p_over        REAL NOT NULL,
      mean          REAL,
      book          TEXT,
      over_odds     INTEGER,
      under_odds    INTEGER,
      market_p_over REAL,
      edge_over     REAL,
      edge_under    REAL,
      bet_side      TEXT,
      kelly_frac    REAL,
      rationale     TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_prop_pred_lookup
      ON prop_predictions(game_date, market, player_name, source);

    CREATE TABLE IF NOT EXISTS prop_results (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      prediction_id INTEGER NOT NULL REFERENCES prop_predictions(id),
      graded_at     TEXT NOT NULL,
      actual        REAL,
      outcome       TEXT,
      bet_won       INTEGER,
      profit_units  REAL,
      closing_p_over REAL,
      clv           REAL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_prop_results_pred
      ON prop_results(prediction_id);

    CREATE TABLE IF NOT EXISTS calibration_runs (
      id                INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id            TEXT NOT NULL,
      created_at        TEXT NOT NULL,
      through_date      TEXT NOT NULL,
      market            TEXT NOT NULL,
      source            TEXT NOT NULL,
      sample_size       INTEGER NOT NULL,
      mode              TEXT NOT NULL,
      brier_score       REAL,
      log_loss          REAL,
      calibration_error REAL,
      metadata_json     TEXT,
      UNIQUE(run_id, market, source)
    );
    CREATE INDEX IF NOT EXISTS idx_calibration_runs_lookup
      ON calibration_runs(market, source, through_date, created_at);

    CREATE TABLE IF NOT EXISTS calibration_buckets (
      id                     INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id                 TEXT NOT NULL,
      market                 TEXT NOT NULL,
      source                 TEXT NOT NULL,
      bucket_low             REAL NOT NULL,
      bucket_high            REAL NOT NULL,
      sample_size            INTEGER NOT NULL,
      mean_predicted         REAL NOT NULL,
      actual_rate            REAL NOT NULL,
      calibrated_probability REAL NOT NULL,
      UNIQUE(run_id, market, source, bucket_low)
    );
    CREATE INDEX IF NOT EXISTS idx_calibration_buckets_lookup
      ON calibration_buckets(run_id, market, source, bucket_low);
    """)
    # Additive migration: old forecasts remain identifiable and cannot silently
    # become training data for a different model version.
    extra = {
        "raw_p_over": "REAL", "p_push": "REAL",
        "model_version": "TEXT", "model_status": "TEXT",
        "game_start_time": "TEXT", "line_as_of": "TEXT",
        "lineup_confirmed": "INTEGER", "role_verified": "INTEGER",
        "eligibility_reason": "TEXT", "calibration_run_id": "TEXT",
        "stress_p_over": "REAL",
        "price_verified": "INTEGER",
        "role_evidence_json": "TEXT",
    }
    present = {r[1] for r in conn.execute("PRAGMA table_info(prop_predictions)")}
    for column, declaration in extra.items():
        if column not in present:
            conn.execute(f"ALTER TABLE prop_predictions ADD COLUMN {column} {declaration}")
    present_lines = {r[1] for r in conn.execute("PRAGMA table_info(prop_lines)")}
    for column, declaration in (("data_source", "TEXT"), ("is_live", "INTEGER NOT NULL DEFAULT 0")):
        if column not in present_lines:
            conn.execute(f"ALTER TABLE prop_lines ADD COLUMN {column} {declaration}")
    for table in ("prop_lines", "prop_predictions", "prop_results",
                  "calibration_runs", "calibration_buckets"):
        for operation in ("UPDATE", "DELETE"):
            conn.execute(f"""CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation}
                BEFORE {operation} ON {table}
                BEGIN SELECT RAISE(ABORT, 'append-only audit ledger'); END""")
    conn.commit()


def insert_lines(conn, rows):
    """rows: iterable of dicts matching prop_lines columns (minus id)."""
    rows = list(rows)
    if not rows:
        return 0
    rows = [{**row, "data_source": row.get("data_source"),
             "is_live": int(bool(row.get("is_live")))} for row in rows]
    conn.executemany(
        """INSERT INTO prop_lines
           (game_date, game_pk, event_id, player_name, player_id, market,
            book, line, over_odds, under_odds, as_of, data_source, is_live)
           VALUES (:game_date, :game_pk, :event_id, :player_name, :player_id,
                   :market, :book, :line, :over_odds, :under_odds, :as_of,
                   :data_source, :is_live)""",
        rows,
    )
    conn.commit()
    return len(rows)


def insert_prediction(conn, row):
    row = integrity.enforce(row)
    extras = ("raw_p_over", "p_push", "model_version", "model_status",
              "game_start_time", "line_as_of", "lineup_confirmed",
              "role_verified", "eligibility_reason", "calibration_run_id")
    extras += ("stress_p_over", "price_verified", "role_evidence_json")
    for name in extras:
        row.setdefault(name, None)
    cur = conn.execute(
        """INSERT INTO prop_predictions
           (run_id, created_at, game_date, game_pk, player_name, player_id,
            market, line, source, p_over, mean, book, over_odds, under_odds,
            market_p_over, edge_over, edge_under, bet_side, kelly_frac, rationale,
            raw_p_over, p_push, model_version, model_status, game_start_time,
            line_as_of, lineup_confirmed, role_verified, eligibility_reason,
            calibration_run_id, stress_p_over, price_verified, role_evidence_json)
           VALUES (:run_id, :created_at, :game_date, :game_pk, :player_name,
                   :player_id, :market, :line, :source, :p_over, :mean, :book,
                   :over_odds, :under_odds, :market_p_over, :edge_over,
                   :edge_under, :bet_side, :kelly_frac, :rationale,
                   :raw_p_over, :p_push, :model_version, :model_status,
                   :game_start_time, :line_as_of, :lineup_confirmed,
                   :role_verified, :eligibility_reason, :calibration_run_id, :stress_p_over,
                   :price_verified, :role_evidence_json)""",
        row,
    )
    conn.commit()
    return cur.lastrowid


def latest_lines(conn, game_date, market=None):
    """Return rows from each book's newest complete board snapshot.

    Every fetcher stamps all rows from one book with the same ``as_of`` value.
    Selecting the latest timestamp per book drops moved thresholds and players
    whose markets disappeared, while preserving all ladders offered in that
    current snapshot. Partitioning by player or numeric line would retain stale
    rows indefinitely.
    """
    q = """
      SELECT * FROM prop_lines pl
      WHERE pl.game_date = ?
        AND pl.as_of = (
          SELECT MAX(as_of) FROM prop_lines
          WHERE game_date = pl.game_date AND book = pl.book
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
