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
from datetime import datetime, timezone
from pathlib import Path
from timeutils import slate_today

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "sportsaid.db"
DB_PATH = Path(os.getenv("SPORTSAID_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()
MARKETS = ("pitcher_strikeouts", "pitcher_outs")


def connect(db_path=None):
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def _ensure_column(conn, table, column, declaration):
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


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
      raw_p_over    REAL,
      decision_p_over REAL,
      p_push        REAL NOT NULL DEFAULT 0,
      mean          REAL,
      book          TEXT,
      over_odds     INTEGER,
      under_odds    INTEGER,
      market_p_over REAL,
      edge_over     REAL,
      edge_under    REAL,
      candidate_side TEXT,
      bet_side      TEXT,
      kelly_frac    REAL,
      rationale     TEXT,
      policy_version TEXT NOT NULL DEFAULT 'legacy',
      quality_json  TEXT
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
      policy_version    TEXT NOT NULL DEFAULT 'legacy',
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
      policy_version         TEXT NOT NULL DEFAULT 'legacy',
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
    _ensure_column(conn, "prop_predictions", "raw_p_over", "REAL")
    _ensure_column(conn, "prop_predictions", "decision_p_over", "REAL")
    _ensure_column(
        conn, "prop_predictions", "p_push", "REAL NOT NULL DEFAULT 0"
    )
    _ensure_column(
        conn,
        "prop_predictions",
        "policy_version",
        "TEXT NOT NULL DEFAULT 'legacy'",
    )
    _ensure_column(conn, "prop_predictions", "quality_json", "TEXT")
    _ensure_column(conn, "prop_predictions", "candidate_side", "TEXT")
    _ensure_column(
        conn,
        "calibration_runs",
        "policy_version",
        "TEXT NOT NULL DEFAULT 'legacy'",
    )
    _ensure_column(
        conn,
        "calibration_buckets",
        "policy_version",
        "TEXT NOT NULL DEFAULT 'legacy'",
    )
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
    row = dict(row)
    row.setdefault("raw_p_over", row.get("p_over"))
    row.setdefault("decision_p_over", row.get("p_over"))
    row.setdefault("p_push", 0.0)
    row.setdefault("policy_version", "legacy")
    row.setdefault("quality_json", None)
    row.setdefault("candidate_side", row.get("bet_side"))
    cur = conn.execute(
        """INSERT INTO prop_predictions
           (run_id, created_at, game_date, game_pk, player_name, player_id,
            market, line, source, p_over, raw_p_over, decision_p_over, p_push,
            mean, book, over_odds, under_odds, market_p_over, edge_over,
            edge_under, candidate_side, bet_side, kelly_frac, rationale,
            policy_version, quality_json)
           VALUES (:run_id, :created_at, :game_date, :game_pk, :player_name,
                   :player_id, :market, :line, :source, :p_over, :raw_p_over,
                   :decision_p_over, :p_push, :mean, :book, :over_odds,
                   :under_odds, :market_p_over, :edge_over, :edge_under,
                   :candidate_side, :bet_side, :kelly_frac, :rationale,
                   :policy_version, :quality_json)""",
        row,
    )
    conn.commit()
    return cur.lastrowid


def latest_lines(conn, game_date, market=None, as_of_before=None):
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
          {cutoff}
        )
    """.format(cutoff="AND as_of <= ?" if as_of_before else "")
    args = [game_date]
    if as_of_before:
        args.append(as_of_before)
    if market:
        q += " AND pl.market = ?"
        args.append(market)
    return conn.execute(q, args).fetchall()


def active_lines(conn, game_date, market=None, max_age_minutes=30, now=None):
    """Newest complete per-book snapshots, rejecting stale live rows.

    Historical dates remain queryable for grading and audits.  On today's
    slate, every returned book snapshot must have been observed recently.
    """
    rows = latest_lines(conn, game_date, market)
    if game_date != slate_today():
        return rows
    current = now or datetime.now(timezone.utc)
    fresh = []
    for row in rows:
        try:
            observed = datetime.fromisoformat(
                str(row["as_of"]).replace("Z", "+00:00")
            )
            age = (current - observed).total_seconds() / 60.0
        except (TypeError, ValueError):
            continue
        if 0 <= age <= max_age_minutes:
            fresh.append(row)
    return fresh


def closing_lines(
    conn,
    game_date,
    player_name,
    market,
    line,
    as_of_before,
):
    """Last two-way price per book strictly before first pitch."""
    return conn.execute(
        """
        SELECT pl.*
        FROM prop_lines pl
        WHERE pl.game_date = ?
          AND LOWER(pl.player_name) = LOWER(?)
          AND pl.market = ?
          AND pl.line = ?
          AND pl.as_of = (
            SELECT MAX(p2.as_of)
            FROM prop_lines p2
            WHERE p2.game_date = pl.game_date
              AND LOWER(p2.player_name) = LOWER(pl.player_name)
              AND p2.market = pl.market
              AND p2.line = pl.line
              AND p2.book = pl.book
              AND p2.as_of <= ?
          )
        """,
        (game_date, player_name, market, line, as_of_before),
    ).fetchall()


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
