"""SQLite schema and immutable-ledger helpers for the MLB props pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

try:
    from timeutils import slate_today
except ImportError:  # pragma: no cover - standalone migration utility
    def slate_today():
        return datetime.now(timezone.utc).date().isoformat()

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "sportsaid.db"
DB_PATH = Path(os.getenv("SPORTSAID_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()
MARKETS = ("pitcher_strikeouts", "pitcher_outs")


def connect(db_path=None):
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def _columns(conn, table):
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_column(conn, table, column, declaration):
    if column not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def ensure_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS pipeline_runs (
      run_id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL,
      game_date TEXT NOT NULL,
      policy_version TEXT NOT NULL,
      mode TEXT NOT NULL,
      source TEXT,
      board_as_of TEXT,
      line_count INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'started',
      metadata_json TEXT
    );

    CREATE TABLE IF NOT EXISTS prop_lines (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      game_date TEXT NOT NULL,
      game_pk INTEGER,
      event_id TEXT,
      player_name TEXT NOT NULL,
      player_id INTEGER,
      market TEXT NOT NULL,
      book TEXT NOT NULL,
      line REAL NOT NULL,
      over_odds INTEGER,
      under_odds INTEGER,
      as_of TEXT NOT NULL,
      line_tier TEXT NOT NULL DEFAULT 'standard',
      source_type TEXT NOT NULL DEFAULT 'live',
      verified INTEGER NOT NULL DEFAULT 1,
      first_pitch TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_prop_lines_lookup
      ON prop_lines(game_date, market, player_name, as_of);

    CREATE TABLE IF NOT EXISTS prop_predictions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id TEXT NOT NULL,
      created_at TEXT NOT NULL,
      game_date TEXT NOT NULL,
      game_pk INTEGER,
      player_name TEXT NOT NULL,
      player_id INTEGER,
      market TEXT NOT NULL,
      line REAL NOT NULL,
      source TEXT NOT NULL,
      p_over REAL NOT NULL,
      raw_p_over REAL,
      decision_p_over REAL,
      p_push REAL NOT NULL DEFAULT 0,
      mean REAL,
      book TEXT,
      over_odds INTEGER,
      under_odds INTEGER,
      market_p_over REAL,
      edge_over REAL,
      edge_under REAL,
      candidate_side TEXT,
      bet_side TEXT,
      kelly_frac REAL,
      rationale TEXT,
      policy_version TEXT NOT NULL DEFAULT 'legacy',
      quality_json TEXT,
      decision_status TEXT NOT NULL DEFAULT 'pass',
      official INTEGER NOT NULL DEFAULT 0,
      selected_for_card INTEGER NOT NULL DEFAULT 0,
      provenance TEXT NOT NULL DEFAULT 'model',
      line_as_of TEXT,
      event_start TEXT,
      line_verified INTEGER NOT NULL DEFAULT 0,
      line_tier TEXT NOT NULL DEFAULT 'standard',
      lineup_status TEXT NOT NULL DEFAULT 'unknown',
      starter_verified INTEGER NOT NULL DEFAULT 0,
      workload_verified INTEGER NOT NULL DEFAULT 0,
      calibration_status TEXT NOT NULL DEFAULT 'raw',
      external_mode TEXT,
      external_projection REAL,
      external_as_of TEXT,
      suppression_reason TEXT,
      rule_trace_json TEXT,
      uncertainty_score REAL NOT NULL DEFAULT 0,
      stress_p_under_plus2 REAL,
      stress_p_under_plus4 REAL,
      under_basis TEXT,
      entry_market_p_side REAL,
      ledger_key TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_prop_pred_lookup
      ON prop_predictions(game_date, market, player_name, source, created_at);

    CREATE TABLE IF NOT EXISTS decision_ledger (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      prediction_id INTEGER NOT NULL UNIQUE REFERENCES prop_predictions(id),
      ledger_key TEXT NOT NULL UNIQUE,
      recorded_at TEXT NOT NULL,
      snapshot_hash TEXT NOT NULL,
      snapshot_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS prop_results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      prediction_id INTEGER NOT NULL UNIQUE REFERENCES prop_predictions(id),
      graded_at TEXT NOT NULL,
      actual REAL,
      outcome TEXT,
      settlement_status TEXT NOT NULL DEFAULT 'unsettled',
      refund_reason TEXT,
      bet_won INTEGER,
      profit_units REAL,
      closing_p_over REAL,
      entry_p_side REAL,
      closing_p_side REAL,
      clv REAL,
      clv_available INTEGER NOT NULL DEFAULT 0,
      roi_eligible INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS calibration_runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id TEXT NOT NULL,
      created_at TEXT NOT NULL,
      through_date TEXT NOT NULL,
      market TEXT NOT NULL,
      source TEXT NOT NULL,
      policy_version TEXT NOT NULL DEFAULT 'legacy',
      sample_size INTEGER NOT NULL,
      mode TEXT NOT NULL,
      brier_score REAL,
      log_loss REAL,
      calibration_error REAL,
      metadata_json TEXT,
      UNIQUE(run_id, market, source)
    );

    CREATE TABLE IF NOT EXISTS calibration_buckets (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id TEXT NOT NULL,
      market TEXT NOT NULL,
      source TEXT NOT NULL,
      policy_version TEXT NOT NULL DEFAULT 'legacy',
      bucket_low REAL NOT NULL,
      bucket_high REAL NOT NULL,
      sample_size INTEGER NOT NULL,
      mean_predicted REAL NOT NULL,
      actual_rate REAL NOT NULL,
      calibrated_probability REAL NOT NULL,
      UNIQUE(run_id, market, source, bucket_low)
    );

    CREATE TABLE IF NOT EXISTS calibration_models (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id TEXT NOT NULL,
      created_at TEXT NOT NULL,
      through_date TEXT NOT NULL,
      market TEXT NOT NULL,
      source TEXT NOT NULL,
      policy_version TEXT NOT NULL,
      target TEXT NOT NULL,
      sample_size INTEGER NOT NULL,
      method TEXT NOT NULL,
      intercept REAL NOT NULL,
      slope REAL NOT NULL,
      brier_score REAL,
      log_loss REAL,
      ece REAL,
      metadata_json TEXT,
      UNIQUE(run_id, market, source, target)
    );

    CREATE TABLE IF NOT EXISTS external_projection_snapshots (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      game_date TEXT NOT NULL,
      game_pk INTEGER,
      player_name TEXT NOT NULL,
      player_id INTEGER,
      market TEXT NOT NULL,
      projection REAL NOT NULL,
      provider TEXT NOT NULL,
      mode TEXT NOT NULL,
      as_of TEXT NOT NULL,
      source_url TEXT,
      metadata_json TEXT
    );
    """)

    # Safe migrations from every earlier schema version.
    line_cols = {
        "line_tier": "TEXT NOT NULL DEFAULT 'standard'",
        "source_type": "TEXT NOT NULL DEFAULT 'live'",
        "verified": "INTEGER NOT NULL DEFAULT 1",
        "first_pitch": "TEXT",
    }
    for name, declaration in line_cols.items():
        _ensure_column(conn, "prop_lines", name, declaration)

    pred_cols = {
        "raw_p_over": "REAL", "decision_p_over": "REAL",
        "p_push": "REAL NOT NULL DEFAULT 0",
        "candidate_side": "TEXT", "policy_version": "TEXT NOT NULL DEFAULT 'legacy'",
        "quality_json": "TEXT", "decision_status": "TEXT NOT NULL DEFAULT 'pass'",
        "official": "INTEGER NOT NULL DEFAULT 0",
        "selected_for_card": "INTEGER NOT NULL DEFAULT 0",
        "provenance": "TEXT NOT NULL DEFAULT 'model'", "line_as_of": "TEXT",
        "event_start": "TEXT", "line_verified": "INTEGER NOT NULL DEFAULT 0",
        "line_tier": "TEXT NOT NULL DEFAULT 'standard'",
        "lineup_status": "TEXT NOT NULL DEFAULT 'unknown'",
        "starter_verified": "INTEGER NOT NULL DEFAULT 0",
        "workload_verified": "INTEGER NOT NULL DEFAULT 0",
        "calibration_status": "TEXT NOT NULL DEFAULT 'raw'",
        "external_mode": "TEXT", "external_projection": "REAL",
        "external_as_of": "TEXT", "suppression_reason": "TEXT",
        "rule_trace_json": "TEXT", "uncertainty_score": "REAL NOT NULL DEFAULT 0",
        "stress_p_under_plus2": "REAL", "stress_p_under_plus4": "REAL",
        "under_basis": "TEXT", "entry_market_p_side": "REAL", "ledger_key": "TEXT",
    }
    for name, declaration in pred_cols.items():
        _ensure_column(conn, "prop_predictions", name, declaration)

    result_cols = {
        "settlement_status": "TEXT NOT NULL DEFAULT 'unsettled'",
        "refund_reason": "TEXT", "entry_p_side": "REAL", "closing_p_side": "REAL",
        "clv_available": "INTEGER NOT NULL DEFAULT 0",
        "roi_eligible": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, declaration in result_cols.items():
        _ensure_column(conn, "prop_results", name, declaration)
    _ensure_column(conn, "calibration_runs", "policy_version", "TEXT NOT NULL DEFAULT 'legacy'")
    _ensure_column(conn, "calibration_buckets", "policy_version", "TEXT NOT NULL DEFAULT 'legacy'")
    conn.commit()


def record_run(conn, run_id, game_date, policy_version, *, mode="shadow", source=None,
               board_as_of=None, line_count=0, status="started", metadata=None):
    conn.execute(
        """INSERT OR REPLACE INTO pipeline_runs
           (run_id, created_at, game_date, policy_version, mode, source,
            board_as_of, line_count, status, metadata_json)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (run_id, utcnow(), game_date, policy_version, mode, source, board_as_of,
         int(line_count), status, json.dumps(metadata or {}, sort_keys=True)),
    )
    conn.commit()


def insert_lines(conn, rows):
    rows = [dict(row) for row in rows]
    if not rows:
        return 0
    for row in rows:
        row.setdefault("line_tier", "standard")
        row.setdefault("source_type", "live")
        row.setdefault("verified", 1)
        row.setdefault("first_pitch", None)
    conn.executemany(
        """INSERT INTO prop_lines
           (game_date, game_pk, event_id, player_name, player_id, market,
            book, line, over_odds, under_odds, as_of, line_tier, source_type,
            verified, first_pitch)
           VALUES (:game_date,:game_pk,:event_id,:player_name,:player_id,:market,
                   :book,:line,:over_odds,:under_odds,:as_of,:line_tier,:source_type,
                   :verified,:first_pitch)""",
        rows,
    )
    conn.commit()
    return len(rows)


def _prediction_defaults(row):
    row = dict(row)
    row.setdefault("raw_p_over", row.get("p_over"))
    row.setdefault("decision_p_over", row.get("p_over"))
    row.setdefault("p_push", 0.0)
    row.setdefault("candidate_side", row.get("bet_side"))
    row.setdefault("policy_version", "legacy")
    row.setdefault("quality_json", None)
    row.setdefault("decision_status", "official" if row.get("bet_side") else "pass")
    row.setdefault("official", 1 if row.get("decision_status") == "official" else 0)
    row.setdefault("selected_for_card", 0)
    row.setdefault("provenance", "model")
    row.setdefault("line_as_of", None)
    row.setdefault("event_start", None)
    row.setdefault("line_verified", 0)
    row.setdefault("line_tier", "standard")
    row.setdefault("lineup_status", "unknown")
    row.setdefault("starter_verified", 0)
    row.setdefault("workload_verified", 0)
    row.setdefault("calibration_status", "raw")
    row.setdefault("external_mode", None)
    row.setdefault("external_projection", None)
    row.setdefault("external_as_of", None)
    row.setdefault("suppression_reason", None)
    row.setdefault("rule_trace_json", None)
    row.setdefault("uncertainty_score", 0.0)
    row.setdefault("stress_p_under_plus2", None)
    row.setdefault("stress_p_under_plus4", None)
    row.setdefault("under_basis", None)
    row.setdefault("entry_market_p_side", None)
    row.setdefault("ledger_key", None)
    return row


def insert_prediction(conn, row):
    row = _prediction_defaults(row)
    cur = conn.execute(
        """INSERT INTO prop_predictions
           (run_id,created_at,game_date,game_pk,player_name,player_id,market,line,
            source,p_over,raw_p_over,decision_p_over,p_push,mean,book,over_odds,
            under_odds,market_p_over,edge_over,edge_under,candidate_side,bet_side,
            kelly_frac,rationale,policy_version,quality_json,decision_status,official,
            selected_for_card,provenance,line_as_of,event_start,line_verified,line_tier,
            lineup_status,starter_verified,workload_verified,calibration_status,
            external_mode,external_projection,external_as_of,suppression_reason,
            rule_trace_json,uncertainty_score,stress_p_under_plus2,
            stress_p_under_plus4,under_basis,entry_market_p_side,ledger_key)
           VALUES (:run_id,:created_at,:game_date,:game_pk,:player_name,:player_id,
            :market,:line,:source,:p_over,:raw_p_over,:decision_p_over,:p_push,:mean,
            :book,:over_odds,:under_odds,:market_p_over,:edge_over,:edge_under,
            :candidate_side,:bet_side,:kelly_frac,:rationale,:policy_version,
            :quality_json,:decision_status,:official,:selected_for_card,:provenance,
            :line_as_of,:event_start,:line_verified,:line_tier,:lineup_status,
            :starter_verified,:workload_verified,:calibration_status,:external_mode,
            :external_projection,:external_as_of,:suppression_reason,:rule_trace_json,
            :uncertainty_score,:stress_p_under_plus2,:stress_p_under_plus4,
            :under_basis,:entry_market_p_side,:ledger_key)""",
        row,
    )
    prediction_id = cur.lastrowid
    ledger_key = row.get("ledger_key") or (
        f"{row['run_id']}:{prediction_id}:{row['source']}:{row['market']}:{row['line']}"
    )
    snapshot = {k: row.get(k) for k in sorted(row)}
    snapshot["prediction_id"] = prediction_id
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    conn.execute(
        """INSERT INTO decision_ledger
           (prediction_id,ledger_key,recorded_at,snapshot_hash,snapshot_json)
           VALUES (?,?,?,?,?)""",
        (prediction_id, ledger_key, utcnow(), digest, canonical),
    )
    conn.execute("UPDATE prop_predictions SET ledger_key=? WHERE id=?", (ledger_key, prediction_id))
    conn.commit()
    return prediction_id


def latest_lines(conn, game_date, market=None, as_of_before=None):
    cutoff = "AND as_of <= ?" if as_of_before else ""
    q = f"""
      SELECT * FROM prop_lines pl
      WHERE pl.game_date = ?
        AND pl.as_of = (
          SELECT MAX(as_of) FROM prop_lines
          WHERE game_date = pl.game_date AND book = pl.book {cutoff}
        )
    """
    args = [game_date]
    if as_of_before:
        args.append(as_of_before)
    if market:
        q += " AND pl.market = ?"
        args.append(market)
    return conn.execute(q, args).fetchall()


def active_lines(conn, game_date, market=None, max_age_minutes=30, now=None):
    rows = latest_lines(conn, game_date, market)
    if game_date != slate_today():
        return rows
    current = now or datetime.now(timezone.utc)
    fresh = []
    for row in rows:
        if not row["verified"] or row["source_type"] != "live":
            continue
        try:
            observed = datetime.fromisoformat(str(row["as_of"]).replace("Z", "+00:00"))
            age = (current - observed).total_seconds() / 60.0
        except (TypeError, ValueError):
            continue
        if 0 <= age <= max_age_minutes:
            fresh.append(row)
    return fresh


def closing_lines(conn, game_date, player_name, market, line, as_of_before):
    """Last two-way price per book strictly before first pitch, exact same line."""
    return conn.execute(
        """SELECT pl.* FROM prop_lines pl
           WHERE pl.game_date=? AND LOWER(pl.player_name)=LOWER(?)
             AND pl.market=? AND pl.line=? AND pl.verified=1
             AND pl.source_type='live'
             AND pl.as_of=(
               SELECT MAX(p2.as_of) FROM prop_lines p2
               WHERE p2.game_date=pl.game_date
                 AND LOWER(p2.player_name)=LOWER(pl.player_name)
                 AND p2.market=pl.market AND p2.line=pl.line
                 AND p2.book=pl.book AND p2.as_of < ?
                 AND p2.verified=1 AND p2.source_type='live'
             )""",
        (game_date, player_name, market, line, as_of_before),
    ).fetchall()


def ungraded_predictions(conn, game_date):
    return conn.execute(
        """SELECT p.* FROM prop_predictions p
           LEFT JOIN prop_results r ON r.prediction_id=p.id
           WHERE p.game_date=? AND r.id IS NULL""",
        (game_date,),
    ).fetchall()


if __name__ == "__main__":
    connection = connect()
    ensure_schema(connection)
    print(f"Schema ensured in {DB_PATH}")
