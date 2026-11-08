"""Explicit PostgreSQL persistence. No SQLite fallback and no network on import."""
from pathlib import Path
import os
from uuid import uuid4

from nba.ledger.domain import canonical_json, content_hash, decimal_value, timestamp, utcnow


def connect(url=None):
    from dotenv import load_dotenv
    from shared.paths import PROJECT_ROOT
    load_dotenv(PROJECT_ROOT / ".env")
    url = url or os.getenv("NBA_DATABASE_URL")
    if not url:
        raise RuntimeError("Set NBA_DATABASE_URL in .env to a PostgreSQL connection string")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("Install nba/requirements-forward.txt first") from exc
    return psycopg.connect(url, autocommit=True, row_factory=dict_row, connect_timeout=10,
                           options="-c timezone=UTC")


def initialize(conn):
    """Transactional, serialized version-1 migration; never drop user objects."""
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(734920261)")
        conn.execute("CREATE SCHEMA IF NOT EXISTS nba_ledger")
        conn.execute("""CREATE TABLE IF NOT EXISTS nba_ledger.migrations
                     (version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT clock_timestamp())""")
        versions = [r["version"] for r in conn.execute("SELECT version FROM nba_ledger.migrations")]
        if any(v != 1 for v in versions):
            raise RuntimeError("Unsupported NBA ledger schema version")
        if 1 not in versions:
            conn.execute(Path(__file__).with_name("schema.sql").read_text())
            conn.execute("INSERT INTO nba_ledger.migrations (version) VALUES (1)")


def jsonb(value):
    from psycopg.types.json import Jsonb
    canonical_json(value)  # Reject NaN / unserializable evidence before sending SQL.
    return Jsonb(value)


def start_run(conn, stage, config, source="nba_forward", version="nba_forward_v1"):
    run_id = uuid4()
    conn.execute("""INSERT INTO nba_ledger.runs (id,stage,source,version,config)
                  VALUES (%s,%s,%s,%s,%s)""", (run_id, stage, source, version, jsonb(config)))
    return run_id


def log_event(conn, run_id, kind, payload):
    conn.execute("INSERT INTO nba_ledger.run_events (run_id,kind,payload) VALUES (%s,%s,%s)",
                 (run_id, kind, jsonb(payload)))


def save_snapshot(conn, run_id, packet):
    snapshot_id = uuid4()
    conn.execute("""INSERT INTO nba_ledger.market_snapshots
        (id,run_id,event_id,market_slug,scheduled_start,observed_at,pregame,long_team,short_team,
         long_price,short_price,long_shares,short_shares,raw_event,raw_market,raw_book)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", (
        snapshot_id, run_id, packet["event_id"], packet["market_slug"], timestamp(packet["scheduled_start"]),
        timestamp(packet["observed_at"]), packet["pregame"], packet["long_team"], packet["short_team"],
        packet["long_price"], packet["short_price"], packet["long_shares"], packet["short_shares"],
        jsonb(packet["raw_event"]), jsonb(packet["raw_market"]), jsonb(packet["raw_book"])))
    return snapshot_id


def save_attempt(conn, run_id, market_slug, model, prompt_version, prompt, inputs):
    attempt_id = uuid4()
    conn.execute("""INSERT INTO nba_ledger.forecast_attempts
        (id,run_id,market_slug,producer,producer_model,prompt_version,prompt_sha256,inputs_sha256,inputs)
        VALUES (%s,%s,%s,'anthropic',%s,%s,%s,%s,%s)""", (
        attempt_id, run_id, market_slug, model, prompt_version, content_hash(prompt),
        content_hash(inputs), jsonb(inputs)))
    return attempt_id


def save_prediction(conn, *, run_id, snapshot_id, packet, forecast, inputs, raw_response,
                    source, model, version, attempt_id=None, fee_per_share=None, paper_shares=1):
    """Public seam for future numeric-model forecasts; quotes never enter agent facts."""
    p = decimal_value(forecast["long_win_probability"], minimum=0, maximum=1)
    fee = None if fee_per_share is None else decimal_value(fee_per_share, minimum=0)
    shares = decimal_value(paper_shares, minimum=0)
    if shares <= 0 or type(forecast["abstain"]) is not bool:
        raise ValueError("Require positive shares and boolean abstain")
    candidates = []
    for direction, probability, key in ((True, p, "long_price"), (False, 1-p, "short_price")):
        if packet[key] is not None:
            price = decimal_value(packet[key], minimum=0, maximum=1)
            candidates.append((probability-price, direction, price))
    if not candidates:
        raise ValueError("No valid entry price")
    edge, chosen_long, entry = max(candidates)
    quantity = packet["long_shares" if chosen_long else "short_shares"]
    abstain = forecast["abstain"]
    reason = ("agent_abstained" if abstain else "fees_unknown" if fee is None
              else "nonpositive_net_ev" if edge-fee <= 0
              else "insufficient_displayed_size" if quantity is None or quantity < shares
              else "positive_net_ev_paper_only")
    prediction_id = uuid4()
    conn.execute("""INSERT INTO nba_ledger.predictions
        (id,run_id,snapshot_id,attempt_id,source,producer_model,producer_version,inputs_sha256,
         inputs,raw_response,long_win_probability,chosen_long,entry_price,fee_per_share,
         paper_shares,paper_bet,abstain,decision_reason)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", (
        prediction_id, run_id, snapshot_id, attempt_id, source, model, version, content_hash(inputs),
        jsonb(inputs), jsonb(raw_response), p, chosen_long, entry, fee, shares,
        reason == "positive_net_ev_paper_only", abstain, reason))
    return prediction_id


def save_settlement(conn, run_id, market_slug, response):
    payout = decimal_value(response["settlement"], minimum=0, maximum=1)
    if response.get("slug") != market_slug:
        raise ValueError("Settlement does not match market")
    # Repeated grading is idempotent; actual provider corrections append new rows.
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (market_slug,))
        previous = conn.execute("""SELECT long_payout FROM nba_ledger.settlements
            WHERE market_slug=%s ORDER BY observed_at DESC,id DESC LIMIT 1""", (market_slug,)).fetchone()
        if previous and previous["long_payout"] == payout:
            return False
        conn.execute("""INSERT INTO nba_ledger.settlements (run_id,market_slug,long_payout,raw_response)
                     VALUES (%s,%s,%s,%s)""", (run_id, market_slug, payout, jsonb(response)))
    return True


def tracked_markets(conn, *, unsettled_only=False):
    where = "AND NOT EXISTS (SELECT 1 FROM nba_ledger.settlements z WHERE z.market_slug=s.market_slug)" if unsettled_only else ""
    return conn.execute("""SELECT DISTINCT ON (s.market_slug) s.* FROM nba_ledger.market_snapshots s
        WHERE EXISTS (SELECT 1 FROM nba_ledger.predictions p WHERE p.snapshot_id=s.id) """ + where +
        " ORDER BY s.market_slug, s.observed_at DESC").fetchall()


def report(conn, limit=100):
    return conn.execute("""SELECT id,run_id,created_at,source,producer_model,producer_version,
        market_slug,long_team,short_team,long_win_probability,chosen_long,entry_price,scheduled_start,
        paper_bet,paper_shares,abstain,decision_reason,fee_per_share,gross_ev_per_share,net_ev_per_share,
        close_observed_at,closing_price,clv_price,clv_relative,result,result_observed_at,long_payout,
        gross_pnl_per_share,net_pnl_per_share,brier,log_loss
        FROM nba_ledger.prediction_report ORDER BY created_at DESC,id DESC LIMIT %s""", (limit,)).fetchall()
