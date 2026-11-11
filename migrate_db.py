#!/usr/bin/env python3
"""Atomically import five SQLite tables into quant_ledger using psycopg 3.

Requires DATABASE_URL and an initialized v2 ledger. Destination web/market tables
must be empty. Source is read-only; no trigger bypasses or skipped conflicts.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

TABLES = ("users", "support_tickets", "saved_picks", "events", "prices")
COLUMNS = {
    "users": ("id", "legacy_sqlite_id", "email", "password_hash", "created_at"),
    "support_tickets": ("id", "legacy_sqlite_id", "user_email", "subject", "category", "message", "created_at", "status"),
    "saved_picks": ("id", "legacy_sqlite_id", "user_id", "game_id", "league", "market", "selection", "odds_american", "stake_cents", "notes", "created_at"),
    "events": ("event_id", "source_event_id", "league", "commence_time", "home", "away"),
    "prices": ("id", "event_id", "book", "market", "outcome", "line", "odds_american", "as_of"),
}


def parse_time(value, context, assume_naive_utc=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: missing timestamp; refusing to invent history")
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context}: invalid ISO timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        if not assume_naive_utc:
            raise ValueError(f"{context}: naive timestamp; use --assume-naive-utc for known UTC values")
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def required(row, key, context):
    if row[key] is None:
        raise ValueError(f"{context}: {key} is NULL")
    return row[key]


def integer(value, context, nullable=False, bits=64):
    if value is None and nullable:
        return None
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError
        result = int(number)
    except (ArithmeticError, ValueError) as exc:
        raise ValueError(f"{context}: expected an integer") from exc
    if not -(2 ** (bits-1)) <= result < 2 ** (bits-1):
        raise ValueError(f"{context}: outside PostgreSQL integer range")
    return result


def read_source(path):
    """Read a coherent SQLite snapshot; missing paths never create empty DBs."""
    path = Path(path).resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        try:
            existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            missing = set(TABLES) - existing
            if missing:
                raise ValueError(f"Missing SQLite tables: {sorted(missing)}")
            return {table: [dict(r) for r in conn.execute(f'SELECT * FROM "{table}"')]
                    for table in TABLES}
        finally:
            conn.rollback()


def build_batches(source, assume_naive_utc=False):
    """Allocate UUIDs, map child FKs and validate values before any PG writes."""
    user_map, event_map = {}, {}
    batches = {table: [] for table in TABLES}

    def time_for(row, field, context):
        return parse_time(row[field], f"{context}.{field}", assume_naive_utc)

    for row in source["users"]:
        old = integer(row["id"], "users.id")
        if old in user_map:
            raise ValueError("Duplicate SQLite user ID")
        new = user_map[old] = uuid4()
        ctx = f"users[{old}]"
        batches["users"].append((new, old, required(row, "email", ctx),
                                  required(row, "password_hash", ctx), time_for(row, "created_at", ctx)))
    for row in source["support_tickets"]:
        old = integer(row["id"], "support_tickets.id")
        ctx = f"support_tickets[{old}]"
        batches["support_tickets"].append((uuid4(), old, *(
            required(row, c, ctx) for c in ("user_email", "subject", "category", "message")),
            time_for(row, "created_at", ctx), required(row, "status", ctx)))
    for row in source["events"]:
        old = required(row, "event_id", "events")
        if not isinstance(old, str) or old in event_map:
            raise ValueError("SQLite event IDs must be distinct non-NULL strings")
        new = event_map[old] = uuid4()
        ctx = f"events[{old}]"
        batches["events"].append((new, old, required(row, "league", ctx),
                                   time_for(row, "commence_time", ctx),
                                   required(row, "home", ctx), required(row, "away", ctx)))
    for row in source["saved_picks"]:
        old = integer(row["id"], "saved_picks.id")
        ctx = f"saved_picks[{old}]"
        user = integer(row["user_id"], f"{ctx}.user_id")
        if user not in user_map:
            raise ValueError(f"{ctx}: orphan user_id")
        batches["saved_picks"].append((uuid4(), old, user_map[user],
            integer(row["game_id"], f"{ctx}.game_id", nullable=True), row["league"],
            required(row, "market", ctx), required(row, "selection", ctx),
            integer(row["odds_american"], f"{ctx}.odds_american", nullable=True, bits=32),
            integer(row["stake_cents"], f"{ctx}.stake_cents", nullable=True), row["notes"],
            time_for(row, "created_at", ctx)))
    for row in source["prices"]:
        old = required(row, "event_id", "prices")
        if old not in event_map:
            raise ValueError("prices: orphan event_id")
        line = None if row["line"] is None else Decimal(str(row["line"]))
        if line is not None and not line.is_finite():
            raise ValueError("prices.line: expected finite numeric value")
        batches["prices"].append((uuid4(), event_map[old], *(
            required(row, c, "prices") for c in ("book", "market", "outcome")), line,
            integer(row["odds_american"], "prices.odds_american", bits=32),
            time_for(row, "as_of", "prices")))
    return batches


def write_batches(conn, batches, batch_size=1000, dry_run=False):
    """All five tables commit together. Locks prevent concurrent duplicate imports."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("SET LOCAL TIME ZONE 'UTC'")
            cur.execute("SET LOCAL lock_timeout = '10s'")
            targets = ", ".join(f"quant_ledger.{t}" for t in TABLES)
            cur.execute(f"LOCK TABLE {targets} IN ACCESS EXCLUSIVE MODE")
            for table in TABLES:
                cur.execute(f"SELECT EXISTS (SELECT 1 FROM quant_ledger.{table})")
                if cur.fetchone()[0]:
                    raise ValueError(f"Destination quant_ledger.{table} is not empty; import refused")
            for table in ("users", "support_tickets", "events", "saved_picks", "prices"):
                columns = COLUMNS[table]
                statement = (f"INSERT INTO quant_ledger.{table} ({', '.join(columns)}) "
                             f"VALUES ({', '.join(['%s'] * len(columns))})")
                rows = batches[table]
                for offset in range(0, len(rows), batch_size):
                    cur.executemany(statement, rows[offset:offset+batch_size])
                cur.execute(f"SELECT count(*) FROM quant_ledger.{table}")
                if cur.fetchone()[0] != len(rows):
                    raise RuntimeError(f"Row-count verification failed for {table}")
            if dry_run:
                import psycopg
                raise psycopg.Rollback()


def default_source():
    root = Path(__file__).resolve().parent
    candidates = (root / "sportsaid.db", root / "shared" / "data" / "sportsaid.db")
    found = [p for p in candidates if p.is_file()]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError("Two source databases found; choose one with --sqlite")
    raise FileNotFoundError("sportsaid.db not found; specify --sqlite /absolute/path/sportsaid.db")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--assume-naive-utc", action="store_true",
                        help="Interpret timezone-less source timestamps as known UTC values")
    parser.add_argument("--dry-run", action="store_true", help="Execute/verify inserts, then roll back")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass
    url = os.environ.get("DATABASE_URL")
    if not url:
        parser.error("Set DATABASE_URL to the destination PostgreSQL connection string")
    source = read_source(args.sqlite or default_source())
    batches = build_batches(source, args.assume_naive_utc)
    import psycopg
    with psycopg.connect(url, autocommit=True, connect_timeout=10) as conn:
        write_batches(conn, batches, args.batch_size, args.dry_run)
    print("Validated and rolled back" if args.dry_run else "Migration committed")
    for table in TABLES:
        print(f"{table}: {len(batches[table])} rows")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if isinstance(exc, (ValueError, FileNotFoundError, KeyError)):
            print(f"Migration failed: {exc}")
        else:
            # Never echo DSNs, hashes or arbitrary provider/DB error messages.
            print(f"Migration failed ({type(exc).__name__}); no import transaction was committed.")
        raise SystemExit(1)
