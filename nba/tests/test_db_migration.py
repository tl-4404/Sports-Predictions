"""SQLite transformation tests and opt-in PostgreSQL migration tests.

Set MIGRATION_TEST_DATABASE_URL to an isolated empty PostgreSQL database to run
integration tests. Each test rolls back its DDL/data and refuses existing schemas.
"""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from migrate_db import build_batches, parse_time, read_source, write_batches


def source_rows():
    return {
        "users": [dict(id=42, email="user@example.test", password_hash="keep-this-hash", created_at="2026-01-01T12:00:00")],
        "support_tickets": [dict(id=9, user_email="guest@example.test", subject="Question", category="other", message="Test", created_at="2026-01-01T12:00:00Z", status="open")],
        "saved_picks": [dict(id=7, user_id=42, game_id=225000123, league="NBA", market="h2h", selection="Home", odds_american=-110, stake_cents=500, notes=None, created_at="2026-01-01T12:00:00Z")],
        "events": [dict(event_id="provider-event-123", league="NBA", commence_time="2026-01-01T20:00:00-05:00", home="Home", away="Away")],
        "prices": [dict(event_id="provider-event-123", book="Book", market="h2h", outcome="Home", line=None, odds_american=-110, as_of="2026-01-01T12:00:00Z")],
    }


class TransformTests(unittest.TestCase):
    def test_uuid_mapping_and_original_values(self):
        batch = build_batches(source_rows(), True)
        for table in batch:
            self.assertIsInstance(batch[table][0][0], UUID)
        self.assertEqual(batch["saved_picks"][0][2], batch["users"][0][0])
        self.assertEqual(batch["prices"][0][1], batch["events"][0][0])
        self.assertEqual(batch["users"][0][1], 42)
        self.assertEqual(batch["users"][0][3], "keep-this-hash")
        self.assertEqual(batch["events"][0][1], "provider-event-123")
        self.assertEqual(batch["saved_picks"][0][3], 225000123)

    def test_timestamp_policy_and_preserved_instants(self):
        with self.assertRaisesRegex(ValueError, "naive timestamp"):
            build_batches(source_rows())
        self.assertEqual(parse_time("2026-01-01T20:00:00-05:00", "test"),
                         datetime(2026, 1, 2, 1, tzinfo=timezone.utc))
        for value in (None, "", "not-a-date"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_time(value, "test", True)

    def test_orphans_and_missing_start_rejected(self):
        for table, key, value in (("saved_picks", "user_id", 99), ("prices", "event_id", "missing"),
                                  ("events", "commence_time", None)):
            source = source_rows()
            source[table][0][key] = value
            with self.subTest(table=table), self.assertRaises(ValueError):
                build_batches(source, True)

    def test_invalid_numeric_values_rejected(self):
        for table, key, value in (("saved_picks", "game_id", "garbage"), ("saved_picks", "stake_cents", 1.5),
                                  ("prices", "odds_american", 2**40), ("prices", "line", float("inf"))):
            source = source_rows()
            source[table][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_batches(source, True)

    def test_read_only_sqlite_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sportsaid.db"
            rows = source_rows()
            with sqlite3.connect(path) as conn:
                for table, values in rows.items():
                    columns = list(values[0])
                    conn.execute(f"CREATE TABLE {table} ({', '.join(columns)})")
                    conn.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})",
                                     [tuple(row.values()) for row in values])
            before = path.read_bytes()
            self.assertEqual(read_source(path), rows)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(FileNotFoundError):
                read_source(path.with_name("missing.db"))
            self.assertFalse(path.with_name("missing.db").exists())


@unittest.skipUnless(os.getenv("MIGRATION_TEST_DATABASE_URL"), "No isolated PostgreSQL test DSN supplied")
class PostgreSQLTests(unittest.TestCase):
    legacy = False

    def setUp(self):
        import psycopg
        self.conn = psycopg.connect(os.environ["MIGRATION_TEST_DATABASE_URL"], autocommit=True)
        self.addCleanup(self.conn.close)
        state = self.conn.execute("SELECT to_regnamespace('quant_ledger'), to_regnamespace('nba_ledger')").fetchone()
        if state != (None, None):
            self.skipTest("Database already contains a ledger schema; refusing to touch it")
        self.enterContext(self.conn.transaction(force_rollback=True))
        schema = "nba_ledger" if self.legacy else "quant_ledger"
        self.conn.execute(f"CREATE SCHEMA {schema}")
        self.conn.execute(f"CREATE TABLE {schema}.migrations (version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT clock_timestamp())")
        schema_file = ROOT / ("nba/tests/fixtures/ledger_v1.sql" if self.legacy else "nba/ledger/schema.sql")
        self.conn.execute(schema_file.read_text())
        self.conn.execute(f"INSERT INTO {schema}.migrations(version) VALUES (1)")
        if self.legacy:
            self.conn.execute("INSERT INTO nba_ledger.runs(id,stage,source,version,config) VALUES (%s,'odds','test','v1','{}')", (uuid4(),))
            self.conn.execute((ROOT / "nba/ledger/upgrade_quant_ledger.sql").read_text())
            self.assertEqual(self.conn.execute("SELECT count(*) FROM quant_ledger.runs").fetchone(), (1,))
            self.assertIsNone(self.conn.execute("SELECT to_regnamespace('nba_ledger')").fetchone()[0])
        else:
            self.conn.execute("INSERT INTO quant_ledger.migrations(version) VALUES (2)")

    def test_import_relationships_and_rerun_guard(self):
        batch = build_batches(source_rows(), True)
        write_batches(self.conn, batch, batch_size=1)
        self.assertEqual(self.conn.execute("SELECT u.legacy_sqlite_id FROM quant_ledger.saved_picks p JOIN quant_ledger.users u ON u.id=p.user_id").fetchone(), (42,))
        self.assertEqual(self.conn.execute("SELECT e.source_event_id FROM quant_ledger.prices p JOIN quant_ledger.events e ON e.event_id=p.event_id").fetchone(), ("provider-event-123",))
        with self.assertRaisesRegex(ValueError, "not empty"):
            write_batches(self.conn, batch)

    def test_immutable_markets_and_mutable_user_tables(self):
        import psycopg
        write_batches(self.conn, build_batches(source_rows(), True))
        for table in ("events", "prices"):
            assignment = "home=home" if table == "events" else "book=book"
            for sql in (f"UPDATE quant_ledger.{table} SET {assignment}", f"DELETE FROM quant_ledger.{table}",
                        f"TRUNCATE quant_ledger.{table} CASCADE"):
                with self.subTest(sql=sql), self.assertRaises(psycopg.errors.RaiseException):
                    with self.conn.transaction():
                        self.conn.execute(sql)
        self.conn.execute("UPDATE quant_ledger.users SET email='changed@example.test'")
        self.conn.execute("UPDATE quant_ledger.support_tickets SET status='closed'")
        self.conn.execute("DELETE FROM quant_ledger.saved_picks")

    def test_failed_insert_and_dry_run_roll_back_all_rows(self):
        import psycopg
        valid = build_batches(source_rows(), True)
        invalid = deepcopy(valid)
        row = list(invalid["prices"][0])
        row[2] = None
        invalid["prices"][0] = tuple(row)
        with self.assertRaises(psycopg.errors.NotNullViolation):
            write_batches(self.conn, invalid)
        for table in valid:
            self.assertEqual(self.conn.execute(f"SELECT count(*) FROM quant_ledger.{table}").fetchone(), (0,))
        write_batches(self.conn, valid, dry_run=True)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM quant_ledger.users").fetchone(), (0,))


class PostgreSQLUpgradeTests(PostgreSQLTests):
    legacy = True


if __name__ == "__main__":
    unittest.main()
