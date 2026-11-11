# Quant ledger and SQLite import

The repository for this work is `/Users/tonylin/Sports Prediction Platform`.
`app.py` delegates to `shared/web/app.py`, where the SQLite schema is defined.

## Schema and runtime

`nba/ledger/schema.sql` now defines `quant_ledger` objects and five new tables:
users, support_tickets, saved_picks, events and prices. All five have caller-supplied
UUID primary keys. New timestamp columns use `timestamptz NOT NULL DEFAULT
clock_timestamp()`. Events/prices join the existing immutability trigger array;
users, tickets and the user's saved-pick watchlist remain mutable.

Original user/ticket/pick IDs remain in `legacy_sqlite_id`. Events retain the UUID
column name `event_id` and preserve the provider's old text ID as `source_event_id`.
Provider IDs are indexed rather than unique so later event observations can append.
Prices gain a UUID primary key, allowing repeated price observations. Their event
UUID FK uses RESTRICT, preventing cascading deletion of market history.

`saved_picks.user_id` maps to the new user UUID. Its game_id remains an external
bigint identifier: the original SQLite schema has no event FK for saved picks.

`nba/ledger/db.py` uses DATABASE_URL, with NBA_DATABASE_URL as a compatibility
fallback, and initializes schema version 2. It upgrades an existing v1 database
using `upgrade_quant_ledger.sql`, which renames the schema without copying rows and
replaces function bodies containing the old name. It refuses ambiguous databases
that contain both schema names. Fresh installation and upgrade run transactionally.
Existing reporting views follow the renamed schema through stored dependencies.

## Initialize or upgrade

Set DATABASE_URL in the repository .env or environment, then run from the root:

```bash
python -m nba.forward init-db
```

The initializer supports both a fresh install and an existing v1 ledger, and can
be rerun after v2 is installed. Do not run the fresh schema.sql against an existing
database. An alternative for an existing v1 database is:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 --single-transaction -f nba/ledger/upgrade_quant_ledger.sql
```

Use one upgrade route. The raw SQL upgrade deliberately rejects repeat application.

## Import SQLite rows

Pause source writers for cutover and retain a consistent SQLite backup. Install
the psycopg 3 dependency from nba/requirements-forward.txt. The source database is
normally shared/data/sportsaid.db; --sqlite can select the earlier root location.

```bash
python migrate_db.py --sqlite shared/data/sportsaid.db --assume-naive-utc --dry-run
python migrate_db.py --sqlite shared/data/sportsaid.db --assume-naive-utc
```

The old app uses datetime.utcnow().isoformat(); --assume-naive-utc explicitly treats
these timezone-less values as UTC. Already-aware timestamps preserve their instants.
Missing or invalid historical times (including unknown event start times) fail
validation rather than silently using the current time. Recording times use defaults.

The importer opens SQLite read-only and reads all five tables from one coherent
snapshot. It generates UUIDs, checks orphan references and numeric values before
writing, locks all five PostgreSQL destination tables, and requires them to be empty.
It inserts parents before children with executemany batches (default 1000), verifies
row counts, and commits once. Any failure rolls back the whole import. Successful
imports are intentionally one-shot; reruns fail instead of duplicating history.
Dry-run performs the inserts and checks, then rolls back; it still acquires locks.

## Verification

```bash
python nba/tests/test_db_migration.py
python scripts/run_tests.py --suite nba
```

Integration tests are opt-in via MIGRATION_TEST_DATABASE_URL pointing to an isolated
empty PostgreSQL database. They refuse existing ledger schemas and roll back all
DDL/data. They exercise fresh install and v1 upgrade, FK mapping, market mutation
guards, mutable user tables, failure rollback and dry-run behavior.

## Separate web cutover

The Flask application still uses SQLite until its connection/query layer is
refactored. It will need psycopg placeholders, UUID handling, RETURNING id for
inserts and timestamp-aware formatting. Market writers must append instead of
deleting/updating records, and readers must explicitly select latest observations.
The schema and importer alone do not switch the running web application.
