-- Existing v1 database upgrade. Execute inside ONE transaction.
-- The updated db.initialize() invokes this file transactionally.
SELECT pg_advisory_xact_lock(734920261);
DO $$ BEGIN
    IF to_regnamespace('nba_ledger') IS NOT NULL AND to_regnamespace('quant_ledger') IS NOT NULL THEN
        RAISE EXCEPTION 'Both ledger schemas exist; resolve explicitly before upgrading';
    ELSIF to_regnamespace('nba_ledger') IS NOT NULL THEN
        ALTER SCHEMA nba_ledger RENAME TO quant_ledger;
    ELSIF to_regnamespace('quant_ledger') IS NULL THEN
        RAISE EXCEPTION 'No v1 ledger exists; use db.initialize for a fresh install';
    END IF;
END $$;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM quant_ledger.migrations WHERE version=1)
       OR EXISTS (SELECT 1 FROM quant_ledger.migrations WHERE version <> 1) THEN
        RAISE EXCEPTION 'Upgrade requires exactly schema version 1';
    END IF;
END $$;
-- Schema renaming does not rewrite PL/pgSQL string bodies. Replace them explicitly.
CREATE OR REPLACE FUNCTION quant_ledger.validate_prediction() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    s quant_ledger.market_snapshots;
    a quant_ledger.forecast_attempts;
    price numeric;
    qty numeric;
BEGIN
    NEW.created_at := clock_timestamp();
    SELECT * INTO STRICT s FROM quant_ledger.market_snapshots WHERE id = NEW.snapshot_id;
    price := CASE WHEN NEW.chosen_long THEN s.long_price ELSE s.short_price END;
    qty := CASE WHEN NEW.chosen_long THEN s.long_shares ELSE s.short_shares END;
    IF NOT s.pregame OR NEW.created_at >= s.scheduled_start
       OR s.observed_at > NEW.created_at OR s.recorded_at > NEW.created_at
       OR NEW.created_at - s.observed_at > interval '120 seconds'
       OR price IS NULL OR NEW.entry_price <> price THEN
        RAISE EXCEPTION 'Prediction requires a fresh, matching pregame entry snapshot';
    END IF;
    IF NEW.source = 'agent' THEN
        SELECT * INTO STRICT a FROM quant_ledger.forecast_attempts WHERE id = NEW.attempt_id;
        IF a.run_id <> NEW.run_id OR a.market_slug <> s.market_slug
           OR a.inputs_sha256 <> NEW.inputs_sha256 OR a.producer_model <> NEW.producer_model THEN
            RAISE EXCEPTION 'Agent prediction provenance mismatch';
        END IF;
    END IF;
    IF NEW.paper_bet AND (NEW.abstain OR NEW.fee_per_share IS NULL OR qty IS NULL
       OR qty < NEW.paper_shares OR
       (CASE WHEN NEW.chosen_long THEN NEW.long_win_probability
             ELSE 1 - NEW.long_win_probability END) <= NEW.entry_price + NEW.fee_per_share) THEN
        RAISE EXCEPTION 'Paper bet requires non-abstention, known fees, size and positive net EV';
    END IF;
    RETURN NEW;
END $$;
CREATE OR REPLACE FUNCTION quant_ledger.reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'NBA ledger is append-only; append a new observation instead'; END $$;

-- UUIDs are supplied by callers. Original SQLite IDs remain traceable.
CREATE TABLE quant_ledger.users (
    id uuid PRIMARY KEY,
    legacy_sqlite_id bigint UNIQUE,
    email text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE quant_ledger.support_tickets (
    id uuid PRIMARY KEY,
    legacy_sqlite_id bigint UNIQUE,
    user_email text NOT NULL,
    subject text NOT NULL,
    category text NOT NULL,
    message text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    status text NOT NULL DEFAULT 'open'
);
-- Mutable user watchlist. game_id was never a declared events foreign key.
CREATE TABLE quant_ledger.saved_picks (
    id uuid PRIMARY KEY,
    legacy_sqlite_id bigint UNIQUE,
    user_id uuid NOT NULL REFERENCES quant_ledger.users(id) ON DELETE CASCADE,
    game_id bigint,
    league text,
    market text NOT NULL,
    selection text NOT NULL,
    odds_american integer,
    stake_cents bigint DEFAULT 0,
    notes text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX idx_saved_picks_user ON quant_ledger.saved_picks(user_id);
-- Event versions append; source IDs are indexed but not unique.
CREATE TABLE quant_ledger.events (
    event_id uuid PRIMARY KEY,
    source_event_id text NOT NULL,
    league text NOT NULL,
    commence_time timestamptz NOT NULL DEFAULT clock_timestamp(),
    home text NOT NULL,
    away text NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX idx_events_source ON quant_ledger.events(source_event_id, recorded_at DESC);
-- UUID primary keys allow repeated price observations rather than overwrites.
CREATE TABLE quant_ledger.prices (
    id uuid PRIMARY KEY,
    event_id uuid NOT NULL REFERENCES quant_ledger.events(event_id) ON DELETE RESTRICT,
    book text NOT NULL,
    market text NOT NULL,
    outcome text NOT NULL,
    line numeric,
    odds_american integer NOT NULL,
    as_of timestamptz NOT NULL DEFAULT clock_timestamp(),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX idx_prices_event ON quant_ledger.prices(event_id, as_of DESC);
CREATE INDEX idx_prices_market ON quant_ledger.prices(market, as_of DESC);

DO $$ DECLARE t text; BEGIN
    FOREACH t IN ARRAY ARRAY['events','prices'] LOOP
        EXECUTE format('CREATE TRIGGER immutable_rows BEFORE UPDATE OR DELETE ON quant_ledger.%I FOR EACH ROW EXECUTE FUNCTION quant_ledger.reject_mutation()', t);
        EXECUTE format('CREATE TRIGGER immutable_table BEFORE TRUNCATE ON quant_ledger.%I FOR EACH STATEMENT EXECUTE FUNCTION quant_ledger.reject_mutation()', t);
    END LOOP;
END $$;
INSERT INTO quant_ledger.migrations (version) VALUES (2);
