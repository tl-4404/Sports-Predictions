-- Fresh install: ledger v1 plus web/market migration v2. All instants are UTC.
-- Schema and migration tracking are created by db.initialize().
CREATE TABLE quant_ledger.runs (
    id uuid PRIMARY KEY,
    stage text NOT NULL CHECK (stage IN ('odds', 'agent', 'grade')),
    source text NOT NULL,
    version text NOT NULL,
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    config jsonb NOT NULL
);
CREATE TABLE quant_ledger.run_events (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES quant_ledger.runs(id),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    kind text NOT NULL,
    payload jsonb NOT NULL
);
CREATE TABLE quant_ledger.market_snapshots (
    id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES quant_ledger.runs(id),
    event_id text NOT NULL,
    market_slug text NOT NULL,
    scheduled_start timestamptz NOT NULL,
    observed_at timestamptz NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    pregame boolean NOT NULL,
    long_team text NOT NULL,
    short_team text NOT NULL,
    long_price numeric CHECK (long_price > 0 AND long_price < 1),
    short_price numeric CHECK (short_price > 0 AND short_price < 1),
    long_shares numeric CHECK (long_shares >= 0),
    short_shares numeric CHECK (short_shares >= 0),
    raw_event jsonb NOT NULL,
    raw_market jsonb NOT NULL,
    raw_book jsonb NOT NULL,
    CHECK (NOT pregame OR observed_at < scheduled_start)
);
CREATE INDEX market_close_lookup ON quant_ledger.market_snapshots
    (market_slug, observed_at DESC) WHERE pregame;
CREATE TABLE quant_ledger.forecast_attempts (
    id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES quant_ledger.runs(id),
    market_slug text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    producer text NOT NULL,
    producer_model text NOT NULL,
    prompt_version text NOT NULL,
    prompt_sha256 text NOT NULL,
    inputs_sha256 text NOT NULL,
    inputs jsonb NOT NULL
);
CREATE TABLE quant_ledger.predictions (
    id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES quant_ledger.runs(id),
    snapshot_id uuid NOT NULL REFERENCES quant_ledger.market_snapshots(id),
    attempt_id uuid REFERENCES quant_ledger.forecast_attempts(id),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    source text NOT NULL CHECK (source IN ('agent', 'model', 'manual')),
    producer_model text NOT NULL,
    producer_version text NOT NULL,
    inputs_sha256 text NOT NULL,
    inputs jsonb NOT NULL,
    raw_response jsonb NOT NULL,
    long_win_probability numeric NOT NULL CHECK (long_win_probability BETWEEN 0 AND 1),
    chosen_long boolean NOT NULL,
    entry_price numeric NOT NULL CHECK (entry_price > 0 AND entry_price < 1),
    fee_per_share numeric CHECK (fee_per_share >= 0),
    paper_shares numeric NOT NULL CHECK (paper_shares > 0),
    paper_bet boolean NOT NULL,
    abstain boolean NOT NULL,
    decision_reason text NOT NULL,
    UNIQUE (run_id, snapshot_id, source, producer_model, producer_version)
);
CREATE TABLE quant_ledger.settlements (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES quant_ledger.runs(id),
    market_slug text NOT NULL,
    observed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    source text NOT NULL DEFAULT 'polymarket_us_settlement',
    long_payout numeric NOT NULL CHECK (long_payout BETWEEN 0 AND 1),
    raw_response jsonb NOT NULL
);
CREATE INDEX market_result_lookup ON quant_ledger.settlements (market_slug, observed_at DESC);

-- Defence in depth: storage callers cannot rewrite or backdate an accepted forecast.
CREATE FUNCTION quant_ledger.validate_prediction() RETURNS trigger LANGUAGE plpgsql AS $$
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
CREATE TRIGGER validate_prediction BEFORE INSERT ON quant_ledger.predictions
    FOR EACH ROW EXECUTE FUNCTION quant_ledger.validate_prediction();

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

CREATE FUNCTION quant_ledger.reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'NBA ledger is append-only; append a new observation instead'; END $$;
DO $$ DECLARE t text; BEGIN
    FOREACH t IN ARRAY ARRAY['runs','run_events','market_snapshots','forecast_attempts','predictions','settlements','events','prices'] LOOP
        EXECUTE format('CREATE TRIGGER immutable_rows BEFORE UPDATE OR DELETE ON quant_ledger.%I FOR EACH ROW EXECUTE FUNCTION quant_ledger.reject_mutation()', t);
        EXECUTE format('CREATE TRIGGER immutable_table BEFORE TRUNCATE ON quant_ledger.%I FOR EACH STATEMENT EXECUTE FUNCTION quant_ledger.reject_mutation()', t);
    END LOOP;
END $$;

-- Closing-price convention: latest same-contract/same-side executable entry price
-- captured in the 15 minutes before the ORIGINAL prediction's scheduled start.
-- A changed start time invalidates the comparison. No post-start/live snapshots.
CREATE VIEW quant_ledger.prediction_report AS
SELECT p.*, s.event_id, s.market_slug, s.scheduled_start, s.long_team, s.short_team,
    s.observed_at AS entry_observed_at,
    c.observed_at AS close_observed_at, c.closing_price,
    c.closing_price - p.entry_price AS clv_price,
    c.closing_price / p.entry_price - 1 AS clv_relative,
    r.long_payout, r.observed_at AS result_observed_at,
    CASE WHEN r.long_payout = 1 THEN 'long_win'
         WHEN r.long_payout = 0 THEN 'short_win'
         WHEN r.long_payout IS NOT NULL THEN 'non_binary_settlement'
         ELSE 'pending' END AS result,
    (CASE WHEN p.chosen_long THEN p.long_win_probability ELSE 1 - p.long_win_probability END)
        - p.entry_price AS gross_ev_per_share,
    (CASE WHEN p.chosen_long THEN p.long_win_probability ELSE 1 - p.long_win_probability END)
        - p.entry_price - p.fee_per_share AS net_ev_per_share,
    (CASE WHEN p.chosen_long THEN r.long_payout ELSE 1-r.long_payout END)
        - p.entry_price AS gross_pnl_per_share,
    (CASE WHEN p.chosen_long THEN r.long_payout ELSE 1-r.long_payout END)
        - p.entry_price - p.fee_per_share AS net_pnl_per_share,
    CASE WHEN r.long_payout IN (0,1) THEN power(p.long_win_probability-r.long_payout, 2) END AS brier,
    CASE WHEN r.long_payout IN (0,1) THEN
        -(r.long_payout * ln(greatest(0.000000000000001, least(0.999999999999999,p.long_win_probability)))
          + (1-r.long_payout) * ln(greatest(0.000000000000001, least(0.999999999999999,1-p.long_win_probability))))
        END AS log_loss
FROM quant_ledger.predictions p
JOIN quant_ledger.market_snapshots s ON s.id = p.snapshot_id
LEFT JOIN LATERAL (
    SELECT q.observed_at, CASE WHEN p.chosen_long THEN q.long_price ELSE q.short_price END AS closing_price
    FROM quant_ledger.market_snapshots q
    WHERE q.market_slug = s.market_slug AND q.event_id = s.event_id AND q.pregame
      AND q.long_team = s.long_team AND q.short_team = s.short_team
      AND q.scheduled_start = s.scheduled_start
      AND q.observed_at >= p.created_at AND q.observed_at < s.scheduled_start
      AND q.recorded_at < s.scheduled_start
      AND q.observed_at >= s.scheduled_start - interval '15 minutes'
      AND clock_timestamp() >= s.scheduled_start
      AND CASE WHEN p.chosen_long THEN q.long_price ELSE q.short_price END IS NOT NULL
    ORDER BY q.observed_at DESC, q.recorded_at DESC, q.id DESC LIMIT 1
) c ON true
LEFT JOIN LATERAL (
    SELECT z.long_payout, z.observed_at FROM quant_ledger.settlements z
    WHERE z.market_slug = s.market_slug ORDER BY z.observed_at DESC, z.id DESC LIMIT 1
) r ON true;
