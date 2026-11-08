# Sports Prediction Platform

Sports Prediction Platform is a quantitative sports analytics and execution framework designed for predictive prediction markets and exchanges (such as Polymarket's Central Limit Order Book) and sportsbooks.

The system pairs bottom-up probabilistic simulations (MLB plate-appearance simulator, NBA possession-level Monte Carlo) with top-down market de-vigging, independent qualitative audits, bounded risk judges, and an immutable PostgreSQL paper-trading ledger tracking Closing Line Value (CLV).

The project is an analytical aid, not a guarantee of outcomes. Its primary engineering goal is to generate timestamped, auditable forecasts evaluated against efficient market closes rather than short-term outcome variance.

---

## System Architecture

The platform separates deterministic simulation math, event-driven qualitative parsing, and execution order management into distinct components:

1. **Simulation & Quantitative Modeling:**
   - **MLB Pitcher Props:** PA-level state simulation modeling pitcher strikeout distributions, pitch mix, batter matchups, umpire tendencies, and park factors[cite: 1, 2].
   - **NBA Possession/Rotation Engine:** Lineup-stint Monte Carlo tracking possession state transitions, offensive/defensive ratings, garbage-time substitutions, and endgame intentional foul mechanics[cite: 1, 2].
2. **Top-Down De-vigging & Market Priors:**
   - Multiplicative, Power, and **Shin de-vigging** routines strip overround and favorite-longshot bias from sharp market consensus (Pinnacle/Circa) to establish true baseline probabilities[cite: 1, 2].
3. **Independent Qualitative Audit:**
   - Event-driven LLM lanes audit breaking news, injury scratches, weather, and bullpen usage without visibility into simulation forecasts or market prices, avoiding continuous polling costs[cite: 1, 2].
4. **Bounded Judge & Risk Gates:**
   - Reconciles bottom-up model projections with de-vigged market lines, enforcing strict edge thresholds, stress tests (e.g., `+4 batters faced`), and correlated market limits before outputting actionable orders.
5. **Execution & Ledger:**
   - Focuses on resting limit orders (passive maker strategy) on prediction market CLOBs[cite: 1, 2].
   - Maintains an append-only PostgreSQL ledger with database-level mutation rejection triggers to record entry price, fair market value, settled PnL, and Closing Line Value (CLV)[cite: 1, 2].

---

## Repository Organization

```text
nba/                        NBA module
  data/                     Cached datasets, generated outputs, fixture space
  data_fetching/            Dataset builders, NBA Stats API, and odds adapters
  models/                   Prediction and vectorized simulation logic
  saved_models/             Trained weights, scalers, and feature lists
  tests/                    NBA regression and forward-testing suites
  ledger/                   PostgreSQL append-only ledger schema and migration scripts
  docs/                     NBA backtesting, data, and evaluation docs
  deployment/               NBA PostgreSQL Compose configuration
mlb/                        MLB module
  data/                     Cache, fixtures, outputs, and audit ledgers
  data_fetching/            Statcast, StatsAPI, and odds adapters
  models/                   Plate-appearance simulation and training code
  saved_models/             Serialized trained model artifacts
  tests/                    MLB pipeline, grading, data, and integration suites
  docs/                     MLB pipeline architecture and design
  pipeline.py               MLB orchestration entrypoint
shared/                     Cross-League Components
  data_fetching/            Polymarket CLOB client, sportsbook feeds, PrizePicks
  betting/                  Odds math, de-vigging (Multiplicative, Power, Shin), Kelly sizing
  ledger/                   Shared database schemas, CLV calculation, and grading logic
  web/                      Web UI, API routes, templates, and dashboard monitors
  tests/                    Shared client, math, and architecture tests
scripts/                    Developer utilities, backtest runners, and test orchestration
docs/                       System architecture, market microstructure, and directory guides
app.py                      Web dashboard and API server
run_pipeline.py             CLI pipeline runner
