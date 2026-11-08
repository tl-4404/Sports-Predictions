# MLB Pitcher-Props Pipeline

## Purpose

The MLB pipeline produces auditable pitcher-prop forecasts for strikeouts, pitching
outs, fantasy score and related markets. The system is designed to improve through
clean historical evaluation rather than by repeatedly changing rules after individual
wins or losses.

Its core goals are:

- use exact, timestamped live lines;
- maintain independent numerical and qualitative model lanes;
- incorporate current lineup, role and workload information;
- avoid overconfidence and correlated exposure;
- preserve predictions before results are known;
- evaluate calibration, closing-line value and unique-pitcher accuracy.

## System overview

```text
Live odds / pick'em lines
        |
        v
Append-only line snapshots -----> Market consensus / de-vigging
        |
        +-------------------------------+
        |                               |
        v                               v
Numerical simulator             Independent LLM audit
(rates + distributions)         (current research + judgment)
        |                               |
        +---------------+---------------+
                        v
                  Bounded judge
            (risk gates + one primary bet)
                        |
                        v
             Stored pregame prediction
                        |
                        v
              Postgame grading + CLV
                        |
                        v
           Chronological calibration run
```

## Major modules

### `mlb/data_fetching/providers/mlb_api.py`

Retrieves schedule, probable pitchers, pitcher profiles, batter rates, lineups and
other current MLB inputs used by the model.

### `mlb/data_fetching/providers/odds.py`

Captures sportsbook pitcher-prop lines and prices. Every book snapshot receives a
shared timestamp so moved or removed lines can be distinguished from the current
board.

### `mlb/models/prediction/daily.py`

Runs the numerical simulator. This lane should remain independent of the LLM audit.
Its purpose is to convert stable pitcher skill, expected batters faced and game context
into probability distributions.

### `mlb/models/prediction/llm_saved_pipeline.py`

Defines the current chat-aligned qualitative research and bounded-judge policy. Policy
changes are versioned because prompt behavior affects historical comparability.

### `mlb/models/prediction/llm_chat_runner.py`

Refreshes current research on every invocation, then executes the independent LLM
predictor and bounded judge. It also applies the configured external-model policy.

### `mlb/models/prediction/policy_flags.py`

Controls the role of THE BAT X through `SPORTSAID_BATX_MODE`.

### `mlb/db.py`

Defines append-only line, prediction, result and calibration tables in SQLite.
It also stores idempotent postgame simulation artifacts: pitcher arsenal usage,
batter whiff response by pitch type, pitcher batted-ball shape, and team fielding
conversion/error rates.

### `mlb/data_fetching/game_artifacts.py`

Parses the completed game's pitch-by-pitch feed during grading. Reruns replace
the same game/player/pitch rows, so artifact refreshes cannot double-count a game.
Daily prediction reads only artifact rows dated before the game being predicted.

### `mlb/models/training/calibration.py`

Builds chronological calibration snapshots using one unique, priced, graded
observation per game/player/market/line/source. Repeated intraday copies are excluded.

### `mlb/pipeline.py`

Provides the command-line entry point for odds, prediction, LLM, grading, calibration
and diagnostic stages.

## Forecast construction

### Stable skill prior

Recent results must not replace a stable pitcher-skill estimate by themselves. A
recent start should materially affect the forecast only when supported by an
underlying change such as:

- velocity or movement;
- pitch-mix usage;
- CSW or whiff rate;
- command change;
- injury or recovery status;
- role or pitch-count change.

### Workload states

Each pitcher should be modeled through three workload states:

1. **Short** — command failure, injury caution, weather interruption or early hook.
2. **Normal** — expected current role and typical pitch count.
3. **Extended** — efficient outing, rested pitcher and favorable game state.

The model uses the weighted mixture rather than treating the most pessimistic state as
the median. An outs under should not qualify solely because the short state is
favorable.

### Strikeout model

Strikeouts are driven by:

- pitcher strikeout skill;
- expected batters faced;
- opponent contact and K rate by handedness;
- pitch-mix fit against the actual lineup;
- park, catcher and umpire effects;
- times-through-the-order effects;
- workload uncertainty.

The numerical simulator applies an arsenal matchup factor by weighting the
starter's observed pitch usage against the posted lineup's credibility-weighted
whiff rate for each pitch type. Missing or thin history resolves toward a neutral
factor.

Every strikeout under receives a `+4 batters faced` stress test. If the under loses
its edge under that modest workload increase, it should be downgraded rather than
presented as a high-confidence play.

### Pitching-outs model

Outs require a full distribution, not only a mean. The model should represent early
hook probability, normal completion and efficient deep-start outcomes. Inputs include:

- recent pitch counts and completed workload ramp;
- manager tendencies;
- bullpen availability;
- days rest;
- opener or bulk-role risk;
- opponent OBP and lineup quality;
- weather interruption.

Because recent performance showed overconfidence in early-hook scenarios, outs unders
currently require a higher edge than ordinary strikeout markets.

The numerical hook is a probability, not a predetermined pitch limit. Recent
pitch counts establish a workload center; pitcher-created traffic and third-time-
through exposure raise removal probability, while efficient performance lowers it.
Defensive errors extend pitch/inning workload but are excluded from the pitcher's
poor-performance traffic count.

### Fantasy-score model

Pitcher fantasy score should be simulated jointly. Outs, strikeouts, earned runs, win
and quality-start bonuses are correlated and must not be added as independent median
components.

A recommended simulation order is:

1. Simulate workload and batters faced.
2. Simulate strikeouts conditional on batters faced.
3. Simulate earned runs conditional on innings and matchup.
4. Determine quality-start status from the same innings/run outcome.
5. Determine win eligibility and game outcome conditionally.
6. Apply the platform's exact scoring rules.

## Decision layer

### Current edge requirements

| Market | Minimum edge |
|---|---:|
| Standard strikeout or normal market | 6 percentage points |
| Pitching-outs under | 8 percentage points |
| Pitcher fantasy score | 10 percentage points |

These are temporary conservative requirements while the clean historical sample is
still growing.

### Probability cap

Ordinary final side probabilities are temporarily capped at 64%. This prevents the
system from publishing 70%+ confidence before its dispersion and calibration are
supported by enough unique priced observations.

### One primary market per pitcher

Only one official primary prop can be selected for each pitcher. Other favorable
markets are stored as correlated alternatives. This prevents a single pitcher thesis
from appearing as several independent model wins or losses.

### Lineup updates

Confirmed lineups are required, but routine changes should not cause large subjective
swings. A movement larger than roughly two percentage points requires a measurable
change such as:

- material lineup K-rate change;
- material OBP change;
- multiple meaningful platoon changes;
- absence of a star or elite contact hitter;
- extreme pitch-type matchup change.

## THE BAT X and external-model policy

### Principle

THE BAT X is an external benchmark, not a component that should define SportsAid's
standalone projection. Its purpose is to identify possible blind spots while the
internal model is still developing.

It must never:

- supply the base SportsAid projection;
- automatically choose the over or under;
- replace verified role, lineup or workload information;
- become a permanent requirement for the pipeline to function.

### Feature flag

```bash
SPORTSAID_BATX_MODE=bootstrap_gate
SPORTSAID_BATX_MODE=advisory
SPORTSAID_BATX_MODE=off
```

#### `bootstrap_gate`

Default during early development. A large disagreement—currently 0.75 Ks, 1.5 outs
or 3.5 fantasy points—triggers review or pass unless a verified current input explains
why SportsAid should differ.

This is a guardrail, not an averaging rule. THE BAT X still cannot create the base
forecast or reverse a side automatically.

#### `advisory`

Target mode for a healthy standalone pipeline. External projections are logged and
may reduce confidence, but cannot independently veto a play. Decisions are controlled
by the SportsAid model, verified current facts and market evidence.

#### `off`

Ablation mode. External projections are excluded from the decision prompt so the team
can measure standalone performance and detect hidden dependence.

### Graduation criteria

Move the default from `bootstrap_gate` to `advisory` after the standalone system has:

- at least 200 unique, priced, graded judge predictions overall;
- at least 75 observations in each primary market being promoted;
- stable or improving Brier score and log loss across multiple rolling windows;
- acceptable calibration error in the 55%-65% probability range;
- no repeated stale-line, role or lineup integrity failures;
- evidence that standalone ablation performance is not materially worse than gated
  performance;
- documented review of major SportsAid/THE BAT X disagreements.

These thresholds are policy targets, not automatic proof of model quality. Promotion
should use rolling out-of-sample results rather than one strong week.

## Data integrity

A valid scored recommendation requires:

- exact platform and line;
- payout type or market price;
- pregame timestamp;
- confirmed pitcher role;
- confirmed opposing lineup for official play status;
- stored probability and decision rationale;
- no edits after the result is known.

When the live PrizePicks or Underdog board is inaccessible, the automatic pipeline
must record a zero-line run rather than substitute stale or inferred lines.

## Performance evaluation

Raw hit rate is insufficient. Track:

- unique-pitcher hit rate;
- Brier score;
- log loss;
- calibration error by probability bucket;
- closing-line value;
- results by market and model source;
- results by workload-state confidence;
- results with THE BAT X in gate, advisory and off modes;
- performance before and after lineup confirmation;
- frequency and outcome of late ranking reversals.

Same-pitcher alternatives should never count as independent observations in slate
summaries.

## Running the pipeline

```bash
# Full pipeline
python mlb/pipeline.py

# Refresh odds
python mlb/pipeline.py --stage odds

# Numerical simulation
python mlb/pipeline.py --stage predict

# Fresh research, independent LLM estimate and bounded judge
python mlb/pipeline.py --stage llm

# Grade a date and update calibration
python mlb/pipeline.py --stage grade --date 2026-08-03

# Print or rebuild calibration
python mlb/pipeline.py --stage calibrate

# Check environment and source readiness
python mlb/pipeline.py --stage doctor
```

## Building chronological backtest data

The real-data builder is `mlb/data_fetching/build_backtest_data.py`. It reconstructs the
serving inputs in game order rather than asking the current API for a completed
season aggregate.

```bash
python -m mlb.data_fetching.build_backtest_data \
  2025-03-27 2025-09-28 \
  --warmup-start 2024-03-20 \
  --out mlb/data/cache/backtest_2025.json

python -m mlb.models.training.backtest \
  mlb/data/cache/backtest_2025.json \
  --sims 10000
```

### Chronological lifecycle

For every completed game, the builder performs these operations in order:

1. Read only pitcher, batter, workload, arsenal and fielding history accumulated
   from earlier games.
2. Snapshot both starters' simulation inputs.
3. Generate synthetic evaluation lines from a separate small pregame simulation.
4. Attach the final strikeout and out results for scoring.
5. Ingest the completed game's plate appearances, pitch usage, batted-ball data,
   DP opportunities and errors for subsequent games.

The in-memory artifact database is isolated from `shared/data/sportsaid.db`. Building historical
data therefore cannot contaminate live serving artifacts.

Game feeds are downloaded concurrently within each date, but their state changes are
applied in scheduled order. `--workers` controls download concurrency (default 8)
without changing the chronological model state.

### Warmup period

Use a complete prior regular season for a serious test. The warmup games produce no
evaluation rows; they establish:

- previous-season pitcher and batter priors;
- the starter's last five pitch counts;
- pitcher arsenal and ground-ball history;
- hitter whiff history by pitch type;
- team DP conversion and error rates.

When `--warmup-start` is omitted, the builder starts on March 1 of the previous year.
Spring-training games are excluded. A shorter warmup is useful for smoke tests but
will cause more inputs to resolve toward neutral league defaults.

### Leakage controls

- The current game is ingested only after its pregame row is constructed.
- Same-day games are ordered by scheduled start time, so the first game of a
  doubleheader may inform the second.
- Historical artifact queries read only games already processed in memory.
- Synthetic test lines are centered on an independent pregame simulation, never on
  the starter's actual result.
- The output records rate sample sizes and the resolved defense/arsenal inputs so a
  projection can be audited without rebuilding the source state.
- Park and umpire caches default to neutral because a cache built from the test
  season would leak future games. `--cached-context-factors` should be used only when
  those cache files were constructed exclusively from dates before the test period.

### Output files

The primary JSON remains a list accepted directly by `mlb.models.training.backtest`. Each
row contains identifiers, pregame model inputs, model-derived test lines and actual
results. A neighboring `.meta.json` file records the requested period, warmup range,
schema version, TTO setting, line-generation simulation count and any skipped games.

By default the build fails on the first missing/corrupt game so incomplete history is
not silently treated as real history. `--skip-errors` is available for exploration;
always inspect the metadata failure list before trusting that dataset.

StatsAPI does not provide historical sportsbook prices. Synthetic lines support
calibration, Brier score and log-loss evaluation, but not honest ROI. For ROI and CLV,
merge a timestamped odds archive:

```bash
python -m mlb.models.training.backtest mlb/data/cache/backtest_2025.json \
  --odds path/to/historical_odds.csv --sims 10000
```

Each start receives several nearby synthetic thresholds. Those thresholds test the
shape of the forecast distribution, but they are correlated observations from the
same game. Treat the number of unique `(game_pk, pitcher_id)` starts—not the larger
number of scored thresholds—as the independent sample size.

### Known historical limitations

- Historical CSW/velocity form is currently neutral (`k_form = 1.0`).
- Park and umpire factors are neutral unless explicitly enabled with a cache known
  to predate the evaluation window.
- The hook model sees traffic, pitch workload, errors and TTO exposure, but still
  does not reconstruct the historical score or bullpen state.
- Generated lines are evaluation thresholds, not evidence that a sportsbook offered
  those markets.

## Change-management rules

Any material change to probability, selection, exposure or grading behavior should:

1. Increment the policy or model version.
2. Add or update regression tests.
3. Document the reason and expected effect.
4. Avoid retroactively changing historical predictions.
5. Be evaluated on a future out-of-sample window.

## Near-term roadmap

1. Capture a complete, timestamped manual-board ledger when platform APIs are
   inaccessible.
2. Calibrate strikeout dispersion and workload-state distributions from historical
   starter game logs.
3. Add joint fantasy-score simulation.
4. Report unique-pitcher performance and late-lineup-reversal outcomes directly.
5. Run scheduled `bootstrap_gate`, `advisory` and `off` comparisons.
6. Promote THE BAT X to advisory-only when the graduation criteria are met.
