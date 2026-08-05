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

### `mlb/data_today.py`

Retrieves schedule, probable pitchers, pitcher profiles, batter rates, lineups and
other current MLB inputs used by the model.

### `mlb/odds_props.py`

Captures sportsbook pitcher-prop lines and prices. Every book snapshot receives a
shared timestamp so moved or removed lines can be distinguished from the current
board.

### `mlb/predict.py`

Runs the numerical simulator. This lane should remain independent of the LLM audit.
Its purpose is to convert stable pitcher skill, expected batters faced and game context
into probability distributions.

### `mlb/llm_saved_pipeline.py`

Defines the current chat-aligned qualitative research and bounded-judge policy. Policy
changes are versioned because prompt behavior affects historical comparability.

### `mlb/llm_chat_runner.py`

Refreshes current research on every invocation, then executes the independent LLM
predictor and bounded judge. It also applies the configured external-model policy.

### `mlb/policy_flags.py`

Controls the role of THE BAT X through `SPORTSAID_BATX_MODE`.

### `mlb/db.py`

Defines append-only line, prediction, result and calibration tables in SQLite.

### `mlb/calibration.py`

Builds chronological calibration snapshots using one unique, priced, graded
observation per game/player/market/line/source. Repeated intraday copies are excluded.

### `run_pipeline.py`

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

| Market | Minimum edge | Minimum final side probability |
|---|---:|---:|
| Pitcher strikeouts | 8 percentage points | 60% |
| Pitching-outs under | 10 percentage points | 60% |
| Ordinary non-K market | 6 percentage points | 57% |
| Pitcher fantasy score | 10 percentage points | 57% |

These are temporary conservative requirements while the clean historical sample is
still growing.

### Deterministic lane-consistency gates

The v4.1 decision layer does not rely on the judge prompt alone. A candidate is
suppressed from the official card when:

- the numerical simulator and independent LLM materially support opposite sides;
- their P(over) estimates differ by more than 12 percentage points;
- the final side probability is below the market-specific floor;
- the edge is below the market-specific threshold;
- the line is unpriced, stale or otherwise unverified.

The blend is 75% simulator and 25% independent LLM. The bounded judge may move the
blend by at most three percentage points. This limits narrative-driven promotions
after lineup confirmation while preserving the LLM's ability to veto a stale role or
workload assumption.

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
swings. A movement larger than roughly 1.5 percentage points requires a measurable
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

For strikeouts, an accessible reputable external projection within 0.25 Ks of the
posted line is treated as a review/pass signal unless a verified current role,
workload or pitch-mix input explains why SportsAid should differ.

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
python run_pipeline.py

# Refresh odds
python run_pipeline.py --stage odds

# Numerical simulation
python run_pipeline.py --stage predict

# Fresh research, independent LLM estimate and bounded judge
python run_pipeline.py --stage llm

# Grade a date and update calibration
python run_pipeline.py --stage grade --date 2026-08-03

# Print or rebuild calibration
python run_pipeline.py --stage calibrate

# Check environment and source readiness
python run_pipeline.py --stage doctor
```

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
