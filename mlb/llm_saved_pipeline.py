"""Chat-aligned policy layer for the independent MLB LLM pipeline.

Version 4 applies the August 2026 two-week audit without rewriting the numerical
simulator. It tightens the decision layer, keeps the simulator and LLM independent,
and makes the final judge conservative about workload, recent box-score noise,
model disagreement, correlated markets, and published confidence.
"""
from __future__ import annotations

import json

import llm as base

PIPELINE_VERSION = "chat-preserved-v4.0"
MAX_SIDE_PROBABILITY = 0.64
BASE_MIN_EDGE = 0.06
OUTS_UNDER_MIN_EDGE = 0.08
FANTASY_MIN_EDGE = 0.10

base.MAX_SEARCHES_PER_GAME = 10
base.MIN_EDGE = BASE_MIN_EDGE

base.FACTS_SYSTEM = """
You are the research lane for an MLB pitcher-prop betting pipeline. Mirror the
preserved SportsAid chat audit. Find current, decision-relevant context; never
manufacture facts and never rely on stale previews when an official or newer source
exists.

Source priority:
1. MLB official probable pitchers, box scores, transactions and Statcast.
2. Team/manager reporting and credentialed beat writers.
3. Current sportsbook or pick'em pages for line context only.
4. Reputable public models or analysis such as THE BAT/X, Dimers and ESPN.

For every pitcher, investigate all of the following where available:
- confirmed starter and current opponent; flag stale or mismatched sources;
- last 3-5 pitch counts, innings and batters faced, plus any explicit usage ramp;
- injury return, rest, velocity, movement, command and pitch-mix changes;
- manager hook tendencies, bullpen fatigue and likely leash;
- opponent K/contact profile by handedness and confirmed-lineup changes;
- career/team H2H and current-roster BvP, always labeling small samples;
- pitch-type matchup against the likely lineup;
- park, roof, weather and home-plate umpire;
- THE BAT X or other public-model projection/disagreement when accessible;
- material line movement and the strongest reason a soft line may be intentional.

Recency rules:
- Prefer information from today, then the last 3 days.
- Recent box scores are not evidence of a skill change by themselves.
- A recent adjustment matters only when supported by velocity, pitch mix, CSW/whiff,
  role, pitch-count, health or other underlying evidence.
- Explicitly identify unavailable or conflicting information.

Do not invent exact projections, odds or line values. The application supplies hard
numbers separately. Return compact factual flags with source labels and dates.
"""

base.FACTS_PROMPT = """Game date: {game_date}. Matchup: {away} at {home}.
Official probable starters supplied by the app: {away_pitcher} and {home_pitcher}.

Perform a full audit on BOTH pitchers. Search current sources and return ONLY JSON:
{{
  "pitchers": [
    {{
      "name": "...",
      "flags": [
        "USAGE — dated factual note",
        "HEALTH/STUFF — dated factual note",
        "MATCHUP — handedness/lineup contact note",
        "H2H/BvP — result plus sample-size warning",
        "PITCH MIX — relevant matchup note",
        "LEASH/BULLPEN — factual note",
        "MARKET/MODELS — THE BAT X or other disagreement, or unavailable",
        "TRAP CASE — strongest reason a soft-looking line may be correct"
      ],
      "outlook": "neutral|positive|negative"
    }}
  ],
  "game_notes": ["weather/roof/umpire/lineup-status notes"],
  "sources": ["source name, publication date, URL"]
}}

An empty or explicitly unavailable field is better than a guess. Confirm every source
refers to today's pitcher, opponent and role before using it."""

base.PREDICT_SYSTEM = """
You are the independent LLM lane of an MLB pitcher-prop pipeline. Follow the same
preserved SportsAid chat process used for full-board analysis. You see posted lines
but do NOT see prices, market probabilities or simulator output. This separation is
mandatory.

For every strikeout and pitching-outs line:
1. Build THREE workload states: short, normal and extended. Assign probabilities and
   use their mixture; never make an early hook the default merely because a pitcher
   walked hitters or had one short recent start.
2. Translate each workload state into batters faced and outs.
3. Derive strikeouts from pitcher skill, lineup contact by handedness, pitch-mix fit,
   park and umpire—not season K/9 or recent box scores alone.
4. For every strikeout UNDER, repeat the estimate with FOUR additional batters faced.
   If the under no longer survives, downgrade it to pass/lean rather than a top play.
5. Treat recent results as modest evidence unless corroborated by velocity, pitch mix,
   CSW/whiff, role, health or a documented pitch-count change.
6. Audit manager leash, bullpen rest, days rest, injury context and completed ramps.
7. Quantify lineup impact. A routine confirmed-lineup change should move probability
   no more than two points unless K%, OBP, platoon count or a star absence changes
   materially.
8. Identify THE BAT X/model disagreement. Differences above 0.75 Ks or 1.5 outs are
   pass/review signals unless a verified input explains the gap.
9. Explain why the book may have posted the line and the strongest counterargument.
10. Keep probabilities conservative. Most props belong near 50%; ordinary published
    side probabilities are capped near 64% until the historical sample is larger.

Do not force a side when conflicting evidence removes the edge. The rationale MUST
include: PROJECTION; WORKLOAD-STATES; PLUS-4-BF; USAGE; MATCHUP; PITCH-MIX;
WHY-LINE-EXISTS; MODEL-DISAGREEMENT; COUNTERARGUMENT; UNCERTAINTY; VERDICT.
"""

base.PREDICT_PROMPT = """Pipeline policy version: {pipeline_version}
{game_date}: {pitcher} ({hand}HP, {team_side} starter).

Pitcher season per-PA rates, shrunk toward league average:
K {pk:.3f}; BB {pbb:.3f}; hit {phit:.3f}.
Recent stuff/velocity form: {form_summary}
Recent start pitch counts: {pitch_counts}
Opposing lineup per-PA K rates vs {hand}HP, slots 1-9: {lineup_ks}
League-average K rate: 0.223
Park K factor: {k_park:.2f}
Home-plate umpire K factor: {k_ump:.2f} ({ump_name})
Lineup status: {lineup_status}

Current research audit:
{flags}
Game-level notes: {game_notes}

Posted props:
{lines_block}

For every line, estimate P(over), excluding pushes. Work from workload-state mixture
-> batters faced -> outs -> strikeouts so K and outs forecasts are internally
consistent. Explicitly show the +4-batters-faced stress test for K unders and explain
any material difference from THE BAT X or another public model. Confidence reflects
data quality, not conviction.""".replace("{pipeline_version}", PIPELINE_VERSION)

base.JUDGE_SYSTEM = """
You are the final bounded judge for two parallel MLB prop pipelines:
- a plate-appearance Monte Carlo model, strong on numerical rates/distributions;
- an independent chat-aligned LLM audit, strong on current usage and qualitative news.
You also see the de-vigged market, which is a strong prior.

Apply these hard decision rules:
- normal strikeout plays require at least a six-point edge;
- outs UNDERS require at least an eight-point edge because early-hook assumptions have
  been overconfident; fantasy-score plays require ten points when supported;
- veto a K under that fails the +4-batters-faced stress test;
- veto workload unders that win only in the short-workload state;
- treat THE BAT X disagreement above 0.75 Ks, 1.5 outs or 3.5 fantasy points as a
  pass/review signal unless a verified current input explains it;
- routine lineup confirmation should not swing probability by more than two points;
- recent box scores cannot override stable skill without an underlying change;
- select at most ONE primary bet per pitcher across all markets; other favorable
  markets are correlated alternatives, not independent official plays;
- prefer passing over manufacturing confidence.

Your numeric authority remains narrow and is clamped in code. Ordinary final side
probability is capped at 64%. The rationale MUST contain: WHY LINE EXISTS; SUPPORT;
WORKLOAD STATES; STRESS TEST; MODEL DISAGREEMENT; COUNTERARGUMENT; DATA QUALITY;
PRIMARY MARKET; ACTION.
"""

base.JUDGE_PROMPT = """Pipeline policy version: {pipeline_version}
{game_date}: final audit for {pitcher}.

Research flags: {flags}
Game notes: {game_notes}

For each prop, the application provides simulator P(over), independent LLM P(over)
with confidence, de-vigged market P(over), and a {w_sim:.0%}/{w_llm:.0%} blend:
{props_block}

Apply the full audit. For every prop:
1. Explain why the market line may be intentionally correct.
2. Name the strongest supporting evidence and strongest counterargument.
3. Check short/normal/extended workload states.
4. For K unders, verify the +4-batters-faced result.
5. Flag stale data, role uncertainty, recent-box-score overreaction and meaningful
   THE BAT X/market disagreement.
6. Return final_p_over within the allowed ±{band:.0%} band.
7. Mark bet only when the edge survives the required threshold.
8. Across this pitcher's markets, identify only ONE PRIMARY MARKET; mark all other
   markets pass even when they are useful correlated alternatives.

Use the labels WHY LINE EXISTS; SUPPORT; WORKLOAD STATES; STRESS TEST;
MODEL DISAGREEMENT; COUNTERARGUMENT; DATA QUALITY; PRIMARY MARKET; ACTION.""".replace(
    "{pipeline_version}", PIPELINE_VERSION
)


def _cap_side_probability(p_over: float) -> float:
    """Temporarily cap either side's published probability at 64%."""
    return min(MAX_SIDE_PROBABILITY, max(1.0 - MAX_SIDE_PROBABILITY, float(p_over)))


def _side_and_edge(p_over: float, market_p_over: float):
    over_edge = float(p_over) - float(market_p_over)
    under_edge = float(market_p_over) - float(p_over)
    return ("over", over_edge) if over_edge >= under_edge else ("under", under_edge)


def _required_edge(market: str, side: str) -> float:
    if market == "pitcher_fantasy_score":
        return FANTASY_MIN_EDGE
    if market == "pitcher_outs" and side == "under":
        return OUTS_UNDER_MIN_EDGE
    return BASE_MIN_EDGE


def _select_primary_key(records):
    """Choose at most one eligible primary market for a pitcher."""
    eligible = []
    for record in records:
        if record.get("action") != "bet" or record.get("market_p") is None:
            continue
        side, edge = _side_and_edge(record["p_over"], record["market_p"])
        required = _required_edge(record["key"][0], side)
        if edge + 1e-12 >= required:
            eligible.append((edge, record["key"]))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
    return eligible[0][1]


_original_priced_row = base._priced_row


def _priced_row_v4(*args, **kwargs):
    """Attach the policy version to every LLM/judge output."""
    raw = kwargs.get("rationale")
    try:
        metadata = json.loads(raw) if raw else {}
        if not isinstance(metadata, dict):
            metadata = {"rationale": raw}
    except (TypeError, json.JSONDecodeError):
        metadata = {"rationale": raw}
    metadata["policy_version"] = PIPELINE_VERSION
    metadata["probability_cap"] = MAX_SIDE_PROBABILITY
    metadata["base_min_edge"] = BASE_MIN_EDGE
    kwargs["rationale"] = json.dumps(metadata)
    return _original_priced_row(*args, **kwargs)


base._priced_row = _priced_row_v4


def _judge_pitcher_v4(conn, client, run_id, game_date, pitcher_name, candidates,
                       flags, lines_by_player):
    """Bound, cap and de-correlate judge decisions at the pitcher level."""
    props_block = []
    blends = {}
    for (market, line), (sim, llm_row) in sorted(candidates.items()):
        blend = base.W_SIM * sim["p_over"] + (1 - base.W_SIM) * llm_row["p_over"]
        blends[(market, line)] = blend
        llm_meta = base._lenient_json(llm_row["rationale"] or "") or {}
        props_block.append(
            f"- {market} line {line}: sim {sim['p_over']:.3f}, "
            f"llm {llm_row['p_over']:.3f} (conf {llm_meta.get('confidence', '?')}), "
            f"market {sim['market_p_over']:.3f}, blend {blend:.3f}"
        )

    prompt = base.JUDGE_PROMPT.format(
        game_date=game_date,
        pitcher=pitcher_name,
        flags=flags["flags"] or "none found",
        game_notes=flags["game_notes"] or "none",
        w_sim=base.W_SIM,
        w_llm=1 - base.W_SIM,
        band=base.JUDGE_BAND,
        props_block="\n".join(props_block),
    )
    response = client.messages.create(
        model=base.MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=base.JUDGE_SYSTEM,
        output_config={"format": base.JUDGE_SCHEMA},
        messages=[{"role": "user", "content": prompt}],
    )
    data = json.loads(base._text_of(response))

    prepared = []
    for decision in data.get("decisions", []):
        key = (decision["market"], float(decision["line"]))
        if key not in candidates:
            continue
        sim, llm_row = candidates[key]
        blend = blends[key]
        p_final = min(
            max(float(decision["final_p_over"]), blend - base.JUDGE_BAND),
            blend + base.JUDGE_BAND,
        )
        p_final = _cap_side_probability(p_final)
        prepared.append({
            "key": key,
            "action": decision["action"],
            "p_over": p_final,
            "market_p": sim["market_p_over"],
            "decision": decision,
            "sim": sim,
            "llm": llm_row,
            "blend": blend,
        })

    primary_key = _select_primary_key(prepared)
    logged = 0
    for record in prepared:
        key = record["key"]
        sim = record["sim"]
        books = lines_by_player.get((base._norm(pitcher_name), key[0], key[1]), [])
        selected = key == primary_key and record["action"] == "bet"
        suppression_reason = None
        if record["action"] == "pass":
            suppression_reason = "judge_veto"
        elif primary_key is None:
            suppression_reason = "edge_below_v4_threshold"
        elif key != primary_key:
            suppression_reason = "secondary_correlated_market"

        row = base._priced_row(
            game_date,
            {"game_pk": sim["game_pk"]},
            {"name": pitcher_name, "id": sim["player_id"]},
            key[0],
            key[1],
            record["p_over"],
            books,
            sim["market_p_over"],
            run_id=run_id,
            source="judge",
            rationale=json.dumps({
                "action": record["action"],
                "rationale": record["decision"].get("rationale"),
                "blend": round(record["blend"], 4),
                "primary_market": selected,
                "suppression_reason": suppression_reason,
            }),
            suppress_bet=not selected,
        )
        base.db.insert_prediction(conn, row)
        logged += 1
    return logged


base._judge_pitcher = _judge_pitcher_v4


def run_all(game_date=None):
    """Run the existing three stages under the recalibrated chat policy."""
    return base.run_all(game_date)


def facts_stage(game_date=None):
    return base.facts_stage(game_date)


def llm_predict_stage(game_date=None, run_id=None):
    return base.llm_predict_stage(game_date, run_id)


def judge_stage(game_date=None, run_id=None):
    return base.judge_stage(game_date, run_id)


if __name__ == "__main__":
    import sys

    run_all(sys.argv[1] if len(sys.argv) > 1 else None)
