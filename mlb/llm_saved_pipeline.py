"""Chat-aligned qualitative layer for the merged v4.2/v5 MLB policy.

The independent LLM sees posted lines but not prices or simulator output. The final
judge uses the deterministic 75/25 blend, may move at most three points, preserves
the simulator side, and can only affirm or veto an existing simulator candidate.
"""
from __future__ import annotations

import json

import calibration
import decision
import llm as base

PIPELINE_VERSION = decision.POLICY_VERSION
base.MAX_SEARCHES_PER_GAME = 10
base.W_SIM = 0.75
base.JUDGE_BAND = 0.03

base.FACTS_SYSTEM = """
You are the independent research lane for an MLB pitcher-prop pipeline. Use current,
source-dated facts and never manufacture hard numbers. Prioritize MLB/team sources,
credentialed reporting, then reputable market/external-model context. For each
pitcher audit confirmed role, last 3-5 workloads, rest/ramp, health, velocity/stuff,
pitch mix, actual lineup contact by handedness, manager hook behavior, bullpen,
weather/roof/umpire, and material line movement. Label unavailable/conflicting data
and small H2H samples. External projections are diagnostics, never the base forecast.
"""

base.PREDICT_SYSTEM = """
You are an independent MLB pitcher-prop forecaster. You may see the exact posted line
but not sportsbook prices, market probabilities, or simulator output.

For each line:
1. Start from a stable multi-season/rest-of-season skill prior; recent game logs alone
   are not a skill change.
2. Build explicit short, normal, and extended workload states with probabilities.
3. Translate state -> pitches -> batters faced -> outs -> strikeouts.
4. Use handedness, actual lineup, pitch-mix fit, park, umpire, stuff/velocity and true
   times-through-order effects.
5. For K unders report base, +2 BF, and +4 BF probabilities and label skill-under vs
   workload-under. Do not use a blanket +4 BF veto.
6. Treat projected lineups as shadow analysis; confirmed lineups are required for an
   official decision.
7. Explain why the line exists, strongest support, strongest counterargument, model
   disagreement, and data quality. Most probabilities belong near 50%.
Return compact JSON only under the existing schema.
"""

base.JUDGE_SYSTEM = """
You are an adversarial final safety judge. The simulator creates candidates; the LLM
is an independent audit; the market is a strong prior. Use a deterministic 75% sim /
25% LLM blend. You may move no more than three percentage points and may only erode,
never strengthen or flip, the simulator side. Pass when sim and LLM differ by more
than 12 points, favor opposite sides, exact lines are stale/unverified, role/workload
is uncertain, or external near-line disagreement lacks a unique verified explanation.
Only one primary market per pitcher may become official. During probation, at most two
independent K props per slate and no official outs until workload validation is enabled.
"""

base.JUDGE_PROMPT = """{game_date}: final audit for {pitcher}.
Research flags: {flags}
Game notes: {game_notes}

Each row shows simulator P(over), independent LLM P(over), market P(over), simulator
candidate side, and deterministic 75/25 blend:
{props_block}

For every row return final_p_over within ±{band:.0%} of the blend and action bet/pass.
A bet only affirms an existing simulator candidate. Explain WHY LINE EXISTS; SUPPORT;
WORKLOAD STATES; BASE/+2/+4 BF STRESS; MODEL DISAGREEMENT; COUNTERARGUMENT; DATA
QUALITY; PRIMARY MARKET; ACTION."""


def _json_meta(value):
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {"rationale": value}
    except (TypeError, ValueError):
        return {"rationale": value}


def _best_book(books, side):
    if side not in {"over", "under"}:
        return books[0] if books else None
    from betting_math import american_to_decimal
    column = f"{side}_odds"
    priced = [row for row in books if row[column] is not None]
    return max(priced, key=lambda row: american_to_decimal(row[column]), default=None)


def _judge_pitcher_v5(conn, client, run_id, game_date, pitcher_name, candidates,
                       flags, lines_by_player):
    props_block, blends = [], {}
    for key, (sim, llm_row) in sorted(candidates.items()):
        blend = base.W_SIM * float(sim["p_over"]) + (1.0 - base.W_SIM) * float(llm_row["p_over"])
        blends[key] = blend
        props_block.append(
            f"- {key[0]} {key[1]}: sim {sim['p_over']:.3f}; llm {llm_row['p_over']:.3f}; "
            f"market {sim['market_p_over']:.3f}; candidate "
            f"{sim['candidate_side'] or sim['bet_side'] or 'HOLD'}; blend {blend:.3f}"
        )
    prompt = base.JUDGE_PROMPT.format(
        game_date=game_date, pitcher=pitcher_name,
        flags=flags["flags"] or "none found",
        game_notes=flags["game_notes"] or "none",
        band=base.JUDGE_BAND, props_block="\n".join(props_block),
        w_sim=base.W_SIM, w_llm=1-base.W_SIM,
    )
    response = client.messages.create(
        model=base.MODEL, max_tokens=16000, thinking={"type":"adaptive"},
        system=base.JUDGE_SYSTEM, output_config={"format":base.JUDGE_SCHEMA},
        messages=[{"role":"user","content":prompt}],
    )
    decisions = {
        (item["market"], float(item["line"])): item
        for item in json.loads(base._text_of(response)).get("decisions", [])
    }

    prepared = []
    for key, (sim, llm_row) in candidates.items():
        market, line = key
        market_p = float(sim["market_p_over"])
        approved_side = sim["candidate_side"] or sim["bet_side"]
        sim_side = "over" if float(sim["p_over"]) >= market_p else "under"
        llm_side = "over" if float(llm_row["p_over"]) >= market_p else "under"
        lane_gap = abs(float(sim["p_over"]) - float(llm_row["p_over"]))
        item = decisions.get(key, {"action":"pass", "final_p_over":blends[key],
                                   "rationale":"judge omitted line"})
        requested = min(max(float(item["final_p_over"]), blends[key]-base.JUDGE_BAND),
                        blends[key]+base.JUDGE_BAND)
        final_p = decision.erode_only(float(sim["p_over"]), requested, approved_side, base.JUDGE_BAND)
        reasons = []
        sim_meta = _json_meta(sim["quality_json"])
        reasons.extend(sim_meta.get("hold_reasons") or [])
        if lane_gap > decision.MAX_LANE_GAP:
            reasons.append(f"sim/LLM gap {lane_gap:.1%} exceeds {decision.MAX_LANE_GAP:.1%}")
        if approved_side and sim_side != llm_side:
            reasons.append("simulator and independent LLM favor opposite sides")
        if item.get("action") != "bet":
            reasons.append(f"LLM judge veto: {item.get('rationale') or 'unspecified'}")

        books = lines_by_player.get((base._norm(pitcher_name), market, line), [])
        priced_books = {b["book"] for b in books
                        if b["over_odds"] is not None and b["under_odds"] is not None}
        validation = calibration.validation_status(
            conn, market, "sim", as_of_date=game_date,
            policy_version=decision.POLICY_VERSION,
        )
        line_verified = bool(books) and all(bool(b["verified"]) for b in books)
        standard = bool(books) and all((b["line_tier"] or "standard") == "standard" for b in books)
        call = decision.decide(
            final_p, market_p, book_count=len(priced_books), validation=validation,
            quality_reasons=reasons, allow_auto=bool(approved_side) and item.get("action")=="bet",
            required_side=approved_side, market=market, line_verified=line_verified,
            standard_tier=standard, lineup_confirmed=sim["lineup_status"]=="confirmed",
            starter_verified=bool(sim["starter_verified"]),
            workload_verified=bool(sim["workload_verified"]),
            uncertainty=float(sim["uncertainty_score"] or 0.0), lane_gap=lane_gap,
        )
        edge = abs(final_p-market_p)
        prepared.append({"key":key,"sim":sim,"llm":llm_row,"item":item,"books":books,
                         "call":call,"final_p":final_p,"blend":blends[key],"edge":edge,
                         "approved_side":approved_side,"validation":validation,
                         "lane_gap":lane_gap})

    official_candidates = [row for row in prepared if row["call"].status=="official"]
    primary = max(official_candidates, key=lambda row: row["edge"], default=None)
    logged = 0
    for record in prepared:
        call = record["call"]
        selected = record is primary and call.status == "official"
        status = call.status
        suppression = call.suppression_reason
        bet_side = call.bet_side if selected else None
        if call.status == "official" and not selected:
            status = "candidate"
            suppression = "secondary_correlated_market"
        best = _best_book(record["books"], bet_side or record["approved_side"])
        kelly = 0.0
        if selected and best:
            side_p = record["final_p"] if bet_side=="over" else 1.0-record["final_p"]
            odds = best["over_odds"] if bet_side=="over" else best["under_odds"]
            kelly = base.kelly_fraction(side_p, odds, cap=decision.MAX_KELLY)
        sim = record["sim"]
        rationale = {
            "action": record["item"].get("action"),
            "rationale": record["item"].get("rationale"),
            "simulator_p": float(sim["p_over"]),
            "llm_p": float(record["llm"]["p_over"]),
            "blend_75_25": record["blend"],
            "lane_gap": record["lane_gap"],
            "authority": "three-point-erode-only",
            "primary_market": selected,
            "decision_status": status,
            "suppression_reason": suppression,
            "hold_reasons": list(call.hold_reasons),
            "validation": record["validation"],
        }
        base.db.insert_prediction(conn, {
            "run_id":run_id,"created_at":base.db.utcnow(),"game_date":game_date,
            "game_pk":sim["game_pk"],"player_name":pitcher_name,
            "player_id":sim["player_id"],"market":record["key"][0],
            "line":record["key"][1],"source":"judge","p_over":record["final_p"],
            "raw_p_over":record["final_p"],"decision_p_over":call.decision_p_over,
            "p_push":sim["p_push"],"mean":sim["mean"],
            "book":best["book"] if best else sim["book"],
            "over_odds":best["over_odds"] if best else sim["over_odds"],
            "under_odds":best["under_odds"] if best else sim["under_odds"],
            "market_p_over":sim["market_p_over"],"edge_over":call.edge_over,
            "edge_under":call.edge_under,"candidate_side":record["approved_side"],
            "bet_side":bet_side,"kelly_frac":kelly,
            "rationale":json.dumps(rationale,sort_keys=True),
            "policy_version":decision.POLICY_VERSION,
            "quality_json":json.dumps({"hold_reasons":list(call.hold_reasons),
                                       "validation":record["validation"]},sort_keys=True),
            "decision_status":status,"official":1 if selected else 0,
            "selected_for_card":1 if selected else 0,"provenance":"judge",
            "line_as_of":sim["line_as_of"],"event_start":sim["event_start"],
            "line_verified":sim["line_verified"],"line_tier":sim["line_tier"],
            "lineup_status":sim["lineup_status"],
            "starter_verified":sim["starter_verified"],
            "workload_verified":sim["workload_verified"],
            "calibration_status":sim["calibration_status"],
            "external_mode":getattr(__import__(__name__),"BATX_MODE",None),
            "suppression_reason":suppression,
            "rule_trace_json":json.dumps(call.rule_trace),
            "uncertainty_score":sim["uncertainty_score"],
            "stress_p_under_plus2":sim["stress_p_under_plus2"],
            "stress_p_under_plus4":sim["stress_p_under_plus4"],
            "under_basis":sim["under_basis"],
            "entry_market_p_side":sim["entry_market_p_side"],
        })
        logged += 1
    return logged


base._judge_pitcher = _judge_pitcher_v5


def run_all(game_date=None):
    return base.run_all(game_date)


def facts_stage(game_date=None):
    return base.facts_stage(game_date)


def llm_predict_stage(game_date=None, run_id=None):
    return base.llm_predict_stage(game_date, run_id)


def judge_stage(game_date=None, run_id=None):
    return base.judge_stage(game_date, run_id)
