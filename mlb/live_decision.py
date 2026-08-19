"""Promote independent LLM predictions into the live decision surface.

The numerical simulator remains a shadow/testing lane.  This module deliberately
never reads simulator probabilities when choosing a live side.  It consumes only
the latest independent LLM rows plus the market probability already attached to
those rows, applies conservative robustness gates, and writes a final source='judge'
row for backwards-compatible UI selection.

`source='judge'` is therefore a compatibility label, not a blended model judge.
Every promoted row records that neither the simulator nor the numerical model
influenced the live probability.
"""
from __future__ import annotations

import json
from datetime import date

import db
import llm

MAX_SIDE_PROBABILITY = 0.64
MIN_CONFIDENCE = 0.55
BASE_MIN_EDGE = 0.06
OUTS_UNDER_MIN_EDGE = 0.08


def _cap_side_probability(p_over: float) -> float:
    return min(MAX_SIDE_PROBABILITY, max(1.0 - MAX_SIDE_PROBABILITY, float(p_over)))


def _required_edge(market: str, side: str) -> float:
    if market == "pitcher_outs" and side == "under":
        return OUTS_UNDER_MIN_EDGE
    return BASE_MIN_EDGE


def _meta(row) -> dict:
    try:
        value = json.loads(row["rationale"] or "{}")
    except (TypeError, json.JSONDecodeError):
        value = {}
    return value if isinstance(value, dict) else {}


def _stress_complete(market: str, rationale: str) -> bool:
    text = (rationale or "").upper()
    common = all(label in text for label in ("WORKLOAD-STATES", "COUNTERARGUMENT", "VERDICT:"))
    if not common:
        return False
    if market == "pitcher_strikeouts":
        return "K-CONVERSION-STRESS" in text
    if market == "pitcher_outs":
        return "OUTS-TAIL-STRESS" in text
    return False


def _verdict_is_bet(rationale: str) -> bool:
    text = (rationale or "").upper()
    return "VERDICT: BET" in text


def _side_and_edge(p_over: float, market_p_over: float):
    over_edge = float(p_over) - float(market_p_over)
    under_edge = float(market_p_over) - float(p_over)
    return ("over", over_edge) if over_edge >= under_edge else ("under", under_edge)


def _select_primary(records):
    eligible = []
    for record in records:
        if not record["eligible"]:
            continue
        eligible.append((record["edge"], record["key"]))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
    return eligible[0][1]


def _ensure_lane_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS lane_run_snapshots (
      id               INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id           TEXT NOT NULL,
      game_date        TEXT NOT NULL,
      created_at       TEXT NOT NULL,
      lane             TEXT NOT NULL,
      mode             TEXT NOT NULL,
      status           TEXT NOT NULL,
      prediction_count INTEGER NOT NULL DEFAULT 0,
      metadata_json    TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_lane_run_snapshots
      ON lane_run_snapshots(game_date, run_id, lane, id);
    """)
    conn.commit()


def _record_lane(conn, run_id, game_date, lane, mode, status, count, metadata=None):
    conn.execute(
        """INSERT INTO lane_run_snapshots
           (run_id, game_date, created_at, lane, mode, status,
            prediction_count, metadata_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            game_date,
            db.utcnow(),
            lane,
            mode,
            status,
            int(count),
            json.dumps(metadata or {}, sort_keys=True),
        ),
    )
    conn.commit()


def _latest_llm_rows(conn, game_date, run_id):
    rows = conn.execute(
        """SELECT * FROM prop_predictions
           WHERE game_date=? AND run_id=? AND source='llm'
           ORDER BY id""",
        (game_date, run_id),
    ).fetchall()
    latest = {}
    for row in rows:
        key = (row["player_name"], row["market"], float(row["line"]))
        latest[key] = row
    return list(latest.values())


def _latest_llm_run(conn, game_date):
    row = conn.execute(
        """SELECT run_id FROM prop_predictions
           WHERE game_date=? AND source='llm'
           ORDER BY id DESC LIMIT 1""",
        (game_date,),
    ).fetchone()
    return row["run_id"] if row else None


def promote_llm_live(game_date=None, run_id=None):
    """Write backwards-compatible live decision rows from the LLM lane only."""
    game_date = game_date or date.today().isoformat()
    conn = db.connect()
    db.ensure_schema(conn)
    _ensure_lane_schema(conn)

    run_id = run_id or _latest_llm_run(conn, game_date)
    if not run_id:
        print("[live] no LLM run found; nothing promoted")
        conn.close()
        return 0

    sim_count = conn.execute(
        """SELECT COUNT(*) n FROM prop_predictions
           WHERE game_date=? AND run_id=? AND source='sim'""",
        (game_date, run_id),
    ).fetchone()["n"]
    llm_rows = _latest_llm_rows(conn, game_date, run_id)

    # Snapshot lane status before the live handoff.  The numerical model serving
    # lane is not wired in the current codebase; recording that absence is
    # intentional so a missing shadow ledger cannot be mistaken for a clean run.
    _record_lane(
        conn, run_id, game_date, "monte_carlo", "shadow",
        "complete" if sim_count else "missing", sim_count,
        {"influences_live": False},
    )
    _record_lane(
        conn, run_id, game_date, "model", "shadow", "not_wired", 0,
        {"influences_live": False, "reason": "serving model lane not implemented"},
    )
    _record_lane(
        conn, run_id, game_date, "llm", "live",
        "complete" if llm_rows else "missing", len(llm_rows),
        {"influenced_by_monte_carlo": False, "influenced_by_model": False},
    )

    if not llm_rows:
        conn.close()
        return 0

    lines = {}
    for row in db.latest_lines(conn, game_date):
        key = (llm._norm(row["player_name"]), row["market"], float(row["line"]))
        lines.setdefault(key, []).append(row)

    prepared = []
    for row in llm_rows:
        meta = _meta(row)
        rationale = str(meta.get("rationale") or "")
        try:
            confidence = float(meta.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0
        p_over = _cap_side_probability(row["p_over"])
        market_p = row["market_p_over"]
        complete = _stress_complete(row["market"], rationale)
        verdict_bet = _verdict_is_bet(rationale)
        side, edge = (None, None)
        clears_edge = False
        if market_p is not None:
            side, edge = _side_and_edge(p_over, market_p)
            clears_edge = edge + 1e-12 >= _required_edge(row["market"], side)
        eligible = (
            complete
            and verdict_bet
            and confidence >= MIN_CONFIDENCE
            and market_p is not None
            and clears_edge
        )
        prepared.append({
            "row": row,
            "key": (row["market"], float(row["line"])),
            "p_over": p_over,
            "market_p": market_p,
            "confidence": confidence,
            "rationale": rationale,
            "stress_complete": complete,
            "verdict_bet": verdict_bet,
            "side": side,
            "edge": edge if edge is not None else -1.0,
            "eligible": eligible,
        })

    by_pitcher = {}
    for record in prepared:
        by_pitcher.setdefault(record["row"]["player_name"], []).append(record)
    primary = {
        pitcher: _select_primary(records)
        for pitcher, records in by_pitcher.items()
    }

    logged = 0
    selected_count = 0
    for record in prepared:
        src = record["row"]
        key = record["key"]
        selected = record["eligible"] and primary[src["player_name"]] == key
        if selected:
            selected_count += 1
        if not record["stress_complete"]:
            suppression_reason = "stress_evidence_incomplete"
        elif not record["verdict_bet"]:
            suppression_reason = "llm_verdict_pass"
        elif record["confidence"] < MIN_CONFIDENCE:
            suppression_reason = "llm_confidence_below_gate"
        elif src["market_p_over"] is None:
            suppression_reason = "market_probability_missing"
        elif not record["eligible"]:
            suppression_reason = "edge_below_market_specific_gate"
        elif not selected:
            suppression_reason = "secondary_correlated_market"
        else:
            suppression_reason = None

        books = lines.get(
            (llm._norm(src["player_name"]), src["market"], float(src["line"])),
            [],
        )
        live_row = llm._priced_row(
            game_date,
            {"game_pk": src["game_pk"]},
            {"name": src["player_name"], "id": src["player_id"]},
            src["market"],
            float(src["line"]),
            record["p_over"],
            books,
            src["market_p_over"],
            run_id=run_id,
            source="judge",
            rationale=json.dumps({
                "action": "bet" if selected else "pass",
                "rationale": record["rationale"],
                "confidence": record["confidence"],
                "lane_mode": "live",
                "decision_basis": "llm_only",
                "influenced_by_monte_carlo": False,
                "influenced_by_model": False,
                "stress_complete": record["stress_complete"],
                "primary_market": selected,
                "suppression_reason": suppression_reason,
            }),
            suppress_bet=not selected,
        )
        db.insert_prediction(conn, live_row)
        logged += 1

    _record_lane(
        conn, run_id, game_date, "live_decision", "live", "complete", logged,
        {
            "selected_primary_bets": selected_count,
            "decision_basis": "llm_only",
            "influenced_by_monte_carlo": False,
            "influenced_by_model": False,
        },
    )
    conn.close()
    print(f"[live] promoted {logged} LLM-only decisions ({selected_count} primary bets)")
    return logged
