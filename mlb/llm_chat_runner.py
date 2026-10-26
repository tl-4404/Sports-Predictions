"""Execution wrapper for the chat-aligned MLB LLM policy.

Unlike the legacy facts stage, this runner refreshes every game on every invocation.
Snapshots remain append-only, and ``llm._facts_for`` automatically selects the newest
row. That prevents an early-morning research packet from silently becoming stale by
the confirmed-lineup rerun.
"""
from __future__ import annotations

import json
from datetime import date

import data_today
import db
import llm_saved_pipeline as policy
import policy_flags

# Default is bootstrap_gate while the standalone model accumulates a larger clean
# sample. Set SPORTSAID_BATX_MODE=advisory as calibration matures, or =off for
# external-model ablation testing.
BATX_MODE = policy_flags.apply(policy)


def refresh_facts(game_date=None) -> int:
    game_date = game_date or date.today().isoformat()
    conn = db.connect()
    policy.base.ensure_facts_schema(conn)
    games = data_today.get_schedule(game_date)
    inserted = 0

    for game in games:
        try:
            facts = policy.base.fetch_facts(game)
        except Exception as exc:
            print(f"[llm.chat.facts] game {game['game_pk']}: {exc}")
            continue
        conn.execute(
            "INSERT INTO llm_facts (game_date, game_pk, created_at, facts_json) "
            "VALUES (?, ?, ?, ?)",
            (game_date, game["game_pk"], db.utcnow(), json.dumps(facts)),
        )
        conn.commit()
        inserted += 1

    conn.close()
    print(f"[llm.chat.facts] refreshed {inserted} games")
    return inserted


def run_all(game_date=None, refresh_inputs=True):
    """Fresh chat-standard research -> independent prediction -> bounded judge."""
    print(f"[llm.chat] external model mode: BAT X={BATX_MODE}")
    if refresh_inputs:
        import odds_props
        import predict
        if odds_props.refresh(game_date) <= 0:
            print("[llm.chat] no refreshed prices; publication aborted")
            return 0
        if predict.predict_date(game_date) <= 0:
            print("[llm.chat] no fresh shadow forecasts; publication aborted")
            return 0
    refresh_facts(game_date)
    policy.base.llm_predict_stage(game_date)
    return policy.base.judge_stage(game_date)


if __name__ == "__main__":
    import sys

    run_all(sys.argv[1] if len(sys.argv) > 1 else None)
