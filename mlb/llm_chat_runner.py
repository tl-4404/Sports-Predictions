"""Execution wrapper for the chat-aligned MLB LLM policy.

Fresh research is collected on every invocation.  The independent LLM predictor is
the live decision input; numerical simulator/model lanes remain shadow-only and are
not supplied to the live probability or ranking path.
"""
from __future__ import annotations

import json
from datetime import date

import data_today
import db
import live_decision
import llm_saved_pipeline as policy
import policy_flags

# Default is bootstrap_gate while the standalone pipeline accumulates a larger clean
# sample. External models remain diagnostic only and never supply the live base
# probability or chosen side.
BATX_MODE = policy_flags.apply(policy)

GRADE_GUARDRAIL = """

THREE-DAY GRADE GUARDRAIL (Aug. 16-18, 2026):
- Keep workload opportunity and strikeout conversion as separate uncertainty axes.
- For EVERY strikeout prop, include K-CONVERSION-STRESS: compare the base estimate
  with a materially lower and higher plausible K-conversion rate (about +/-10% in
  K odds/rate, adjusted when pitcher-specific evidence supports a wider range) while
  also considering short/normal/extended workload. A side that flips under a modest
  adverse conversion assumption is not a top play.
- For EVERY outs prop, include OUTS-TAIL-STRESS: explicitly test short, normal and
  extended workload. An under that wins only in the short state or an over that wins
  only in the extended state must be a pass.
- Rank robustness above raw point probability when the two disagree.
- The rationale must contain WORKLOAD-STATES, COUNTERARGUMENT, and either
  K-CONVERSION-STRESS or OUTS-TAIL-STRESS as appropriate, and must end with exactly
  VERDICT: BET or VERDICT: PASS.
"""

if "THREE-DAY GRADE GUARDRAIL" not in policy.base.PREDICT_SYSTEM:
    policy.base.PREDICT_SYSTEM += GRADE_GUARDRAIL


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


def run_all(game_date=None):
    """Fresh research -> independent LLM prediction -> LLM-only live promotion.

    The legacy bounded blend/judge remains available for historical comparison and
    ablation tests, but it is intentionally not invoked by the live runner because
    simulator/model outputs are shadow-only under the current operating policy.
    """
    print(f"[llm.chat] external model mode: BAT X={BATX_MODE}")
    refresh_facts(game_date)
    policy.base.llm_predict_stage(game_date)
    return live_decision.promote_llm_live(game_date)


if __name__ == "__main__":
    import sys

    run_all(sys.argv[1] if len(sys.argv) > 1 else None)
