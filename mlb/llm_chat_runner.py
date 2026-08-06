"""Fresh-facts execution wrapper for the merged chat-aligned MLB policy."""
from __future__ import annotations

import json

import data_today
import db
import llm_saved_pipeline as policy
import policy_flags
from timeutils import slate_today

BATX_MODE = policy_flags.apply(policy)


def refresh_facts(game_date=None):
    game_date = game_date or slate_today()
    conn = db.connect(); policy.base.ensure_facts_schema(conn)
    inserted = 0
    for game in data_today.get_schedule(game_date):
        try:
            facts = policy.base.fetch_facts(game)
        except Exception as exc:
            print(f"[llm.chat.facts] game {game['game_pk']}: {exc}")
            continue
        conn.execute(
            "INSERT INTO llm_facts (game_date,game_pk,created_at,facts_json) VALUES (?,?,?,?)",
            (game_date,game["game_pk"],db.utcnow(),json.dumps(facts)),
        )
        conn.commit(); inserted += 1
    conn.close(); print(f"[llm.chat.facts] refreshed {inserted} games")
    return inserted


def run_all(game_date=None):
    print(f"[llm.chat] external model mode: BAT X={BATX_MODE}")
    refresh_facts(game_date)
    policy.base.llm_predict_stage(game_date)
    return policy.base.judge_stage(game_date)


if __name__ == "__main__":
    import sys
    run_all(sys.argv[1] if len(sys.argv)>1 else None)
