"""Adapter for reviewed, dated role/workload evidence.

Input is an upstream research/reviewer record, not a guess from pitch counts or
an unverified LLM string. No records are generated automatically by this module.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from integrity import timestamp

DEFAULT_PATH = Path(__file__).parent / "cache/usage_evidence.json"


def match(records, game, pitcher, now=None):
    now = now or datetime.now(timezone.utc)
    start = timestamp(game.get("game_start_time"))
    valid = []
    for evidence in records:
        if not isinstance(evidence, dict) or not isinstance(evidence.get("source_url"), str):
            continue
        if (evidence.get("game_pk"), evidence.get("player_id"), evidence.get("game_date")) != (
            game.get("game_pk"), pitcher.get("id"), game.get("game_date")):
            continue
        observed, reviewed = timestamp(evidence.get("observed_at")), timestamp(evidence.get("reviewed_at"))
        url = urlparse(evidence.get("source_url") or "")
        cap = evidence.get("pitch_cap")
        if (not start or not observed or not reviewed or not observed <= reviewed <= now < start
                or (now - observed).total_seconds() > 24 * 3600
                or evidence.get("reviewed") is not True
                or evidence.get("expected_role") not in ("starter", "opener", "bulk")
                or url.scheme != "https" or not url.netloc or not evidence.get("evidence")
                or (cap is not None and (type(cap) is not int or not 1 <= cap <= 150))):
            continue
        valid.append(evidence)
    return max(valid, key=lambda e: timestamp(e["reviewed_at"]), default=None)


def attach(game, pitcher, path=None):
    path = Path(path or os.getenv("MLB_USAGE_EVIDENCE_PATH", str(DEFAULT_PATH)))
    try:
        records = json.loads(path.read_text())
        if not isinstance(records, list):
            return dict(pitcher, role_verified=False)
        evidence = match(records, game, pitcher)
    except (OSError, ValueError, TypeError):
        evidence = None
    return dict(pitcher, role_verified=bool(evidence),
                pitch_cap=evidence.get("pitch_cap") if evidence else None,
                role_evidence_json=json.dumps(evidence) if evidence else None)
