"""Decision-time contracts shared by every prediction publisher.

Missing evidence is a HOLD. A prompt or a large probability cannot override it.
This is eligibility, not evidence that a forecasting model has a betting edge.
"""
from datetime import datetime, timezone
import math

MODEL_VERSION = "mlb-audit-v5"
MAX_LINE_AGE_MINUTES = 15


def timestamp(value):
    try:
        value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return value if value.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def publication_reasons(row, now=None):
    """Reasons a row cannot be an actionable, pregame recommendation."""
    now = now or datetime.now(timezone.utc)
    reasons = []
    if row.get("model_version") != MODEL_VERSION:
        reasons.append("legacy_or_unversioned")
    if row.get("model_status") != "active":
        reasons.append("shadow_model")
    if not row.get("game_pk") or not row.get("player_id"):
        reasons.append("missing_game_or_player_identity")
    if not row.get("lineup_confirmed"):
        reasons.append("lineup_unconfirmed")
    if not row.get("role_verified"):
        reasons.append("role_or_workload_unverified")
    start = timestamp(row.get("game_start_time"))
    decision = timestamp(row.get("created_at"))
    observed = timestamp(row.get("line_as_of"))
    if not start or not decision or not (decision <= now < start):
        reasons.append("not_current_pregame")
    if not observed or not decision or not (
        0 <= (decision - observed).total_seconds() <= MAX_LINE_AGE_MINUTES * 60
        and 0 <= (now - observed).total_seconds() <= MAX_LINE_AGE_MINUTES * 60
    ):
        reasons.append("stale_or_missing_price")
    if row.get("market_p_over") is None or not row.get("book"):
        reasons.append("unpriced")
    if not row.get("price_verified"):
        reasons.append("fixture_or_unverified_price_source")
    return reasons


def enforce(row):
    row = dict(row)
    for field in ("p_over", "raw_p_over", "p_push", "market_p_over"):
        value = row.get(field)
        if value is not None and (not math.isfinite(float(value)) or not 0 <= value <= 1):
            raise ValueError(f"invalid {field}")
    reasons = publication_reasons(row)
    if row.get("bet_side"):
        from betting_math import kelly_fraction
        side = row["bet_side"]
        if side not in ("over", "under"):
            raise ValueError("invalid bet_side")
        probability = row["p_over"] if side == "over" else 1 - row["p_over"]
        allowed = kelly_fraction(probability, row.get(side + "_odds"))
        if (row.get("kelly_frac") or 0) <= 0 or allowed <= 0:
            reasons.append("no_positive_price_ev")
        else:
            row["kelly_frac"] = min(row["kelly_frac"], allowed)
    row["eligibility_reason"] = ";".join(reasons) if reasons else "eligible"
    if reasons:
        row["bet_side"], row["kelly_frac"] = None, 0.0
    return row
