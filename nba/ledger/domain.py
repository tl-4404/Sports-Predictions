"""Pure validation and arithmetic shared by CLI, storage, and tests."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math


def utcnow():
    return datetime.now(timezone.utc)


def timestamp(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware timestamp is required")
    return value.astimezone(timezone.utc)


def decimal_value(value, *, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a price/probability")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Expected a finite decimal") from exc
    if not result.is_finite() or (minimum is not None and result < minimum) or (
        maximum is not None and result > maximum
    ):
        raise ValueError("Decimal is outside the permitted range")
    return result


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def full_game_market(market):
    return (market.get("sportsMarketTypeV2") == "SPORTS_MARKET_TYPE_MONEYLINE"
            and market.get("sportsMarketType") == "basketball_team_full_game_winner")


def market_start(event, market):
    # Fail closed on disagreements instead of guessing after a schedule change.
    start = timestamp(event.get("startTime"))
    if market.get("gameStartTime") and timestamp(market["gameStartTime"]) != start:
        raise ValueError("Event and market start times disagree")
    return start


def validate_forecast(value, evidence_ids):
    expected = {"long_win_probability", "rationale", "evidence_ids", "uncertainties", "abstain"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Forecast must have exactly the documented output fields")
    probability = decimal_value(value["long_win_probability"], minimum=0, maximum=1)
    if not isinstance(value["abstain"], bool):
        raise ValueError("abstain must be boolean")
    if not isinstance(value["rationale"], str) or not 1 <= len(value["rationale"]) <= 6000:
        raise ValueError("A bounded nonempty rationale is required")
    for field in ("evidence_ids", "uncertainties"):
        if (not isinstance(value[field], list) or len(value[field]) > 30
                or any(not isinstance(v, str) or not 1 <= len(v) <= 1500 for v in value[field])):
            raise ValueError(f"Invalid {field}")
    if not value["evidence_ids"] or not set(value["evidence_ids"]).issubset(evidence_ids):
        raise ValueError("Forecast references missing/unknown evidence")
    return {**value, "long_win_probability": float(probability)}


def paper_metrics(probability, price, payout=None, fee_per_share=None):
    p = decimal_value(probability, minimum=0, maximum=1)
    entry = decimal_value(price, minimum=0, maximum=1)
    if entry in (0, 1):
        raise ValueError("Entry price must be strictly between zero and one")
    fee = None if fee_per_share is None else decimal_value(fee_per_share, minimum=0)
    result = {"gross_ev_per_share": p - entry, "net_ev_per_share": None if fee is None else p - entry - fee,
              "gross_pnl_per_share": None, "net_pnl_per_share": None}
    if payout is not None:
        settlement = decimal_value(payout, minimum=0, maximum=1)
        result["gross_pnl_per_share"] = settlement - entry
        result["net_pnl_per_share"] = None if fee is None else settlement - entry - fee
    return result


def probability_scores(probability, outcome):
    p = float(decimal_value(probability, minimum=0, maximum=1))
    if outcome not in (0, 1):
        raise ValueError("Probability scores require a binary outcome")
    bounded = max(1e-15, min(1 - 1e-15, p))
    return {"brier": (p - outcome) ** 2,
            "log_loss": -(outcome * math.log(bounded) + (1 - outcome) * math.log(1 - bounded))}
