"""Canonical, deterministic decision policy for the merged v4.2/v5 MLB audits.

Forecasts and wagers are deliberately separate. Every lane may publish a pregame
probability, but only the final judge may mark a row official, and only after all
integrity, validation, market, uncertainty, and exposure gates pass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

POLICY_VERSION = "sportsaid-v5.0-merged-audit"
MIN_DIAGNOSTIC_ROWS = int(os.getenv("MLB_MIN_DIAGNOSTIC_ROWS", "50"))
MIN_VALIDATION_ROWS = int(os.getenv("MLB_MIN_VALIDATION_ROWS", "200"))
FULL_SEGMENT_ROWS = int(os.getenv("MLB_FULL_SEGMENT_ROWS", "500"))
MIN_BOOKS = int(os.getenv("MLB_MIN_BOOKS", "2"))
MAX_KELLY = float(os.getenv("MLB_MAX_KELLY", "0.01"))
MAX_MODEL_MARKET_GAP = float(os.getenv("MLB_MAX_AUTO_GAP", "0.12"))
MAX_LANE_GAP = float(os.getenv("MLB_MAX_LANE_GAP", "0.12"))
MAX_OFFICIAL_KS = int(os.getenv("MLB_MAX_OFFICIAL_KS", "2"))
SHADOW_MODE = os.getenv("MLB_SHADOW_MODE", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
OUTS_OFFICIAL_ENABLED = os.getenv("MLB_OUTS_OFFICIAL_ENABLED", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

# Compatibility constant used by older callers. Market-specific thresholds below
# are authoritative.
MIN_EDGE = float(os.getenv("MLB_MIN_EDGE", "0.06"))

MARKET_RULES = {
    ("pitcher_strikeouts", "over"): (0.08, 0.60),
    ("pitcher_strikeouts", "under"): (0.08, 0.60),
    ("pitcher_outs", "under"): (0.10, 0.60),
    ("pitcher_outs", "over"): (0.06, 0.57),
}


@dataclass(frozen=True)
class Decision:
    model_p_over: float
    decision_p_over: float
    market_p_over: Optional[float]
    edge_over: Optional[float]
    edge_under: Optional[float]
    bet_side: Optional[str]
    hold_reasons: tuple[str, ...]
    policy_version: str = POLICY_VERSION
    status: str = "pass"  # official | candidate | review | pass
    suppression_reason: Optional[str] = None
    rule_trace: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return self.status == "official" and self.bet_side in {"over", "under"}


def _clip(p: float) -> float:
    return min(0.999, max(0.001, float(p)))


def complete_lineup(batter_ids: Iterable[int] | None) -> bool:
    ids = [int(value) for value in (batter_ids or []) if value]
    return len(ids) >= 9 and len(set(ids[:9])) == 9


def market_requirements(market: Optional[str], side: str) -> tuple[float, float]:
    return MARKET_RULES.get((market, side), (MIN_EDGE, 0.57))


def validation_reason(validation: dict | None) -> Optional[str]:
    validation = validation or {}
    n = int(validation.get("sample_size") or 0)
    if n < MIN_VALIDATION_ROWS:
        return (
            f"shadow mode: {n}/{MIN_VALIDATION_ROWS} unique priced "
            f"{POLICY_VERSION} grades"
        )
    model_brier = validation.get("model_brier")
    market_brier = validation.get("market_brier")
    if model_brier is None or market_brier is None:
        return "shadow mode: model-vs-market Brier comparison unavailable"
    if float(model_brier) >= float(market_brier):
        return (
            "market-only: raw model has not beaten the de-vigged market "
            f"(Brier {float(model_brier):.4f} vs {float(market_brier):.4f})"
        )
    average_clv = validation.get("average_clv")
    if average_clv is None or float(average_clv) <= 0:
        return "market-only: canonical paper selections lack positive true CLV"
    clean_slates = int(validation.get("clean_slates") or 0)
    if clean_slates < 14:
        return f"shadow mode: {clean_slates}/14 clean timestamped slates"
    return None


def _quality_status(reasons: Sequence[str]) -> str:
    joined = " ".join(reasons).lower()
    review_terms = (
        "disagreement", "external projection", "bat x", "line moved",
        "workload", "role", "lineup", "stale", "unverified", "missing",
        "uncalibrated", "synthetic", "estimated",
    )
    return "review" if any(term in joined for term in review_terms) else "pass"


def decide(
    model_p_over: float,
    market_p_over: Optional[float],
    *,
    book_count: int,
    validation: dict | None,
    quality_reasons: Iterable[str] = (),
    allow_auto: bool = True,
    required_side: Optional[str] = None,
    market: Optional[str] = None,
    line_verified: bool = True,
    standard_tier: bool = True,
    lineup_confirmed: bool = True,
    starter_verified: bool = True,
    workload_verified: bool = True,
    uncertainty: float = 0.0,
    lane_gap: Optional[float] = None,
    external_near_line: bool = False,
    external_disagreement_explained: bool = False,
    force_shadow: Optional[bool] = None,
) -> Decision:
    """Apply the merged deterministic gate.

    ``required_side`` lets the final judge preserve or veto a simulator candidate.
    It can never create or flip a side. Uncertainty raises the edge threshold rather
    than being used as a vague subjective veto.
    """
    model_p = _clip(model_p_over)
    validation = validation or {}
    market = market or validation.get("market")
    reasons = [str(reason) for reason in quality_reasons if reason]
    trace = []

    if market_p_over is None:
        reasons.append("no live two-way market anchor")
        return Decision(model_p, model_p, None, None, None, None,
                        tuple(dict.fromkeys(reasons)), status="review",
                        suppression_reason="missing_market_anchor",
                        rule_trace=("market_anchor:fail",))

    market_p = _clip(market_p_over)
    raw_gap = model_p - market_p
    candidate_side = "over" if raw_gap > 0 else "under"
    side_p = model_p if candidate_side == "over" else 1.0 - model_p
    base_edge, min_side_p = market_requirements(market, candidate_side)
    uncertainty = min(1.0, max(0.0, float(uncertainty or 0.0)))
    required_edge = base_edge + 0.03 * uncertainty

    if int(book_count) < MIN_BOOKS:
        reasons.append(f"only {int(book_count)} priced book(s); need {MIN_BOOKS}")
        trace.append("two_book_anchor:fail")
    else:
        trace.append("two_book_anchor:pass")
    if not line_verified:
        reasons.append("exact platform line is stale, inferred, moved, or unverified")
        trace.append("exact_line:fail")
    else:
        trace.append("exact_line:pass")
    if not standard_tier:
        reasons.append("alternate payout tier is not eligible for standard-line policy")
        trace.append("standard_tier:fail")
    if not starter_verified:
        reasons.append("starter role is unverified")
        trace.append("starter_role:fail")
    if not lineup_confirmed:
        reasons.append("opposing starting lineup is projected or incomplete")
        trace.append("confirmed_lineup:fail")
    if not workload_verified:
        reasons.append("workload state lacks dated, verified evidence")
        trace.append("workload_evidence:fail")
    if market == "pitcher_outs" and not OUTS_OFFICIAL_ENABLED:
        reasons.append("pitcher outs remain shadow-only pending workload-state validation")
        trace.append("outs_validation:fail")
    if lane_gap is not None and abs(float(lane_gap)) > MAX_LANE_GAP:
        reasons.append(
            f"sim/LLM gap {abs(float(lane_gap)):.1%} exceeds {MAX_LANE_GAP:.1%}"
        )
        trace.append("lane_agreement:fail")
    if external_near_line and not external_disagreement_explained:
        reasons.append("external projection sits near the line without a unique verified counter-signal")
        trace.append("external_projection_review:fail")
    if abs(raw_gap) > MAX_MODEL_MARKET_GAP:
        reasons.append(
            f"model/market gap {abs(raw_gap):.1%} exceeds {MAX_MODEL_MARKET_GAP:.1%}; investigate data/role"
        )
        trace.append("max_gap:fail")
    if abs(raw_gap) + 1e-12 < required_edge:
        reasons.append(
            f"edge {abs(raw_gap):.1%} below {required_edge:.1%} "
            f"({base_edge:.1%} base + uncertainty penalty)"
        )
        trace.append("edge:fail")
    else:
        trace.append("edge:pass")
    if side_p + 1e-12 < min_side_p:
        reasons.append(
            f"side probability {side_p:.1%} below {min_side_p:.1%} bootstrap floor"
        )
        trace.append("side_probability:fail")
    else:
        trace.append("side_probability:pass")
    if required_side and candidate_side != required_side:
        reasons.append(f"downstream lane would flip approved {required_side} side")
        trace.append("side_preservation:fail")
    if not allow_auto:
        reasons.append("analysis-only lane cannot create an official wager")
        trace.append("lane_authority:shadow")

    invalid = validation_reason(validation)
    if invalid:
        reasons.append(invalid)
        trace.append("historical_validation:fail")
    else:
        trace.append("historical_validation:pass")

    shadow = SHADOW_MODE if force_shadow is None else bool(force_shadow)
    if shadow:
        reasons.append("production shadow mode is enabled")
        trace.append("production_mode:shadow")

    unique_reasons = tuple(dict.fromkeys(reasons))
    if unique_reasons:
        status = "candidate" if (
            candidate_side in {"over", "under"}
            and abs(raw_gap) >= required_edge
            and side_p >= min_side_p
            and line_verified
            and int(book_count) >= MIN_BOOKS
        ) else _quality_status(unique_reasons)
        suppression = unique_reasons[0]
        return Decision(
            model_p, market_p, market_p, 0.0, 0.0, None,
            unique_reasons, status=status, suppression_reason=suppression,
            rule_trace=tuple(trace),
        )

    return Decision(
        model_p, model_p, market_p, raw_gap, -raw_gap, candidate_side, (),
        status="official", suppression_reason=None, rule_trace=tuple(trace),
    )


def erode_only(
    simulator_p_over: float,
    requested_p_over: float,
    approved_side: Optional[str],
    band: float,
) -> float:
    base = _clip(simulator_p_over)
    requested = min(max(_clip(requested_p_over), base - band), base + band)
    if approved_side == "over":
        return min(base, requested)
    if approved_side == "under":
        return max(base, requested)
    return base


def select_primary_records(records: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Apply one-primary-per-pitcher and probation exposure limits.

    Returns ``(official, suppressed)``. The input records are not mutated.
    """
    ranked = sorted(
        (dict(row) for row in records if row.get("status") == "official"),
        key=lambda row: (
            -float(row.get("edge") or 0.0),
            -float(row.get("side_probability") or 0.0),
            str(row.get("player_name") or ""),
        ),
    )
    official, suppressed = [], []
    used_pitchers = set()
    k_count = 0
    for row in ranked:
        player = str(row.get("player_name") or "").lower()
        if player in used_pitchers:
            row["status"] = "candidate"
            row["suppression_reason"] = "secondary_correlated_market"
            suppressed.append(row)
            continue
        if row.get("market") == "pitcher_strikeouts" and k_count >= MAX_OFFICIAL_KS:
            row["status"] = "candidate"
            row["suppression_reason"] = "probation_daily_k_cap"
            suppressed.append(row)
            continue
        used_pitchers.add(player)
        if row.get("market") == "pitcher_strikeouts":
            k_count += 1
        official.append(row)
    return official, suppressed
