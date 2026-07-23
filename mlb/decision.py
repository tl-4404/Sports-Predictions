"""One accuracy-first decision policy shared by every MLB prediction lane.

The model is allowed to publish a probability before it is trusted with a
bet.  Automatic bets require:

* a complete, pre-game data packet;
* a fresh, two-book market anchor;
* at least ``MIN_VALIDATION_ROWS`` unique, priced, out-of-sample grades from
  this exact policy version;
* raw-model Brier score better than the market over the validation window; and
* a calibrated discrepancy between ``MIN_EDGE`` and ``MAX_AUTO_GAP``.

The narrow discrepancy band is intentional.  Small differences are noise and
large differences are more likely to be a bad role, line, or data assumption
than a free lunch.  Large gaps remain visible for manual investigation but are
never turned into an automatic wager.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

POLICY_VERSION = "accuracy_v2"
MIN_EDGE = float(os.getenv("MLB_MIN_EDGE", "0.06"))
MIN_MODEL_GAP = float(os.getenv("MLB_MIN_MODEL_GAP", "0.05"))
MAX_AUTO_GAP = float(os.getenv("MLB_MAX_AUTO_GAP", "0.08"))
MIN_BOOKS = int(os.getenv("MLB_MIN_BOOKS", "2"))
MIN_VALIDATION_ROWS = int(os.getenv("MLB_MIN_VALIDATION_ROWS", "50"))
MAX_KELLY = float(os.getenv("MLB_MAX_KELLY", "0.01"))


@dataclass(frozen=True)
class Decision:
    """Canonical output consumed by simulator, LLM, judge, and DFS lanes."""

    model_p_over: float
    decision_p_over: float
    market_p_over: Optional[float]
    edge_over: Optional[float]
    edge_under: Optional[float]
    bet_side: Optional[str]
    hold_reasons: tuple[str, ...]
    policy_version: str = POLICY_VERSION

    @property
    def eligible(self) -> bool:
        return self.bet_side in {"over", "under"}


def _clip(p: float) -> float:
    return min(0.999, max(0.001, float(p)))


def complete_lineup(batter_ids: Iterable[int] | None) -> bool:
    """True only for nine distinct, populated batting-order slots."""
    ids = [int(value) for value in (batter_ids or []) if value]
    return len(ids) >= 9 and len(set(ids[:9])) == 9


def validation_reason(validation: dict | None) -> Optional[str]:
    """Return a fail-closed validation reason, or ``None`` when proven."""
    validation = validation or {}
    n = int(validation.get("sample_size") or 0)
    if n < MIN_VALIDATION_ROWS:
        return (
            f"paper mode: {n}/{MIN_VALIDATION_ROWS} unique priced "
            f"{POLICY_VERSION} grades"
        )
    model_brier = validation.get("model_brier")
    market_brier = validation.get("market_brier")
    if model_brier is None or market_brier is None:
        return "paper mode: model-vs-market Brier comparison unavailable"
    if float(model_brier) >= float(market_brier):
        return (
            "market-only: raw model has not beaten the de-vigged market "
            f"(Brier {float(model_brier):.4f} vs {float(market_brier):.4f})"
        )
    average_clv = validation.get("average_clv")
    if average_clv is None or float(average_clv) <= 0:
        return (
            "market-only: paper selections have not produced positive "
            "entry-to-close CLV"
        )
    return None


def decide(
    model_p_over: float,
    market_p_over: Optional[float],
    *,
    book_count: int,
    validation: dict | None,
    quality_reasons: Iterable[str] = (),
    allow_auto: bool = True,
    required_side: Optional[str] = None,
) -> Decision:
    """Apply the canonical automatic-bet gate to a probability.

    ``required_side`` lets an adversarial downstream judge preserve or veto an
    already-approved simulator side.  It can never create or flip a play.
    """
    model_p = _clip(model_p_over)
    reasons = [str(reason) for reason in quality_reasons if reason]

    if market_p_over is None:
        reasons.append("no live two-way market anchor")
        return Decision(
            model_p, model_p, None, None, None, None, tuple(dict.fromkeys(reasons))
        )

    market_p = _clip(market_p_over)
    if int(book_count) < MIN_BOOKS:
        reasons.append(f"only {int(book_count)} priced book(s); need {MIN_BOOKS}")
    if not allow_auto:
        reasons.append("analysis-only lane cannot create a wager")

    invalid = validation_reason(validation)
    if invalid:
        reasons.append(invalid)

    raw_gap = model_p - market_p
    abs_gap = abs(raw_gap)
    if abs_gap < MIN_MODEL_GAP:
        reasons.append(
            f"model/market gap {abs_gap:.1%} is noise below {MIN_MODEL_GAP:.1%}"
        )
    elif abs_gap > MAX_AUTO_GAP:
        reasons.append(
            f"model/market gap {abs_gap:.1%} exceeds {MAX_AUTO_GAP:.1%}; "
            "freeze for data/role investigation"
        )
    elif abs_gap < MIN_EDGE:
        reasons.append(
            f"actionable edge {abs_gap:.1%} is below {MIN_EDGE:.1%}"
        )

    candidate_side = "over" if raw_gap > 0 else "under"
    if required_side and candidate_side != required_side:
        reasons.append(
            f"downstream lane would flip approved {required_side} side"
        )

    # A held row is market-anchored for decision/Kelly purposes while retaining
    # the model probability separately for paper grading and diagnostics.
    if reasons:
        decision_p = market_p
        return Decision(
            model_p,
            decision_p,
            market_p,
            0.0,
            0.0,
            None,
            tuple(dict.fromkeys(reasons)),
        )

    decision_p = model_p
    edge_over = decision_p - market_p
    edge_under = -edge_over
    bet_side = candidate_side
    return Decision(
        model_p,
        decision_p,
        market_p,
        edge_over,
        edge_under,
        bet_side,
        (),
    )


def erode_only(
    simulator_p_over: float,
    requested_p_over: float,
    approved_side: Optional[str],
    band: float,
) -> float:
    """Clamp a judge adjustment so it may only weaken an approved edge."""
    base = _clip(simulator_p_over)
    requested = min(max(_clip(requested_p_over), base - band), base + band)
    if approved_side == "over":
        return min(base, requested)
    if approved_side == "under":
        return max(base, requested)
    return base
