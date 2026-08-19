"""Focused tests for the LLM-only live decision gate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MLB = ROOT / "mlb"
if str(MLB) not in sys.path:
    sys.path.insert(0, str(MLB))

import live_decision


def test_live_probability_cap_is_symmetric():
    assert live_decision._cap_side_probability(0.81) == 0.64
    assert live_decision._cap_side_probability(0.19) == 0.36
    assert live_decision._cap_side_probability(0.57) == 0.57


def test_market_specific_edge_hurdles_are_conservative():
    assert live_decision._required_edge("pitcher_strikeouts", "over") == 0.06
    assert live_decision._required_edge("pitcher_strikeouts", "under") == 0.06
    assert live_decision._required_edge("pitcher_outs", "over") == 0.06
    assert live_decision._required_edge("pitcher_outs", "under") == 0.08


def test_strikeout_stress_evidence_is_required():
    robust = (
        "WORKLOAD-STATES: short/normal/extended. "
        "K-CONVERSION-STRESS: lower/base/higher all keep the side. "
        "COUNTERARGUMENT: workload could extend. VERDICT: BET"
    )
    fragile = (
        "WORKLOAD-STATES: short/normal/extended. "
        "COUNTERARGUMENT: workload could extend. VERDICT: BET"
    )
    assert live_decision._stress_complete("pitcher_strikeouts", robust)
    assert not live_decision._stress_complete("pitcher_strikeouts", fragile)


def test_outs_tail_stress_evidence_is_required():
    robust = (
        "WORKLOAD-STATES: short/normal/extended. "
        "OUTS-TAIL-STRESS: side survives all plausible states. "
        "COUNTERARGUMENT: efficiency could move the tail. VERDICT: BET"
    )
    wrong_stress = (
        "WORKLOAD-STATES: short/normal/extended. "
        "K-CONVERSION-STRESS: irrelevant here. "
        "COUNTERARGUMENT: efficiency could move the tail. VERDICT: BET"
    )
    assert live_decision._stress_complete("pitcher_outs", robust)
    assert not live_decision._stress_complete("pitcher_outs", wrong_stress)


def test_only_explicit_bet_verdict_is_promotable():
    assert live_decision._verdict_is_bet("VERDICT: BET")
    assert not live_decision._verdict_is_bet("VERDICT: PASS")
    assert not live_decision._verdict_is_bet("lean over")


def test_primary_selection_prefers_robust_edge_not_raw_probability():
    records = [
        {
            "key": ("pitcher_strikeouts", 5.5),
            "eligible": False,
            "edge": 0.14,
        },
        {
            "key": ("pitcher_outs", 15.5),
            "eligible": True,
            "edge": 0.09,
        },
        {
            "key": ("pitcher_strikeouts", 4.5),
            "eligible": True,
            "edge": 0.07,
        },
    ]
    assert live_decision._select_primary(records) == ("pitcher_outs", 15.5)
