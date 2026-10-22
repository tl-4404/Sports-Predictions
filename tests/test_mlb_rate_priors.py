"""Focused tests for threshold-based MLB serving-rate priors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlb.providers import mlb_api


def _totals(k, bb, hit, n):
    return {"k": k, "bb": bb, "hit": hit, "n": n}


def test_full_sample_is_not_diluted():
    observed = _totals(k=60, bb=20, hit=40, n=200)
    rates = mlb_api._threshold_rates(
        observed, {"k": 0.10, "bb": 0.20, "hit": 0.30}, 200
    )
    assert rates == {"k": 0.30, "bb": 0.10, "hit": 0.20}


def test_small_sample_fills_only_gap_with_player_history():
    observed = _totals(k=40, bb=10, hit=20, n=100)
    history = {"k": 0.20, "bb": 0.08, "hit": 0.22}
    rates = mlb_api._threshold_rates(observed, history, 200)
    assert rates["k"] == 0.30
    assert rates["bb"] == 0.09
    assert rates["hit"] == 0.21


def test_thin_history_retains_a_league_safety_net():
    history = _totals(k=20, bb=2, hit=5, n=50)
    rates = mlb_api._credible_rates(history, 200)
    credibility = 0.25
    assert rates["k"] == credibility * 0.40 + (1 - credibility) * mlb_api.LEAGUE["k"]
    assert rates["bb"] == credibility * 0.04 + (1 - credibility) * mlb_api.LEAGUE["bb"]
    assert rates["hit"] == credibility * 0.10 + (1 - credibility) * mlb_api.LEAGUE["hit"]


def test_no_current_sample_uses_historical_prior():
    history = {"k": 0.28, "bb": 0.07, "hit": 0.19}
    assert mlb_api._threshold_rates(_totals(0, 0, 0, 0), history, 200) == history
