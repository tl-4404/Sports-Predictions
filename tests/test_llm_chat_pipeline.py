"""Regression tests for the preserved-chat MLB LLM lane."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MLB = ROOT / "mlb"
if str(MLB) not in sys.path:
    sys.path.insert(0, str(MLB))


def test_saved_policy_installs_recalibrated_audit():
    import llm_saved_pipeline as policy

    assert policy.PIPELINE_VERSION == "chat-preserved-v4.1"
    prompt = policy.base.PREDICT_SYSTEM
    assert "WORKLOAD-STATES" in prompt
    assert "PLUS-4-BF" in prompt
    assert "MODEL-DISAGREEMENT" in prompt
    assert "FOUR additional batters faced" in prompt
    assert "within 0.25 Ks" in prompt
    assert "do NOT see prices" in prompt


def test_judge_remains_bounded_market_aware_and_conservative():
    import llm_saved_pipeline as policy

    prompt = policy.base.JUDGE_SYSTEM
    assert "de-vigged market" in prompt
    assert "ONE primary bet per pitcher" in prompt
    assert policy.base.JUDGE_BAND == 0.03
    assert policy.base.W_SIM == 0.75
    assert policy.base.MIN_EDGE == 0.06
    assert policy.base._judge_pitcher is policy._judge_pitcher_v4


def test_probability_cap_applies_to_either_side():
    import llm_saved_pipeline as policy

    assert policy._cap_side_probability(0.80) == 0.64
    assert policy._cap_side_probability(0.20) == 0.36
    assert policy._cap_side_probability(0.57) == 0.57


def test_dynamic_edge_thresholds_are_market_specific():
    import llm_saved_pipeline as policy

    assert policy._required_edge("pitcher_strikeouts", "under") == 0.08
    assert policy._required_edge("pitcher_strikeouts", "over") == 0.08
    assert policy._required_edge("pitcher_outs", "under") == 0.10
    assert policy._required_edge("pitcher_outs", "over") == 0.06
    assert policy._required_edge("pitcher_fantasy_score", "under") == 0.10


def _record(market, p_over, market_p, sim_p=None, llm_p=None):
    record = {
        "key": (market, 5.5),
        "action": "bet",
        "p_over": p_over,
        "market_p": market_p,
    }
    if sim_p is not None:
        record["sim"] = {"p_over": sim_p}
    if llm_p is not None:
        record["llm"] = {"p_over": llm_p}
    return record


def test_strikeout_play_requires_sixty_percent_side_probability():
    import llm_saved_pipeline as policy

    eligible, reason = policy._eligibility(
        _record("pitcher_strikeouts", 0.41, 0.52, 0.40, 0.42)
    )
    assert not eligible
    assert reason == "side_probability_below_v4_1_floor"


def test_outs_under_requires_sixty_percent_side_probability():
    import llm_saved_pipeline as policy

    eligible, reason = policy._eligibility(
        _record("pitcher_outs", 0.41, 0.52, 0.40, 0.42)
    )
    assert not eligible
    assert reason == "side_probability_below_v4_1_floor"


def test_lane_side_disagreement_suppresses_official_play():
    import llm_saved_pipeline as policy

    eligible, reason = policy._eligibility(
        _record("pitcher_strikeouts", 0.39, 0.50, 0.44, 0.515)
    )
    assert not eligible
    assert reason == "lane_side_disagreement"


def test_large_lane_probability_gap_suppresses_official_play():
    import llm_saved_pipeline as policy

    eligible, reason = policy._eligibility(
        _record("pitcher_outs", 0.39, 0.50, 0.32, 0.47)
    )
    assert not eligible
    assert reason == "lane_probability_gap"


def test_strong_aligned_k_under_remains_eligible():
    import llm_saved_pipeline as policy

    eligible, reason = policy._eligibility(
        _record("pitcher_strikeouts", 0.39, 0.50, 0.38, 0.40)
    )
    assert eligible
    assert reason is None


def test_aligned_outs_over_uses_normal_market_floor():
    import llm_saved_pipeline as policy

    record = _record("pitcher_outs", 0.58, 0.50, 0.57, 0.59)
    record["key"] = ("pitcher_outs", 15.5)
    eligible, reason = policy._eligibility(record)
    assert eligible
    assert reason is None


def test_only_strongest_eligible_market_becomes_primary():
    import llm_saved_pipeline as policy

    records = [
        {
            "key": ("pitcher_strikeouts", 4.5),
            "action": "bet",
            "p_over": 0.43,
            "market_p": 0.50,
        },
        {
            "key": ("pitcher_outs", 15.5),
            "action": "bet",
            "p_over": 0.40,
            "market_p": 0.50,
        },
        {
            "key": ("pitcher_outs", 17.5),
            "action": "bet",
            "p_over": 0.44,
            "market_p": 0.50,
        },
    ]

    assert policy._select_primary_key(records) == ("pitcher_outs", 15.5)


def test_no_primary_when_edges_do_not_clear_v4_hurdles():
    import llm_saved_pipeline as policy

    records = [
        {
            "key": ("pitcher_strikeouts", 4.5),
            "action": "bet",
            "p_over": 0.45,
            "market_p": 0.50,
        },
        {
            "key": ("pitcher_outs", 15.5),
            "action": "bet",
            "p_over": 0.43,
            "market_p": 0.50,
        },
    ]

    assert policy._select_primary_key(records) is None


def test_batx_mode_defaults_to_bootstrap_gate(monkeypatch):
    import policy_flags

    monkeypatch.delenv(policy_flags.BATX_MODE_ENV, raising=False)
    assert policy_flags.batx_mode() == "bootstrap_gate"


def test_batx_mode_supports_advisory_and_off(monkeypatch):
    import policy_flags

    monkeypatch.setenv(policy_flags.BATX_MODE_ENV, "advisory")
    assert policy_flags.batx_mode() == "advisory"
    monkeypatch.setenv(policy_flags.BATX_MODE_ENV, "off")
    assert policy_flags.batx_mode() == "off"


def test_batx_mode_rejects_unknown_value(monkeypatch):
    import policy_flags

    monkeypatch.setenv(policy_flags.BATX_MODE_ENV, "primary_model")
    with pytest.raises(ValueError):
        policy_flags.batx_mode()


def test_chat_runner_applies_external_model_policy():
    import llm_chat_runner as runner

    assert runner.BATX_MODE in {"bootstrap_gate", "advisory", "off"}
    assert runner.policy.BATX_MODE == runner.BATX_MODE


def test_chat_runner_refreshes_before_predict_and_judge(monkeypatch):
    import llm_chat_runner as runner

    calls = []
    monkeypatch.setattr(
        runner,
        "refresh_facts",
        lambda game_date=None: calls.append(("facts", game_date)),
    )
    monkeypatch.setattr(
        runner.policy.base,
        "llm_predict_stage",
        lambda game_date=None: calls.append(("predict", game_date)),
    )
    monkeypatch.setattr(
        runner.policy.base,
        "judge_stage",
        lambda game_date=None: calls.append(("judge", game_date)) or 7,
    )

    result = runner.run_all("2026-07-10")

    assert result == 7
    assert calls == [
        ("facts", "2026-07-10"),
        ("predict", "2026-07-10"),
        ("judge", "2026-07-10"),
    ]
