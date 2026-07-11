"""Regression tests for the preserved-chat MLB LLM lane."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MLB = ROOT / "mlb"
if str(MLB) not in sys.path:
    sys.path.insert(0, str(MLB))


def test_saved_policy_installs_burke_level_audit():
    import llm_saved_pipeline as policy

    assert policy.PIPELINE_VERSION == "chat-preserved-v3.0"
    prompt = policy.base.PREDICT_SYSTEM
    assert "WHY-LINE-EXISTS" in prompt
    assert "COUNTERARGUMENT" in prompt
    assert "usage trajectory" in prompt
    assert "pitch-mix" in prompt
    assert "do NOT see prices" in prompt


def test_judge_remains_bounded_and_market_aware():
    import llm_saved_pipeline as policy

    prompt = policy.base.JUDGE_SYSTEM
    assert "de-vigged market" in prompt
    assert "numeric authority remains narrow" in prompt
    assert policy.base.JUDGE_BAND == 0.05
    assert policy.base.W_SIM == 0.70


def test_chat_runner_refreshes_before_predict_and_judge(monkeypatch):
    import llm_chat_runner as runner

    calls = []
    monkeypatch.setattr(runner, "refresh_facts", lambda game_date=None: calls.append(("facts", game_date)))
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
