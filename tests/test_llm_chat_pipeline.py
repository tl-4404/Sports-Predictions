"""Regression tests for the preserved-chat MLB LLM lane."""
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MLB = ROOT / "mlb"
if str(MLB) not in sys.path:
    sys.path.insert(0, str(MLB))


def test_saved_policy_installs_burke_level_audit():
    import llm_saved_pipeline as policy

    assert policy.PIPELINE_VERSION == "accuracy-v4.0"
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
    assert "never strengthen" in prompt
    assert policy.base.JUDGE_BAND == 0.03
    assert policy.base.W_SIM == 1.00


def test_chat_runner_refreshes_before_predict_and_judge():
    import llm_chat_runner as runner

    calls = []
    with (
        patch.object(
            runner,
            "refresh_facts",
            side_effect=lambda game_date=None: calls.append(("facts", game_date)),
        ),
        patch.object(
            runner.policy.base,
            "llm_predict_stage",
            side_effect=lambda game_date=None: calls.append(
                ("predict", game_date)
            ),
        ),
        patch.object(
            runner.policy.base,
            "judge_stage",
            side_effect=lambda game_date=None: (
                calls.append(("judge", game_date)) or 7
            ),
        ),
    ):
        result = runner.run_all("2026-07-10")

    assert result == 7
    assert calls == [
        ("facts", "2026-07-10"),
        ("predict", "2026-07-10"),
        ("judge", "2026-07-10"),
    ]


if __name__ == "__main__":
    tests = [
        test_saved_policy_installs_burke_level_audit,
        test_judge_remains_bounded_and_market_aware,
        test_chat_runner_refreshes_before_predict_and_judge,
    ]
    for test in tests:
        test()
        print(f"  [ok] {test.__name__}")
    print("\nAll checks passed.")
