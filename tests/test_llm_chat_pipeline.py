"""Regression tests for the merged chat-aligned MLB LLM lane."""
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MLB = ROOT / "mlb"
if str(MLB) not in sys.path:
    sys.path.insert(0, str(MLB))


def test_saved_policy_installs_merged_audit():
    import decision
    import llm_saved_pipeline as policy

    assert policy.PIPELINE_VERSION == decision.POLICY_VERSION
    prompt = policy.base.PREDICT_SYSTEM
    assert "short, normal, and extended" in prompt
    assert "+2 BF" in prompt and "+4 BF" in prompt
    assert "skill-under" in prompt and "workload-under" in prompt
    assert "not sportsbook prices" in prompt


def test_judge_is_75_25_bounded_and_erode_only():
    import llm_saved_pipeline as policy

    prompt = policy.base.JUDGE_SYSTEM
    assert "75% sim" in prompt
    assert "three percentage points" in prompt
    assert "never strengthen or flip" in prompt
    assert policy.base.JUDGE_BAND == 0.03
    assert policy.base.W_SIM == 0.75


def test_chat_runner_refreshes_before_predict_and_judge():
    import llm_chat_runner as runner

    calls = []
    with (
        patch.object(runner, "refresh_facts",
                     side_effect=lambda game_date=None: calls.append(("facts", game_date))),
        patch.object(runner.policy.base, "llm_predict_stage",
                     side_effect=lambda game_date=None: calls.append(("predict", game_date))),
        patch.object(runner.policy.base, "judge_stage",
                     side_effect=lambda game_date=None: (calls.append(("judge", game_date)) or 7)),
    ):
        result = runner.run_all("2026-07-10")
    assert result == 7
    assert calls == [("facts","2026-07-10"),("predict","2026-07-10"),("judge","2026-07-10")]


if __name__ == "__main__":
    for test in (test_saved_policy_installs_merged_audit,
                 test_judge_is_75_25_bounded_and_erode_only,
                 test_chat_runner_refreshes_before_predict_and_judge):
        test(); print(f"  [ok] {test.__name__}")
    print("\nAll checks passed.")
