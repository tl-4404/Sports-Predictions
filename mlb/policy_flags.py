"""Feature flags for external projection systems used by the MLB judge.

THE BAT X is useful while SportsAid's own model is still accumulating a clean,
priced and graded sample, but it must not become a permanent crutch. This module
makes that dependency explicit and removable.

Environment variable
--------------------
SPORTSAID_BATX_MODE controls how THE BAT X disagreement is treated:

``bootstrap_gate`` (default)
    Current development mode. A large disagreement is a pass/manual-review signal
    unless a verified current input explains it. THE BAT X still never supplies the
    base projection or directly chooses a side.

``advisory``
    Standalone-model mode. THE BAT X is logged as an external benchmark and may
    reduce confidence, but cannot independently veto a play.

``off``
    Ablation mode. THE BAT X is excluded from the decision prompt. Use this to
    measure how the standalone pipeline performs without the external benchmark.
"""
from __future__ import annotations

import os

BATX_MODE_ENV = "SPORTSAID_BATX_MODE"
VALID_BATX_MODES = {"bootstrap_gate", "advisory", "off"}
DEFAULT_BATX_MODE = "bootstrap_gate"


def batx_mode() -> str:
    """Return the validated external-model mode."""
    mode = os.getenv(BATX_MODE_ENV, DEFAULT_BATX_MODE).strip().lower()
    if mode not in VALID_BATX_MODES:
        raise ValueError(
            f"Invalid {BATX_MODE_ENV}={mode!r}; expected one of "
            f"{sorted(VALID_BATX_MODES)}"
        )
    return mode


def apply(policy) -> str:
    """Apply the configured THE BAT X role to the loaded chat policy.

    The v4 policy is authored in bootstrap-gate mode. Advisory and off modes remove
    its ability to act as a standalone veto while preserving the rest of the audit.
    The function is idempotent for a given process.
    """
    mode = batx_mode()
    previous = getattr(policy, "BATX_MODE", None)
    if previous == mode:
        return mode

    if mode == "advisory":
        policy.base.PREDICT_SYSTEM = policy.base.PREDICT_SYSTEM.replace(
            "pass/review signals unless a verified input explains the gap",
            "diagnostic signals that should reduce confidence but cannot independently veto a play",
        )
        policy.base.JUDGE_SYSTEM = policy.base.JUDGE_SYSTEM.replace(
            "as a\n  pass/review signal unless a verified current input explains it",
            "as an external diagnostic. It may reduce confidence but cannot independently veto a play",
        )
        role_note = (
            "THE BAT X MODE — advisory: external projections are a benchmark only; "
            "the standalone model, verified news and market evidence control the decision."
        )
    elif mode == "off":
        policy.base.PREDICT_SYSTEM = policy.base.PREDICT_SYSTEM.replace(
            "8. Identify THE BAT X/model disagreement. Differences above 0.75 Ks or 1.5 outs are\n"
            "   pass/review signals unless a verified input explains the gap.\n",
            "8. Do not use external projection systems in this ablation run.\n",
        )
        policy.base.JUDGE_SYSTEM = policy.base.JUDGE_SYSTEM.replace(
            "- treat THE BAT X disagreement above 0.75 Ks, 1.5 outs or 3.5 fantasy points as a\n"
            "  pass/review signal unless a verified current input explains it;\n",
            "- ignore THE BAT X and other external projection systems for this ablation run;\n",
        )
        role_note = (
            "THE BAT X MODE — off: external projections are excluded for standalone-model "
            "ablation and calibration testing."
        )
    else:
        role_note = (
            "THE BAT X MODE — bootstrap_gate: a large disagreement triggers review, not an "
            "automatic reversal. THE BAT X never supplies the base forecast or chosen side."
        )

    policy.base.PREDICT_SYSTEM += "\n\n" + role_note
    policy.base.JUDGE_SYSTEM += "\n\n" + role_note
    policy.BATX_MODE = mode
    return mode
