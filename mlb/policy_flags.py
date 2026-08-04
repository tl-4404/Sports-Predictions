"""Feature flags for external projection systems used by the MLB judge.

THE BAT X is useful while SportsAid's own model is still accumulating a clean,
priced and graded sample, but it must not become a permanent crutch. This module
makes that dependency explicit, configurable and removable.

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

_PREDICT_GATE_TEXT = (
    "8. Identify THE BAT X/model disagreement. Differences above 0.75 Ks or 1.5 outs are\n"
    "   pass/review signals unless a verified input explains the gap."
)
_JUDGE_GATE_TEXT = (
    "- treat THE BAT X disagreement above 0.75 Ks, 1.5 outs or 3.5 fantasy points as a\n"
    "  pass/review signal unless a verified current input explains it;"
)


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
    Original prompts are retained so repeated calls or test-time mode changes are
    deterministic rather than cumulative.
    """
    mode = batx_mode()

    if not hasattr(policy, "_BATX_BASE_PREDICT_SYSTEM"):
        policy._BATX_BASE_PREDICT_SYSTEM = policy.base.PREDICT_SYSTEM
        policy._BATX_BASE_JUDGE_SYSTEM = policy.base.JUDGE_SYSTEM

    predict_system = policy._BATX_BASE_PREDICT_SYSTEM
    judge_system = policy._BATX_BASE_JUDGE_SYSTEM

    if mode == "advisory":
        predict_system = predict_system.replace(
            _PREDICT_GATE_TEXT,
            "8. Identify THE BAT X/model disagreement as a diagnostic signal. It may reduce\n"
            "   confidence but cannot independently veto or reverse a play.",
        )
        judge_system = judge_system.replace(
            _JUDGE_GATE_TEXT,
            "- treat THE BAT X disagreement as an external diagnostic that may reduce\n"
            "  confidence but cannot independently veto or reverse a play;",
        )
        role_note = (
            "THE BAT X MODE — advisory: external projections are a benchmark only; "
            "the standalone model, verified news and market evidence control the decision."
        )
    elif mode == "off":
        predict_system = predict_system.replace(
            _PREDICT_GATE_TEXT,
            "8. Do not use external projection systems in this ablation run.",
        )
        judge_system = judge_system.replace(
            _JUDGE_GATE_TEXT,
            "- ignore THE BAT X and other external projection systems for this ablation run;",
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

    policy.base.PREDICT_SYSTEM = predict_system + "\n\n" + role_note
    policy.base.JUDGE_SYSTEM = judge_system + "\n\n" + role_note
    policy.BATX_MODE = mode
    return mode
