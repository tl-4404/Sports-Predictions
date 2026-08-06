"""Immutable external-projection modes for the MLB audit pipeline."""
from __future__ import annotations

import os

BATX_MODE_ENV = "SPORTSAID_BATX_MODE"
VALID_BATX_MODES = {"bootstrap_gate", "advisory", "off"}
DEFAULT_BATX_MODE = "bootstrap_gate"


def batx_mode():
    mode = os.getenv(BATX_MODE_ENV, DEFAULT_BATX_MODE).strip().lower()
    if mode not in VALID_BATX_MODES:
        raise ValueError(
            f"Invalid {BATX_MODE_ENV}={mode!r}; expected one of {sorted(VALID_BATX_MODES)}"
        )
    return mode


def role_note(mode):
    if mode == "off":
        return (
            "EXTERNAL MODEL MODE — off. Do not use THE BAT X or another external "
            "projection in this ablation run."
        )
    if mode == "advisory":
        return (
            "EXTERNAL MODEL MODE — advisory. External projections are immutable, "
            "timestamped diagnostics only; they cannot veto, reverse, or supply the base forecast."
        )
    return (
        "EXTERNAL MODEL MODE — bootstrap_gate. A near-line or materially different "
        "external projection triggers review unless a unique, verified current input "
        "explains the difference. It never supplies the base forecast or chosen side."
    )


def apply(policy):
    mode = batx_mode()
    if not hasattr(policy, "_EXTERNAL_BASE_PREDICT_SYSTEM"):
        policy._EXTERNAL_BASE_PREDICT_SYSTEM = policy.base.PREDICT_SYSTEM
        policy._EXTERNAL_BASE_JUDGE_SYSTEM = policy.base.JUDGE_SYSTEM
    policy.base.PREDICT_SYSTEM = policy._EXTERNAL_BASE_PREDICT_SYSTEM + "\n\n" + role_note(mode)
    policy.base.JUDGE_SYSTEM = policy._EXTERNAL_BASE_JUDGE_SYSTEM + "\n\n" + role_note(mode)
    policy.BATX_MODE = mode
    return mode
