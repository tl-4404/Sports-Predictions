"""Finish wiring the consolidated MLB v4.2/v5 policy into repository defaults."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = "sportsaid-v5.0-merged-audit"


def patch_env() -> None:
    path = ROOT / ".env.example"
    if not path.exists():
        return
    text = path.read_text()
    text = text.replace("MLB_MIN_VALIDATION_ROWS=50", "MLB_MIN_VALIDATION_ROWS=200")
    text = text.replace("MLB_MAX_AUTO_GAP=0.08", "MLB_MAX_AUTO_GAP=0.12")
    marker = "# -- merged v4.2/v5 audit defaults --"
    if marker not in text:
        text += (
            "\n\n# -- merged v4.2/v5 audit defaults --\n"
            "MLB_SHADOW_MODE=1\n"
            "MLB_MIN_DIAGNOSTIC_ROWS=50\n"
            "MLB_MIN_VALIDATION_ROWS=200\n"
            "MLB_FULL_SEGMENT_ROWS=500\n"
            "MLB_OUTS_OFFICIAL_ENABLED=0\n"
            "MLB_MAX_OFFICIAL_KS=2\n"
            "MLB_MAX_LANE_GAP=0.12\n"
            "MLB_MAX_AUTO_GAP=0.12\n"
            "SPORTSAID_BATX_MODE=bootstrap_gate\n"
        )
    path.write_text(text)


def patch_makefile() -> None:
    path = ROOT / "Makefile"
    if not path.exists():
        return
    text = path.read_text()
    merged = "\t$(PY) tests/test_merged_audit_policy.py\n"
    if merged not in text:
        anchor = "\t$(PY) tests/test_accuracy_policy.py\n"
        text = text.replace(anchor, anchor + merged)
    path.write_text(text)


def patch_app() -> None:
    path = ROOT / "app.py"
    if path.exists():
        path.write_text(path.read_text().replace("accuracy_v2", POLICY))


def patch_readme() -> None:
    path = ROOT / "README.md"
    if not path.exists():
        return
    text = path.read_text()
    marker = "## Merged MLB audit policy"
    if marker not in text:
        text += (
            "\n\n## Merged MLB audit policy\n\n"
            "The active consolidated policy is `sportsaid-v5.0-merged-audit`. "
            "See `docs/MLB_PIPELINE_V5_MERGED_AUDIT.md` for the v4.2/v5 merge, "
            "shadow-mode promotion gates, immutable ledger, calibration, CLV, "
            "and refund/void settlement rules.\n"
        )
    path.write_text(text)


def replace_legacy_test_entrypoint() -> None:
    path = ROOT / "tests" / "test_accuracy_policy.py"
    path.write_text(
        '"""Compatibility entry point for the consolidated MLB audit regressions."""\n'
        "from pathlib import Path\n"
        "import runpy\n\n"
        "runpy.run_path(\n"
        "    str(Path(__file__).with_name('test_merged_audit_policy.py')),\n"
        "    run_name='__main__',\n"
        ")\n"
    )


def remove_one_shot_bootstrap() -> None:
    workflow = ROOT / ".github" / "workflows" / "apply-merged-mlb-audit.yml"
    if workflow.exists():
        workflow.unlink()
    Path(__file__).unlink()


patch_env()
patch_makefile()
patch_app()
patch_readme()
replace_legacy_test_entrypoint()
remove_one_shot_bootstrap()
print("Repository defaults now use the consolidated MLB v4.2/v5 policy")
