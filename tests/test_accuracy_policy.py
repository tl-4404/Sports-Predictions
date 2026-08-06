"""Compatibility entry point for the consolidated MLB audit regressions."""
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).with_name('test_merged_audit_policy.py')),
    run_name='__main__',
)
