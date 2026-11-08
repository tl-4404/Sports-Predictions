"""Run offline suites in isolated processes; legacy checks mutate module globals.

Usage: python scripts/run_tests.py [--suite nba|mlb|shared|web]
Install requirements-dev.txt first. External network connections are blocked.
"""
import argparse
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "nba": ROOT / "nba/tests",
    "mlb": ROOT / "mlb/tests",
    "shared": ROOT / "shared/tests",
    "web": ROOT / "shared/web/tests",
}
BOOTSTRAP = """
import os, runpy, socket, sys
sys.path.insert(0, sys.argv[1])
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
def offline(*args, **kwargs):
    raise RuntimeError('Network disabled during offline tests')
socket.socket.connect = offline
socket.create_connection = offline
if sys.argv[3] == 'pytest':
    import pytest
    raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', sys.argv[2]]))
runpy.run_path(sys.argv[2], run_name='__main__')
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES)
    args = parser.parse_args()
    folders = [SUITES[args.suite]] if args.suite else SUITES.values()
    tests = sorted(p for folder in folders for p in folder.glob("test_*.py"))
    if not tests:
        parser.error("No test suites found")
    failed = []
    for path in tests:
        tree = ast.parse(path.read_text())
        use_pytest = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_") for node in ast.walk(tree)
        )
        print(f"\nRunning {path.relative_to(ROOT)}", flush=True)
        result = subprocess.run(
            [sys.executable, "-B", "-c", BOOTSTRAP, str(ROOT), str(path),
             "pytest" if use_pytest else "script"], cwd=ROOT,
        )
        if result.returncode:
            failed.append(str(path.relative_to(ROOT)))
    print(f"\n{len(tests) - len(failed)}/{len(tests)} offline suites passed.")
    if failed:
        print("Failed: " + ", ".join(failed))
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
