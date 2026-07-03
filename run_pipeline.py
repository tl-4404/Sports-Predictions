"""SportsAid daily pipeline entrypoint — hosting-agnostic.

Designed so any scheduler can drive it (cron, VPS, GitHub Actions):

    python run_pipeline.py                 # full daily run for today
    python run_pipeline.py --stage odds    # just refresh prop lines
    python run_pipeline.py --stage predict # just run the simulator
    python run_pipeline.py --stage llm     # facts -> Claude predictor -> judge
    python run_pipeline.py --stage grade   # grade yesterday's predictions
    python run_pipeline.py --date 2026-07-02

Suggested cron (adjust to taste):
    0 14 * * *  full run (morning ET — probables are posted)
    0 * * * *   --stage odds (hourly line snapshots → line history + CLV)
    30 22 * * * --stage predict (re-run once lineups are confirmed)
    0 11 * * *  --stage grade (grade yesterday)

The llm stage (facts / Claude predictor / judge) needs ANTHROPIC_API_KEY and
`pip install anthropic`; a full run skips it quietly when the key is absent.
"""
import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "mlb"))

STAGES = ("odds", "predict", "llm", "grade", "all")


def main():
    ap = argparse.ArgumentParser(description="SportsAid MLB props pipeline")
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default today; grade defaults to yesterday)")
    ap.add_argument("--sims", type=int, default=10000)
    args = ap.parse_args()

    failures = []

    if args.stage in ("odds", "all"):
        _run("odds", failures, lambda: __import__("odds_props").refresh(args.date))

    if args.stage in ("predict", "all"):
        _run("predict", failures,
             lambda: __import__("predict").predict_date(args.date, n_sims=args.sims))

    if args.stage == "llm" or (args.stage == "all" and os.getenv("ANTHROPIC_API_KEY")):
        _run("llm", failures, lambda: __import__("llm").run_all(args.date))
    elif args.stage == "all":
        print("[pipeline] skipping llm stage (no ANTHROPIC_API_KEY)")

    if args.stage in ("grade", "all"):
        grade_date = args.date if args.stage == "grade" else (
            (date.today() - timedelta(days=1)).isoformat())
        _run("grade", failures,
             lambda: __import__("grade").grade_date(grade_date))

    if failures:
        print(f"[pipeline] FAILED stages: {', '.join(failures)}")
        sys.exit(1)
    print("[pipeline] done")


def _run(name, failures, fn):
    print(f"[pipeline] === stage: {name} ===")
    try:
        fn()
    except Exception as e:
        print(f"[pipeline] stage {name} failed: {e}")
        failures.append(name)


if __name__ == "__main__":
    main()
