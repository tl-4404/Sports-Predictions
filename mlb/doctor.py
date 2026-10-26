"""Preflight readiness check for a live run.

Tells you, before you run the pipeline for real, exactly which pieces are
live-ready and which need attention: Python deps, the LLM key, StatsAPI
reachability, the configured odds source, PrizePicks, and the optional accuracy
caches. Every check is isolated so one failure can't mask the others.

    python mlb/doctor.py
    python run_pipeline.py --stage doctor

Exit code is non-zero when any hard FAIL is present (handy in scripts/CI).
"""
import os
from datetime import date
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "fail"


def _check_deps():
    missing = [m for m in ("numpy", "pandas", "requests")
               if not _importable(m)]
    if missing:
        return FAIL, f"missing {', '.join(missing)} — pip install -r requirements.txt"
    extras = "cloudscraper" if _importable("cloudscraper") else None
    note = "numpy, pandas, requests ok"
    return OK, note + (f"; cloudscraper present (Cloudflare-aware)" if extras
                       else "; cloudscraper absent (recommended for PrizePicks)")


def _check_anthropic():
    if not os.getenv("ANTHROPIC_API_KEY"):
        return WARN, "ANTHROPIC_API_KEY unset — LLM lanes will be skipped"
    if not _importable("anthropic"):
        return WARN, "key set but `anthropic` not installed (pip install anthropic)"
    return OK, "key present + anthropic installed"


def _check_statsapi():
    try:
        import data_today
        games = data_today.get_schedule(date.today().isoformat())
        return OK, f"reachable — {len(games)} games today"
    except Exception as e:
        return FAIL, f"unreachable: {_short(e)} (predict/prizepicks/grade need it)"


def _check_odds():
    import odds_props
    source = (os.getenv("ODDS_SOURCE") or "").strip().lower() or (
        "oddsapi" if os.getenv("ODDS_API_KEY") else "stub")
    if source in ("stub", "fixture"):
        return WARN, "ODDS_SOURCE=stub — fake fixture lines (set scraper|oddsapi for real)"
    if source in ("oddsapi", "the-odds-api", "api"):
        if not os.getenv("ODDS_API_KEY"):
            return FAIL, "ODDS_SOURCE=oddsapi but ODDS_API_KEY unset"
        return WARN, "oddsapi selected + key present (paid plan required for props)"
    try:
        rows = odds_props.get_fetcher().fetch()
        books = sorted({r["book"] for r in rows})
        if not rows:
            return FAIL, "scraper returned 0 rows (verify DK/FD ids; Cloudflare/geo?)"
        return OK, f"scraper live — {len(rows)} lines from {', '.join(books)}"
    except Exception as e:
        return FAIL, f"scraper failed: {_short(e)}"


def _check_prizepicks():
    import prizepicks
    if not os.getenv("PRIZEPICKS_LIVE"):
        return WARN, "PRIZEPICKS_LIVE unset — offline stub (set it for live PrizePicks)"
    try:
        rows = prizepicks.PrizePicksFetcher().fetch()
        if not rows:
            return FAIL, "0 projections (Cloudflare? league id? stat labels?)"
        return OK, f"live — {len(rows)} projections"
    except Exception as e:
        return FAIL, f"failed: {_short(e)}"


def _check_caches():
    base = Path(__file__).resolve().parent / "cache"
    present = [n for n in ("park_factors.json", "ump_factors.json",
                           "pitcher_form.json") if (base / n).exists()]
    if not present:
        return WARN, "no factor caches; neutral assumptions are not evidence of betting readiness"
    return WARN, f"present but require provenance/validation: {', '.join(present)}; CSW/velo challenger disabled"


def _check_model_evidence():
    return WARN, ("v5 numerical forecasts remain shadow; role/workload evidence and "
                  "fresh live prices are required for judge publication; no verified pick'em entry ledger")


CHECKS = [
    ("python deps", _check_deps),
    ("anthropic (LLM)", _check_anthropic),
    ("statsapi", _check_statsapi),
    ("odds source", _check_odds),
    ("prizepicks", _check_prizepicks),
    ("accuracy caches", _check_caches),
    ("model evidence", _check_model_evidence),
]


def _importable(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _short(e):
    return str(e).splitlines()[0][:80] if str(e) else type(e).__name__


def run(checks=None):
    results = []
    for name, fn in (checks or CHECKS):
        try:
            status, detail = fn()
        except Exception as e:  # a check should never crash the report
            status, detail = FAIL, f"check crashed: {_short(e)}"
        results.append((name, status, detail))
    return results


def report(results):
    icon = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL"}
    lines = ["", "SportsAid preflight — live-run readiness",
             "=" * 64]
    for name, status, detail in results:
        lines.append(f"[{icon[status]}] {name:<18} {detail}")
    lines.append("=" * 64)
    fails = sum(1 for _, s, _ in results if s == FAIL)
    warns = sum(1 for _, s, _ in results if s == WARN)
    if fails:
        lines.append(f"{fails} blocking issue(s) — pipeline will not fully run live.")
    elif warns:
        lines.append(f"Ready with {warns} note(s) — review the WARN rows.")
    else:
        lines.append("All green — ready for a live run.")
    return "\n".join(lines)


def main():
    import sys
    results = run()
    print(report(results))
    sys.exit(1 if any(s == FAIL for _, s, _ in results) else 0)


if __name__ == "__main__":
    main()
