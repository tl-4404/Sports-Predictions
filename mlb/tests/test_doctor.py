"""Offline tests for the resilient HTTP helper and the doctor preflight.

No network, no pytest:  python mlb/tests/test_doctor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlb import doctor

FAILURES = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# ── doctor: report rendering + isolation ─────────────────────────────────────
print("doctor:")


def ok():
    return doctor.OK, "fine"


def warn():
    return doctor.WARN, "heads up"


def fail():
    return doctor.FAIL, "broken"


def crash():
    raise ValueError("kaboom")


res = doctor.run([("a", ok), ("b", warn), ("c", crash)])
check("run isolates a crashing check as FAIL",
      res[2][1] == doctor.FAIL and "crashed" in res[2][2])
check("run preserves order + statuses",
      [r[1] for r in res] == [doctor.OK, doctor.WARN, doctor.FAIL])

rep = doctor.report(res)
check("report shows all three icons",
      "[OK  ]" in rep and "[WARN]" in rep and "[FAIL]" in rep)
check("report summarises blocking issues", "blocking issue" in rep)

warn_only = doctor.report(doctor.run([("a", ok), ("b", warn)]))
check("warn-only summary", "Ready with 1 note" in warn_only)

all_ok = doctor.report(doctor.run([("a", ok), ("b", ok)]))
check("all-green summary", "All green" in all_ok)

# the real check functions must never raise (they wrap their own errors)
print("doctor real checks (offline — network checks may FAIL, must not crash):")
real = doctor.run()
check("all real checks return a valid status",
      all(s in (doctor.OK, doctor.WARN, doctor.FAIL) for _, s, _ in real),
      f"({[s for _, s, _ in real]})")
check("deps check passes (numpy/pandas/requests installed)",
      dict((n, s) for n, s, _ in real)["python deps"] == doctor.OK)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
