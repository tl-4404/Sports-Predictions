"""Offline tests for the resilient HTTP helper and the doctor preflight.

No network, no pytest:  python tests/test_infra.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlb"))

import netfetch
import doctor

FAILURES = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# ── netfetch: retry + session ────────────────────────────────────────────────
print("netfetch:")


class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status = data, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._data


class _FlakySession:
    """Fails the first `fail_times` GETs, then returns `data`."""

    def __init__(self, fail_times, data):
        self.fail_times, self.data, self.calls = fail_times, data, 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("transient")
        return _Resp(self.data)


s = _FlakySession(2, {"ok": 1})
out = netfetch.get_json("http://x", session=s, retries=3, backoff=0)
check("retries then succeeds", out == {"ok": 1} and s.calls == 3, f"(calls={s.calls})")

s2 = _FlakySession(9, {})
try:
    netfetch.get_json("http://x", session=s2, retries=3, backoff=0)
    check("raises after exhausting retries", False)
except ConnectionError:
    check("raises after exhausting retries", s2.calls == 3, f"(calls={s2.calls})")

s3 = _FlakySession(0, {"v": 2})
check("no-failure path returns immediately",
      netfetch.get_json("http://x", session=s3, backoff=0) == {"v": 2}
      and s3.calls == 1)

check("get_session returns a reusable session",
      netfetch.get_session() is netfetch.get_session())
check("using_cloudscraper is a bool", isinstance(netfetch.using_cloudscraper(), bool))

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
