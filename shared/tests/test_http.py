"""Offline tests for the resilient HTTP helper and the doctor preflight.

No network, no pytest:  python shared/tests/test_http.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.data_fetching import http as netfetch

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


print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
