"""Shared resilient HTTP for the free scrapers.

The sportsbook and PrizePicks endpoints are undocumented and — in PrizePicks'
case — fronted by Cloudflare bot protection, where a plain `requests.get` with
a bare User-Agent often gets a 403 challenge. This module centralises one
hardened path shared by all the scrapers:

  * a browser-like session, optionally `cloudscraper` (which solves Cloudflare's
    JS challenge) when it is installed, else a requests.Session with realistic
    headers, and
  * retries with exponential backoff on transient failures.

cloudscraper is an OPTIONAL dependency (`pip install cloudscraper`). Without it
we fall back to plain requests — fine for DraftKings / FanDuel, weaker against
Cloudflare. `using_cloudscraper()` reports which path is active (the doctor
preflight surfaces it).
"""
import time

import requests

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

_session = None


def _build_session():
    """Prefer cloudscraper (Cloudflare-aware); fall back to plain requests."""
    try:
        import cloudscraper
        sess = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False})
        sess.headers.update({"Accept": BROWSER_HEADERS["Accept"],
                             "Accept-Language": BROWSER_HEADERS["Accept-Language"]})
        return sess
    except Exception:
        sess = requests.Session()
        sess.headers.update(BROWSER_HEADERS)
        return sess


def get_session():
    global _session
    if _session is None:
        _session = _build_session()
    return _session


def using_cloudscraper():
    """True when the active session is a cloudscraper (Cloudflare-aware)."""
    return type(get_session()).__module__.split(".")[0] == "cloudscraper"


def get_json(url, params=None, headers=None, timeout=20, retries=3,
             backoff=1.5, session=None):
    """GET `url` and return parsed JSON, retrying transient failures.

    Raises the last exception if every attempt fails. `session` is injectable
    for tests; it defaults to the shared browser session.
    """
    sess = session or get_session()
    last = None
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if attempt < retries - 1 and backoff:
                time.sleep(backoff ** attempt)
    raise last
