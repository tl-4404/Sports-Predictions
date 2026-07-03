"""MLB pitcher-prop odds ingestion.

Three fetchers behind one interface, selected by the ODDS_SOURCE toggle:

  TheOddsAPIPropsFetcher — real implementation against The Odds API v4
      event endpoints. Player props require a PAID plan; the fetcher raises a
      clear error on the free tier. Markets: pitcher_strikeouts, pitcher_outs.

  SportsbookScraperFetcher — free scraper of a sportsbook's own JSON endpoints
      (DraftKings by default). Best-effort: the endpoints are undocumented and
      their ids drift, so they are configurable via env. Against the book's
      ToS — personal/research use. Single-book by default; add more books for a
      real multi-book de-vig consensus.

  StubPropsFetcher — offline fixture used until an odds source is sorted.
      Reads mlb/cache/stub_prop_lines.json if present, else emits nothing.

Toggle with ODDS_SOURCE=oddsapi|scraper|stub (see get_fetcher). Every fetch is
appended to prop_lines (never overwritten) so we keep full line history —
needed for CLV grading and line-movement features.

Usage:
    python mlb/odds_props.py            # fetch + store today's props
    ODDS_SOURCE=scraper python mlb/odds_props.py   # use the free scraper
"""
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import db

BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
MARKETS = ",".join(db.MARKETS)
BASE_DIR = Path(__file__).resolve().parent
# first existing path wins: your own fixture in cache/, else the shipped sample
STUB_PATHS = (
    BASE_DIR / "cache" / "stub_prop_lines.json",
    BASE_DIR / "fixtures" / "stub_prop_lines.sample.json",
)


class TheOddsAPIPropsFetcher:
    """Fetches pitcher props for all of today's MLB events, all US books."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set ODDS_API_KEY in your environment/.env")

    def _get(self, path, **params):
        params["apiKey"] = self.api_key
        r = requests.get(f"{BASE}{path}", params=params, timeout=20)
        if r.status_code == 401:
            raise RuntimeError(
                "The Odds API rejected the key. Note: player props require a "
                "paid plan — the free tier only serves featured markets."
            )
        r.raise_for_status()
        return r.json()

    def fetch(self, game_date=None):
        """Returns a list of prop_lines row dicts for the given date."""
        game_date = game_date or date.today().isoformat()
        as_of = datetime.now(timezone.utc).isoformat()
        events = self._get(f"/sports/{SPORT}/events", dateFormat="iso")
        rows = []
        for ev in events:
            if not ev.get("commence_time", "").startswith(game_date):
                continue
            # props only exist on the per-event odds endpoint
            data = self._get(
                f"/sports/{SPORT}/events/{ev['id']}/odds",
                regions="us",
                markets=MARKETS,
                oddsFormat="american",
                dateFormat="iso",
            )
            for bk in data.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt["key"] not in db.MARKETS:
                        continue
                    # outcomes come as Over/Under pairs per player+line
                    pair = {}
                    for oc in mkt.get("outcomes", []):
                        key = (oc.get("description"), oc.get("point"))
                        pair.setdefault(key, {})[oc["name"]] = oc.get("price")
                    for (player, line), sides in pair.items():
                        if player is None or line is None:
                            continue
                        rows.append({
                            "game_date": game_date,
                            "game_pk": None,
                            "event_id": ev["id"],
                            "player_name": player,
                            "player_id": None,
                            "market": mkt["key"],
                            "book": bk.get("title") or bk.get("key"),
                            "line": float(line),
                            "over_odds": _int_or_none(sides.get("Over")),
                            "under_odds": _int_or_none(sides.get("Under")),
                            "as_of": as_of,
                        })
        return rows


class StubPropsFetcher:
    """Offline stand-in: serves fixture lines from cache/stub_prop_lines.json.

    Fixture format: [{"player_name": ..., "market": ..., "line": ...,
    "book": ..., "over_odds": ..., "under_odds": ..., "game_pk": ...}, ...]
    """

    def fetch(self, game_date=None):
        game_date = game_date or date.today().isoformat()
        as_of = datetime.now(timezone.utc).isoformat()
        path = next((p for p in STUB_PATHS if p.exists()), None)
        if path is None:
            return []
        with open(path) as f:
            fixtures = json.load(f)
        rows = []
        for fx in fixtures:
            rows.append({
                "game_date": game_date,
                "game_pk": fx.get("game_pk"),
                "event_id": fx.get("event_id"),
                "player_name": fx["player_name"],
                "player_id": fx.get("player_id"),
                "market": fx["market"],
                "book": fx.get("book", "stub"),
                "line": float(fx["line"]),
                "over_odds": _int_or_none(fx.get("over_odds")),
                "under_odds": _int_or_none(fx.get("under_odds")),
                "as_of": as_of,
            })
        return rows


def _int_or_none(v):
    return int(v) if v is not None else None


def _parse_american(v):
    """American odds from a scraped value: handles '+120', '-110', unicode minus."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().replace("−", "-").replace("+", "")
    try:
        return int(s)
    except ValueError:
        return None


# ── free sportsbook scraper (DraftKings) ─────────────────────────────────────

# DraftKings MLB event group + the category ids that hold pitcher props. These
# ids drift over time — override via env and verify against the live board.
DK_BASE = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups"
DK_MLB_EVENTGROUP = "84240"
DK_PITCHER_CATEGORIES = ("1031",)   # "Pitcher Props" — verify against DK


def _dk_market(name):
    """Map a DraftKings subcategory name to our market key (or None)."""
    n = (name or "").lower()
    if "strikeout" in n:
        return "pitcher_strikeouts"
    if "out" in n and ("pitch" in n or "record" in n):
        return "pitcher_outs"
    return None


def _dk_offer_pair(offer):
    """Group a DK offer's outcomes into {(player, line): {'over':, 'under':}}."""
    pair = {}
    for oc in offer.get("outcomes", []) or []:
        label = (oc.get("label") or "").lower()
        side = "over" if "over" in label else "under" if "under" in label else None
        player = oc.get("participant") or offer.get("label")
        line = oc.get("line")
        if side is None or line is None:
            continue
        pair.setdefault((player, line), {})[side] = oc.get("oddsAmerican")
    return pair


class SportsbookScraperFetcher:
    """Free scraper of DraftKings' public JSON endpoints for pitcher props.

    Best-effort: DK's endpoints are undocumented and their event-group /
    category ids drift, so all three are configurable via env (DK_BASE,
    DK_MLB_EVENTGROUP, DK_PITCHER_CATEGORIES). Against DK's ToS — personal /
    research use. Emits the same prop_lines rows as the API fetcher so the rest
    of the pipeline (de-vig consensus, edges, Kelly) is unchanged. Single-book:
    for a real multi-book consensus, scrape a second book and merge the rows.
    """

    BOOK = "DraftKings"

    def __init__(self, eventgroup=None, categories=None, base=None):
        self.base = base or os.getenv("DK_BASE", DK_BASE)
        self.eventgroup = eventgroup or os.getenv("DK_MLB_EVENTGROUP",
                                                  DK_MLB_EVENTGROUP)
        cats = categories or os.getenv("DK_PITCHER_CATEGORIES")
        self.categories = (tuple(c.strip() for c in cats.split(",") if c.strip())
                           if isinstance(cats, str) and cats
                           else tuple(DK_PITCHER_CATEGORIES))

    def fetch(self, game_date=None):
        game_date = game_date or date.today().isoformat()
        as_of = datetime.now(timezone.utc).isoformat()
        rows = []
        for cat in self.categories:
            try:
                payload = self._get_category(cat)
            except Exception as e:  # isolate per-category failures
                print(f"[odds_props] DK category {cat} failed: {e}")
                continue
            rows.extend(self._parse(payload, game_date, as_of))
        return rows

    def _get_category(self, category):
        url = f"{self.base}/{self.eventgroup}/categories/{category}"
        r = requests.get(
            url, params={"format": "json"},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=20)
        r.raise_for_status()
        return r.json()

    def _parse(self, payload, game_date, as_of):
        eg = payload.get("eventGroup", {}) or {}
        event_day = {str(ev.get("eventId")): (ev.get("startDate") or "")[:10]
                     for ev in eg.get("events", []) or []}
        rows = []
        for cat in eg.get("offerCategories", []) or []:
            for sub in cat.get("offerSubcategoryDescriptors", []) or []:
                osub = sub.get("offerSubcategory") or {}
                market = _dk_market(sub.get("name") or osub.get("name"))
                if not market:
                    continue
                for offer_list in osub.get("offers", []) or []:
                    for offer in offer_list or []:
                        eid = str(offer.get("eventId"))
                        day = event_day.get(eid)
                        if day and day != game_date:
                            continue
                        for (player, line), sides in _dk_offer_pair(offer).items():
                            if player is None or line is None:
                                continue
                            rows.append({
                                "game_date": game_date, "game_pk": None,
                                "event_id": eid, "player_name": player,
                                "player_id": None, "market": market,
                                "book": self.BOOK, "line": float(line),
                                "over_odds": _parse_american(sides.get("over")),
                                "under_odds": _parse_american(sides.get("under")),
                                "as_of": as_of})
        return rows


# ── source toggle ────────────────────────────────────────────────────────────

_SOURCES = {
    "oddsapi": TheOddsAPIPropsFetcher, "the-odds-api": TheOddsAPIPropsFetcher,
    "api": TheOddsAPIPropsFetcher,
    "scraper": SportsbookScraperFetcher, "scrape": SportsbookScraperFetcher,
    "sportsbook": SportsbookScraperFetcher, "dk": SportsbookScraperFetcher,
    "stub": StubPropsFetcher, "fixture": StubPropsFetcher,
}


def get_fetcher():
    """Select the odds source via the ODDS_SOURCE toggle.

    ODDS_SOURCE=oddsapi  -> paid The Odds API (needs ODDS_API_KEY)
    ODDS_SOURCE=scraper  -> free DraftKings scraper (SportsbookScraperFetcher)
    ODDS_SOURCE=stub     -> offline fixtures (StubPropsFetcher)

    When ODDS_SOURCE is unset the default preserves prior behavior: the paid API
    if ODDS_API_KEY is set, otherwise the offline stub. Set ODDS_SOURCE=scraper
    to run free now, and flip to oddsapi later once you have a paid plan.
    """
    source = (os.getenv("ODDS_SOURCE") or "").strip().lower()
    if not source:
        source = "oddsapi" if os.getenv("ODDS_API_KEY") else "stub"
    fetcher_cls = _SOURCES.get(source)
    if fetcher_cls is None:
        raise RuntimeError(
            f"Unknown ODDS_SOURCE={source!r} (use oddsapi|scraper|stub)")
    return fetcher_cls()


def refresh(game_date=None):
    conn = db.connect()
    db.ensure_schema(conn)
    fetcher = get_fetcher()
    rows = fetcher.fetch(game_date)
    n = db.insert_lines(conn, rows)
    src = type(fetcher).__name__
    print(f"[odds_props] {src}: stored {n} prop line snapshots")
    conn.close()
    return n


if __name__ == "__main__":
    refresh()
