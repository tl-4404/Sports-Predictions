"""MLB-specific pitcher-prop odds ingestion.

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
      Reads mlb/data/cache/stub_prop_lines.json if present, else emits nothing.

Toggle with ODDS_SOURCE=oddsapi|scraper|stub (see get_fetcher). Every fetch is
appended to prop_lines (never overwritten) so we keep full line history —
needed for CLV grading and line-movement features.

Usage:
    python -m mlb.data_fetching.providers.odds            # fetch + store today's props
    ODDS_SOURCE=scraper python -m mlb.data_fetching.providers.odds   # use the free scraper
"""
import json
import os
from datetime import date, datetime, timezone

from shared.data_fetching.sportsbooks import BASE_URL, OddsAPIClient

try:
    from dotenv import load_dotenv
    from shared.paths import PROJECT_ROOT
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from mlb import db
from mlb.paths import CACHE_DIR, FIXTURES_DIR
from shared.data_fetching import http

BASE = BASE_URL
SPORT = "baseball_mlb"
MARKETS = ",".join(db.MARKETS)
# first existing path wins: your own fixture in cache/, else the shipped sample
STUB_PATHS = (
    CACHE_DIR / "stub_prop_lines.json",
    FIXTURES_DIR / "stub_prop_lines.sample.json",
)


class TheOddsAPIPropsFetcher:
    """Fetches pitcher props for all of today's MLB events, all US books."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set ODDS_API_KEY in your environment/.env")

    def _get(self, path, **params):
        return OddsAPIClient(self.api_key, base_url=BASE).get(path, **params)

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


# ── free sportsbook scrapers (per book) ──────────────────────────────────────
#
# Each scraper hits one book's undocumented JSON endpoints and emits the same
# prop_lines rows as the API fetcher, so edges/de-vig/Kelly are unchanged.
# Best-effort: these endpoints and their ids drift, so everything is
# configurable via env — verify against the live boards. Against the books'
# ToS — personal / research use.


def _prop_market(name):
    """Map a book's market/subcategory label to our market key (or None)."""
    n = (name or "").lower()
    if "strikeout" in n:
        return "pitcher_strikeouts"
    if "out" in n and ("pitch" in n or "record" in n):
        return "pitcher_outs"
    return None


def _row(game_date, event_id, player, market, book, line, over, under, as_of):
    return {
        "game_date": game_date, "game_pk": None, "event_id": event_id,
        "player_name": player, "player_id": None, "market": market,
        "book": book, "line": float(line),
        "over_odds": _parse_american(over), "under_odds": _parse_american(under),
        "as_of": as_of,
    }


# DraftKings MLB event group + the category ids that hold pitcher props.
DK_BASE = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups"
DK_MLB_EVENTGROUP = "84240"
DK_PITCHER_CATEGORIES = ("1031",)   # "Pitcher Props" — verify against DK


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


class DraftKingsScraper:
    """DraftKings pitcher props. Ids drift — override via DK_BASE /
    DK_MLB_EVENTGROUP / DK_PITCHER_CATEGORIES."""

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
        return http.get_json(url, params={"format": "json"})

    def _parse(self, payload, game_date, as_of):
        eg = payload.get("eventGroup", {}) or {}
        event_day = {str(ev.get("eventId")): (ev.get("startDate") or "")[:10]
                     for ev in eg.get("events", []) or []}
        rows = []
        for cat in eg.get("offerCategories", []) or []:
            for sub in cat.get("offerSubcategoryDescriptors", []) or []:
                osub = sub.get("offerSubcategory") or {}
                market = _prop_market(sub.get("name") or osub.get("name"))
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
                            rows.append(_row(
                                game_date, eid, player, market, self.BOOK, line,
                                sides.get("over"), sides.get("under"), as_of))
        return rows


# FanDuel content-managed page for MLB props. FanDuel keys markets by a state
# subdomain and puts the player in the market name ("Gerrit Cole - Strikeouts"),
# the line on each runner's `handicap`, and odds under
# winRunnerOdds.americanDisplayOdds.americanOdds. Override the URL via env.
FD_BASE = ("https://sbapi.nj.sportsbook.fanduel.com/api/content-managed-page")
FD_MLB_PAGE = "mlb"


def _fd_player(market_name):
    """Pull the player name out of a FanDuel market name ('Name - Market')."""
    name = (market_name or "").split(" - ")[0].strip()
    return name or None


class FanDuelScraper:
    """FanDuel pitcher props. Best-effort — override FD_BASE / FD_MLB_PAGE and
    verify the state subdomain against the live board."""

    BOOK = "FanDuel"

    def __init__(self, base=None, page=None):
        self.base = base or os.getenv("FD_BASE", FD_BASE)
        self.page = page or os.getenv("FD_MLB_PAGE", FD_MLB_PAGE)

    def fetch(self, game_date=None):
        game_date = game_date or date.today().isoformat()
        as_of = datetime.now(timezone.utc).isoformat()
        payload = http.get_json(
            self.base,
            params={"page": "CUSTOM", "customPageId": self.page,
                    "_ak": "FhMFpcPWXMeyZxOx", "timezone": "America/New_York"})
        return self._parse(payload, game_date, as_of)

    def _parse(self, payload, game_date, as_of):
        att = payload.get("attachments", {}) or {}
        events = att.get("events", {}) or {}
        event_day = {str(k): (v.get("openDate") or "")[:10]
                     for k, v in events.items()}
        rows = []
        for mkt in (att.get("markets", {}) or {}).values():
            market = _prop_market(mkt.get("marketType") or mkt.get("marketName"))
            if not market:
                continue
            eid = str(mkt.get("eventId"))
            day = event_day.get(eid)
            if day and day != game_date:
                continue
            player = _fd_player(mkt.get("marketName"))
            if not player:
                continue
            sides, line = {}, None
            for run in mkt.get("runners", []) or []:
                name = (run.get("runnerName") or "").lower()
                side = "over" if "over" in name else "under" if "under" in name else None
                if side is None:
                    continue
                line = run.get("handicap", line)
                odds = (((run.get("winRunnerOdds") or {})
                         .get("americanDisplayOdds") or {}).get("americanOdds"))
                sides[side] = odds
            if line is None or not sides:
                continue
            rows.append(_row(game_date, eid, player, market, self.BOOK, line,
                             sides.get("over"), sides.get("under"), as_of))
        return rows


BOOK_SCRAPERS = {
    "draftkings": DraftKingsScraper, "dk": DraftKingsScraper,
    "fanduel": FanDuelScraper, "fd": FanDuelScraper,
}


class SportsbookScraperFetcher:
    """Aggregates one or more single-book scrapers into a multi-book snapshot,
    so edges de-vig against a real consensus rather than one book's price.

    Books via SCRAPE_BOOKS (comma-separated; default 'draftkings,fanduel').
    Per-book failures are isolated so one book going down doesn't sink the run.
    """

    def __init__(self, books=None):
        names = books or os.getenv("SCRAPE_BOOKS") or "draftkings,fanduel"
        if isinstance(names, str):
            names = [n.strip().lower() for n in names.split(",") if n.strip()]
        self.scrapers = [BOOK_SCRAPERS[n]() for n in names if n in BOOK_SCRAPERS]

    def fetch(self, game_date=None):
        rows = []
        for s in self.scrapers:
            try:
                rows.extend(s.fetch(game_date))
            except Exception as e:  # isolate per-book failures
                print(f"[odds_props] {s.BOOK} scrape failed: {e}")
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
