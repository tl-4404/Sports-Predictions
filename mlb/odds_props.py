"""MLB pitcher-prop odds ingestion.

Two fetchers behind one interface:

  TheOddsAPIPropsFetcher — real implementation against The Odds API v4
      event endpoints. Player props require a PAID plan; the fetcher raises a
      clear error on the free tier. Markets: pitcher_strikeouts, pitcher_outs.

  StubPropsFetcher — offline fixture used until an odds source is sorted.
      Reads mlb/cache/stub_prop_lines.json if present, else emits nothing.

Every fetch is appended to prop_lines (never overwritten) so we keep full
line history — needed for CLV grading and line-movement features.

Usage:
    python mlb/odds_props.py            # fetch + store today's props
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


def get_fetcher():
    """Real fetcher when a key is configured, stub otherwise."""
    if os.getenv("ODDS_API_KEY"):
        return TheOddsAPIPropsFetcher()
    return StubPropsFetcher()


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
