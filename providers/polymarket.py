"""Shared Polymarket US public data client for the API walkthrough.

Step 1: python3 -m providers.polymarket
Step 2: python3 -m providers.polymarket --events --limit 3
Moneylines: python3 -m providers.polymarket --moneylines --limit 3
Documentation: docs/POLYMARKET_API.md
"""
import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from urllib.parse import quote

import requests


PUBLIC_BASE_URL = "https://gateway.polymarket.us"


def fetch_league(slug="nba"):
    """Fetch one US league by its URL-friendly name (slug), without a key."""
    if not isinstance(slug, str) or not slug.strip():
        raise ValueError("A nonempty league slug is required")
    slug = slug.strip().lower()
    response = requests.get(
        f"{PUBLIC_BASE_URL}/v2/leagues/{quote(slug, safe='')}",
        timeout=20,
    )
    # Reject HTTP failures before trying to use the body as application data.
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("league"), dict):
        raise ValueError("Expected a JSON object containing a league object")
    league = data["league"]
    if league.get("slug") != slug:
        raise ValueError(f"Expected league slug {slug!r} in the response")
    return league

def fetch_league_events(slug="nba", limit=3, offset=0):
    """Fetch one page of game events, retaining the API's response wrapper."""
    if not isinstance(slug, str) or not slug.strip():
        raise ValueError("A nonempty league slug is required")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    slug = slug.strip().lower()
    response = requests.get(
        f"{PUBLIC_BASE_URL}/v2/leagues/{quote(slug, safe='')}/events",
        params={
            "limit": limit,
            "offset": offset,
            "type": "sport",
            "section": "general",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        raise ValueError("Expected a JSON object containing an events list")
    if any(not isinstance(event, dict) for event in data["events"]):
        raise ValueError("Expected every event to be a JSON object")
    return data


def fetch_market_bbo(slug):
    """Read the best bid and offer for the market's single YES instrument."""
    if not isinstance(slug, str) or not slug.strip():
        raise ValueError("A nonempty market slug is required")
    slug = slug.strip()
    response = requests.get(
        f"{PUBLIC_BASE_URL}/v1/markets/{quote(slug, safe='')}/bbo",
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("marketData"), dict):
        raise ValueError("Expected a JSON object containing marketData")
    book = data["marketData"]
    if book.get("marketSlug") != slug:
        raise ValueError("BBO response does not match the requested market slug")
    return book


def _usd_price(quote_value):
    """Missing, non-USD and invalid prices are unavailable, never zero odds."""
    if not isinstance(quote_value, dict) or quote_value.get("currency") != "USD":
        return None
    try:
        price = Decimal(str(quote_value["value"]))
    except (KeyError, InvalidOperation):
        return None
    return price if price.is_finite() and 0 < price < 1 else None


def price_to_odds(price):
    """Gross equivalent odds for a $1/$0 outcome, excluding fees."""
    price = Decimal(str(price))
    if not price.is_finite() or not 0 < price < 1:
        raise ValueError("Contract price must be strictly between 0 and 1")
    american = (
        -100 * price / (1 - price) if price >= Decimal("0.5")
        else 100 * (1 - price) / price
    )
    return {
        "contract_price": float(price),
        "decimal_odds": round(float(1 / price), 4),
        "american_odds": round(float(american)),
    }


def moneyline_quotes(event, market, book):
    """Map team sides to taker entry prices; preserve missing quotes explicitly."""
    if book.get("marketSlug") != market.get("slug"):
        raise ValueError("Cannot map a book to a different market")
    sides = market.get("marketSides") or []
    if (len(sides) != 2
            or sum(side.get("long") is True for side in sides) != 1
            or sum(side.get("long") is False for side in sides) != 1):
        raise ValueError("Expected exactly one long and one short moneyline side")
    bid = _usd_price(book.get("bestBid"))
    ask = _usd_price(book.get("bestAsk"))
    crossed = bid is not None and ask is not None and bid > ask
    observed_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for side in sides:
        long = side["long"]
        price = ask if long else (1 - bid if bid is not None else None)
        reason = None
        if event.get("closed") is True or event.get("ended") is True:
            reason = "event ended/closed"
        elif market.get("closed") is True or market.get("active") is not True:
            reason = "market inactive/closed"
        elif side.get("tradable") is not True:
            reason = "side not tradable"
        elif book.get("state") != "MARKET_STATE_OPEN":
            reason = f"book state: {book.get('state', 'unknown')}"
        elif crossed:
            reason = "crossed book"
        elif price is None:
            reason = "no valid USD ask" if long else "no valid USD bid"
        team = side.get("team") or {}
        rows.append({
            "event_id": event.get("id"),
            "event": event.get("title"),
            "start_time": event.get("startTime"),
            "live": event.get("live"),
            "market_slug": market["slug"],
            "team": side.get("description") or team.get("name"),
            "team_id": side.get("teamId"),
            "long": long,
            "price_basis": "bestAsk" if long else "1 - bestBid",
            "quoted_shares": book.get("askShares" if long else "bidShares"),
            "state": book.get("state"),
            "observed_at": observed_at,
            "unavailable_reason": reason,
            "odds": price_to_odds(price) if reason is None else None,
        })
    return rows


def fetch_moneylines(slug="nba", limit=3, offset=0):
    """Fetch moneyline BBOs for one discovery page, including per-market errors."""
    page = fetch_league_events(slug, limit=limit, offset=offset)
    rows, errors = [], []
    for event in page["events"]:
        for market in event.get("markets") or []:
            if market.get("sportsMarketTypeV2") != "SPORTS_MARKET_TYPE_MONEYLINE":
                continue
            market_slug = market.get("slug")
            try:
                book = fetch_market_bbo(market_slug)
                rows.extend(moneyline_quotes(event, market, book))
            except (requests.RequestException, ValueError) as exc:
                errors.append({"market_slug": market_slug, "error": str(exc)})
    return {"events_received": len(page["events"]), "quotes": rows, "errors": errors}


# NBA only for the walkthrough; the functions also accept other league slugs.
def main():
    parser = argparse.ArgumentParser(description="Explore Polymarket US NBA data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--events", action="store_true", help="fetch a page of games")
    mode.add_argument("--moneylines", action="store_true", help="fetch moneyline entry odds")
    parser.add_argument("--limit", type=int, default=3, help="events requested per page")
    parser.add_argument("--offset", type=int, default=0, help="events to skip")
    parser.add_argument("--raw", action="store_true", help="print the complete events response")
    args = parser.parse_args()
    if args.moneylines:
        result = fetch_moneylines("nba", limit=args.limit, offset=args.offset)
        print(json.dumps(result, indent=2))
        if result["errors"]:
            raise SystemExit(1)
        return
    if args.events:
        data = fetch_league_events("nba", limit=args.limit, offset=args.offset)
        if args.raw:
            print(json.dumps(data, indent=2))
            return
        # This is a readable view. fetch_league_events retains all raw fields.
        events = data["events"]
        print(f"Received {len(events)} NBA events (offset={args.offset}).")
        for event in events:
            summary = {key: event.get(key) for key in (
                "id", "title", "slug", "startTime", "active", "closed", "live", "ended"
            )}
            summary["markets"] = [
                {
                    **{key: market.get(key) for key in (
                        "id", "question", "slug", "sportsMarketTypeV2", "line",
                        "active", "closed"
                    )},
                    "marketSides": [
                        {key: side.get(key) for key in (
                            "id", "description", "long", "teamId", "tradable"
                        )}
                        for side in (market.get("marketSides") or [])
                    ],
                }
                for market in (event.get("markets") or [])
            ]
            print(json.dumps(summary, indent=2))
        return
    nba = fetch_league("nba")
    print(json.dumps(nba, indent=2))


if __name__ == "__main__":
    main()
