"""NBA full-game Polymarket US discovery and auditable quote packets."""
from nba.ledger.domain import decimal_value, full_game_market, market_start, timestamp, utcnow
from shared.data_fetching import polymarket


def snapshot_packet(event, market, book, observed_at=None):
    if not full_game_market(market):
        raise ValueError("Only full-game NBA moneylines are supported")
    observed_at = timestamp(observed_at or utcnow())
    start = market_start(event, market)
    quotes = polymarket.moneyline_quotes(event, market, book)
    by_long = {row["long"]: row for row in quotes}
    sides = {side["long"]: side for side in market["marketSides"]}
    names = {}
    for direction, side in sides.items():
        team = side.get("team") or {}
        if team.get("league") != "nba" or not team.get("name"):
            raise ValueError("Explicit NBA team identity required for each side")
        names[direction] = team["name"]
    if names[True] == names[False] or not event.get("id"):
        raise ValueError("Two distinct teams and an event ID are required")
    pregame = (observed_at < start and event.get("active") is True
               and not any(event.get(k) is True for k in ("live", "ended", "closed")))
    packet = {
        "event_id": str(event["id"]), "market_slug": market["slug"],
        "scheduled_start": start.isoformat(), "observed_at": observed_at.isoformat(),
        "pregame": pregame, "long_team": names[True], "short_team": names[False],
        "raw_event": {k: v for k, v in event.items() if k != "markets"},
        "raw_market": market, "raw_book": book,
    }
    for direction, label in ((True, "long"), (False, "short")):
        row = by_long[direction]
        packet[label+"_price"] = (decimal_value(row["odds"]["contract_price"], minimum=0, maximum=1)
                                  if row["odds"] else None)
        try:
            packet[label+"_shares"] = decimal_value(row["quoted_shares"], minimum=0)
        except ValueError:
            packet[label+"_shares"] = None
    return packet


def discover_events(page_size=50, max_pages=10):
    """Bounded pagination, deduped by event ID; fail visibly on truncation."""
    if not 1 <= page_size <= 100 or not 1 <= max_pages <= 100:
        raise ValueError("Invalid pagination bounds")
    events = {}
    for page_number in range(max_pages):
        page = polymarket.fetch_league_events("nba", limit=page_size, offset=page_number*page_size)
        for event in page["events"]:
            if not event.get("id"):
                raise ValueError("Discovery event lacks ID")
            events[str(event["id"])] = event
        if len(page["events"]) < page_size:
            return list(events.values())
    raise RuntimeError("Discovery reached max-pages; increase the bound to avoid a silently partial slate")


def refresh_packet(event_id, market_slug):
    event = polymarket.fetch_event(event_id)
    matches = [m for m in event.get("markets", []) if m.get("slug") == market_slug]
    if len(matches) != 1:
        raise ValueError("Tracked market missing/ambiguous in current event response")
    book = polymarket.fetch_market_bbo(market_slug)
    return snapshot_packet(event, matches[0], book)
