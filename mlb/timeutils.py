"""Timezone-safe helpers shared by the live MLB pipeline.

Provider timestamps are normally UTC while an MLB slate is identified by its
Eastern date. Comparing ISO strings with ``startswith(game_date)`` drops late
West-Coast games after midnight UTC, so all provider times are parsed before
they are assigned to a slate.
"""

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_SLATE_TIMEZONE = "America/New_York"


def slate_timezone():
    name = os.getenv("MLB_SLATE_TIMEZONE", DEFAULT_SLATE_TIMEZONE)
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo(DEFAULT_SLATE_TIMEZONE)


def parse_datetime(value):
    """Return an aware datetime, or ``None`` for a missing/invalid value."""
    if isinstance(value, datetime):
        dt = value
    elif value:
        try:
            dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def slate_date(value):
    dt = parse_datetime(value)
    return dt.astimezone(slate_timezone()).date().isoformat() if dt else None


def slate_today(now=None):
    dt = parse_datetime(now) if now is not None else datetime.now(timezone.utc)
    return dt.astimezone(slate_timezone()).date().isoformat()


def is_future_event(start_time, as_of=None):
    """True only when both timestamps are valid and first pitch is in future."""
    start = parse_datetime(start_time)
    current = parse_datetime(as_of) if as_of is not None else datetime.now(timezone.utc)
    return bool(start and current and start > current)


def is_open_slate_event(start_time, game_date, as_of=None):
    """Require the requested Eastern slate date and a not-yet-started event."""
    return slate_date(start_time) == game_date and is_future_event(start_time, as_of)
