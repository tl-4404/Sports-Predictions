"""MLB PrizePicks configuration, projection mapping, and offline fixtures."""
import json
import os
from datetime import date, datetime, timezone
from mlb.paths import CACHE_DIR, FIXTURES_DIR
from shared.data_fetching.prizepicks import (
    PrizePicksFetcher as ProjectionFetcher, normalize_odds_type,
    parse_projections,
)

STUB_PATHS = (
    CACHE_DIR / "stub_prizepicks_lines.json",
    FIXTURES_DIR / "stub_prizepicks_lines.sample.json",
)

STAT_MAP = {
    "pitcher strikeouts": "pitcher_strikeouts",
    "strikeouts": "pitcher_strikeouts",
    "strikeouts (pitcher)": "pitcher_strikeouts",
    "pitching outs": "pitcher_outs",
    "pitcher outs": "pitcher_outs",
    "outs": "pitcher_outs",
    "outs recorded": "pitcher_outs",
}


class PrizePicksFetcher(ProjectionFetcher):
    """PrizePicks client configured for MLB pitcher props."""

    def __init__(self, league_id=None, base=None, state_code=None):
        super().__init__(
            league_id=league_id or os.getenv("PRIZEPICKS_MLB_LEAGUE_ID", "2"),
            stat_map=STAT_MAP, base=base, state_code=state_code,
        )


class StubPrizePicksFetcher:
    """Offline stand-in serving fixture projections from JSON."""

    def fetch(self, game_date=None):
        game_date = game_date or date.today().isoformat()
        as_of = datetime.now(timezone.utc).isoformat()
        path = next((p for p in STUB_PATHS if p.exists()), None)
        if path is None:
            return []
        with open(path) as f:
            fixtures = json.load(f)
        rows = []
        for fixture in fixtures:
            market = fixture.get("market") or STAT_MAP.get(
                (fixture.get("stat_type") or "").strip().lower())
            if not market:
                continue
            rows.append({
                "game_date": game_date,
                "player_name": fixture["player_name"],
                "player_id": fixture.get("player_id"),
                "market": market,
                "line": float(fixture["line"]),
                "odds_type": normalize_odds_type(fixture.get("odds_type")),
                "stat_type": fixture.get("stat_type", market),
                "projection_id": fixture.get("projection_id"),
                "as_of": as_of,
            })
        return rows


def _parse_projections(payload, game_date, as_of):
    return parse_projections(payload, game_date, as_of, STAT_MAP)


def get_fetcher():
    """Use the live fetcher only for an explicit truthy opt-in value."""
    live = os.getenv("PRIZEPICKS_LIVE", "").strip().lower()
    if live in {"1", "true", "yes", "on"}:
        return PrizePicksFetcher()
    return StubPrizePicksFetcher()
