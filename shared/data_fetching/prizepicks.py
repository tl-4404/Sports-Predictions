"""League-independent PrizePicks transport and JSON:API projection parser.

Callers supply the league ID and supported stat mapping. Prediction, fixtures,
model dependencies, and database writes belong to the consuming league.
"""
import os
from datetime import date, datetime, timezone
from shared.data_fetching import http

def normalize_odds_type(value):
    value = (value or "standard").strip().lower()
    return value if value in {"standard", "goblin", "demon"} else "unknown"


class PrizePicksFetcher:
    """Best-effort fetcher for PrizePicks' undocumented projection endpoint."""

    def __init__(self, league_id, stat_map, base=None, state_code=None):
        self.league_id = str(league_id)
        self.stat_map = dict(stat_map)
        self.base = base or os.getenv(
            "PRIZEPICKS_URL", "https://api.prizepicks.com/projections")
        self.state_code = (
            state_code or os.getenv("PRIZEPICKS_STATE_CODE", "")
        ).strip().upper()

    def fetch(self, game_date=None):
        game_date = game_date or date.today().isoformat()
        if not self.state_code:
            raise RuntimeError(
                "PRIZEPICKS_STATE_CODE is required for a jurisdiction-correct board"
            )

        as_of = datetime.now(timezone.utc).isoformat()
        rows = []
        page = 1
        while True:
            payload = http.get_json(
                self.base,
                params={
                    "league_id": self.league_id,
                    "per_page": 250,
                    "page": page,
                    "single_stat": "true",
                    "game_mode": "pickem",
                    "is_live": "false",
                    "state_code": self.state_code,
                },
            )
            rows.extend(parse_projections(payload, game_date, as_of, self.stat_map))
            meta = payload.get("meta") or {}
            try:
                total_pages = max(1, int(meta.get("total_pages") or 1))
            except (TypeError, ValueError):
                total_pages = 1
            if page >= total_pages:
                break
            page += 1

        # Defensive deduplication in case the source repeats a projection across
        # page boundaries during a board update.
        unique = {}
        for row in rows:
            key = (
                row.get("projection_id"), row["player_name"], row["market"],
                row["line"], row["odds_type"],
            )
            unique[key] = row
        return list(unique.values())


def parse_projections(payload, game_date, as_of, stat_map):
    """Turn a PrizePicks JSON:API payload into stored line rows."""
    names = {}
    for included in payload.get("included", []):
        if included.get("type") in ("new_player", "player"):
            attributes = included.get("attributes") or {}
            names[included.get("id")] = (
                attributes.get("name") or attributes.get("display_name")
            )

    rows = []
    for projection in payload.get("data", []):
        attributes = projection.get("attributes", {}) or {}
        market = stat_map.get(
            (attributes.get("stat_type") or "").strip().lower()
        )
        if not market:
            continue
        status = (attributes.get("status") or "pre_game").strip().lower()
        if status not in {"pre_game", "scheduled"}:
            continue
        start = attributes.get("start_time") or ""
        if start and not start.startswith(game_date):
            continue
        relationship = (
            (projection.get("relationships", {}) or {}).get("new_player", {})
        )
        player_ref = (relationship.get("data") or {}).get("id")
        name = names.get(player_ref)
        line = attributes.get("line_score")
        if not name or line is None:
            continue
        rows.append({
            "game_date": game_date,
            "player_name": name,
            "player_id": None,
            "market": market,
            "line": float(line),
            "odds_type": normalize_odds_type(attributes.get("odds_type")),
            "stat_type": attributes.get("stat_type"),
            "projection_id": projection.get("id"),
            "as_of": as_of,
        })
    return rows
