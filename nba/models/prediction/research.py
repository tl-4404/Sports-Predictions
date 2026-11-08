"""Timestamped read-only NBA evidence; no model probabilities or market prices."""
from datetime import timedelta
import json
from zoneinfo import ZoneInfo

import pandas as pd

from nba.ledger.domain import timestamp, utcnow
from nba.data_fetching.providers import nba_api, espn

STATS = {
    "Base": ("GP", "W_PCT", "PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "OREB", "DREB", "AST", "TOV"),
    "Advanced": ("OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "TS_PCT", "EFG_PCT", "OREB_PCT", "DREB_PCT", "TM_TOV_PCT"),
}


def season_for(day):
    year = day.year if day.month >= 7 else day.year-1
    return f"{year}-{(year+1)%100:02d}"


class ResearchCollector:
    """One fresh league snapshot per measure/run; cached only in this process."""
    def __init__(self, now=None):
        self.as_of = timestamp(now or utcnow())
        self.cutoff = self.as_of.astimezone(ZoneInfo("America/New_York")).date() - timedelta(days=1)
        self.season = season_for(self.cutoff)
        self.tables = {}
        self.injuries = {}

    def _stats(self, name, measure):
        if measure not in self.tables:
            frame = nba_api.fetch_team_stats(self.season, self.cutoff.strftime("%m/%d/%Y"), measure)
            missing = {"TEAM_NAME", *STATS[measure]} - set(frame)
            if missing:
                raise ValueError(f"NBA {measure} missing required fields: {sorted(missing)}")
            self.tables[measure] = (frame, utcnow().isoformat())
        frame, observed_at = self.tables[measure]
        # Explicit, narrow provider-name alias, never fuzzy matching.
        official = {"Los Angeles Clippers": "LA Clippers"}.get(name, name)
        rows = frame.loc[frame.TEAM_NAME.isin({name, official}), list(STATS[measure])]
        if len(rows) != 1 or rows.isna().any().any():
            raise ValueError(f"Missing/ambiguous/incomplete {measure} stats for {name}")
        return {"provider": "nba_stats", "endpoint": "LeagueDashTeamStats", "observed_at": observed_at,
                "season": self.season, "through_date": self.cutoff.isoformat(),
                "aggregation": "official season-to-date per-game snapshot",
                "team": name, "stats": json.loads(rows.to_json(orient="records"))[0]}

    def collect(self, packet):
        evidence, roles = {}, {}
        for label, direction in (("long", True), ("short", False)):
            name = packet[label+"_team"]
            for measure in STATS:
                evidence[f"{label}_{measure.lower()}"] = self._stats(name, measure)
            if name not in self.injuries:
                self.injuries[name] = espn.get_team_injury_snapshot(name)
            evidence[f"{label}_injuries"] = self.injuries[name]
            side = next(s for s in packet["raw_market"]["marketSides"] if s["long"] is direction)
            role = (side.get("team") or {}).get("ordering")
            roles[label] = role if role in ("home", "away") else "unknown"
        return {
            "schema": "nba_agent_evidence_v1", "collected_at": utcnow().isoformat(),
            "stats_through_date": self.cutoff.isoformat(), "season": self.season,
            "event": {"event_id": packet["event_id"], "market_slug": packet["market_slug"],
                      "scheduled_start": packet["scheduled_start"], "long_team": packet["long_team"],
                      "short_team": packet["short_team"], "roles": roles},
            "evidence": evidence, "model_included": False, "odds_included": False,
        }
