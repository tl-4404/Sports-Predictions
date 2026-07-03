"""Offline tests for the odds-source toggle, the DraftKings scraper parser,
and the PrizePicks interface loader.

No network, no pytest:  python tests/test_interface_and_toggle.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mlb"))
sys.path.insert(0, str(ROOT))

import db
import odds_props
import app as webapp

FAILURES = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def with_env(**overrides):
    """Set env vars (None deletes), return a restore callable."""
    saved = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    def restore():
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return restore


# ── ODDS_SOURCE toggle ───────────────────────────────────────────────────────
print("odds source toggle:")

r = with_env(ODDS_SOURCE="scraper", ODDS_API_KEY=None)
check("ODDS_SOURCE=scraper -> SportsbookScraperFetcher",
      isinstance(odds_props.get_fetcher(), odds_props.SportsbookScraperFetcher))
r()

r = with_env(ODDS_SOURCE="stub", ODDS_API_KEY=None)
check("ODDS_SOURCE=stub -> StubPropsFetcher",
      isinstance(odds_props.get_fetcher(), odds_props.StubPropsFetcher))
r()

r = with_env(ODDS_SOURCE="oddsapi", ODDS_API_KEY="test-key")
check("ODDS_SOURCE=oddsapi (+key) -> TheOddsAPIPropsFetcher",
      isinstance(odds_props.get_fetcher(), odds_props.TheOddsAPIPropsFetcher))
r()

r = with_env(ODDS_SOURCE=None, ODDS_API_KEY=None)
check("unset + no key -> stub (back-compat)",
      isinstance(odds_props.get_fetcher(), odds_props.StubPropsFetcher))
r()

r = with_env(ODDS_SOURCE=None, ODDS_API_KEY="test-key")
check("unset + key -> odds api (back-compat)",
      isinstance(odds_props.get_fetcher(), odds_props.TheOddsAPIPropsFetcher))
r()

r = with_env(ODDS_SOURCE="nonsense", ODDS_API_KEY=None)
try:
    odds_props.get_fetcher()
    check("bad ODDS_SOURCE raises", False)
except RuntimeError:
    check("bad ODDS_SOURCE raises", True)
r()

# ── DraftKings scraper parser (offline payload) ──────────────────────────────
print("draftkings scraper parser:")

DK_PAYLOAD = {
    "eventGroup": {
        "events": [
            {"eventId": 5001, "startDate": "2026-07-03T23:05:00Z"},
            {"eventId": 5002, "startDate": "2026-07-04T18:10:00Z"},  # other day
        ],
        "offerCategories": [
            {"offerSubcategoryDescriptors": [
                {"name": "Pitcher Strikeouts", "offerSubcategory": {"offers": [[
                    {"eventId": 5001, "label": "Gerrit Cole", "outcomes": [
                        {"label": "Over", "participant": "Gerrit Cole",
                         "line": 6.5, "oddsAmerican": "-115"},
                        {"label": "Under", "participant": "Gerrit Cole",
                         "line": 6.5, "oddsAmerican": "−105"},  # unicode minus
                    ]},
                    {"eventId": 5002, "label": "Tomorrow Guy", "outcomes": [
                        {"label": "Over", "participant": "Tomorrow Guy",
                         "line": 5.5, "oddsAmerican": "+100"},
                        {"label": "Under", "participant": "Tomorrow Guy",
                         "line": 5.5, "oddsAmerican": "-120"},
                    ]},
                ]]}},
                {"name": "Pitcher Outs Recorded", "offerSubcategory": {"offers": [[
                    {"eventId": 5001, "label": "Gerrit Cole", "outcomes": [
                        {"label": "Over", "participant": "Gerrit Cole",
                         "line": 17.5, "oddsAmerican": "-120"},
                        {"label": "Under", "participant": "Gerrit Cole",
                         "line": 17.5, "oddsAmerican": "+100"},
                    ]},
                ]]}},
                {"name": "Total Bases", "offerSubcategory": {"offers": []}},  # ignored
            ]},
        ],
    },
}

rows = odds_props.SportsbookScraperFetcher()._parse(
    DK_PAYLOAD, "2026-07-03", "2026-07-03T12:00:00Z")
check("parsed only today's events (2 rows)", len(rows) == 2, f"({len(rows)})")
by_mkt = {r["market"]: r for r in rows}
check("both markets present",
      set(by_mkt) == {"pitcher_strikeouts", "pitcher_outs"})
k = by_mkt.get("pitcher_strikeouts")
check("strikeouts row parsed", k and k["player_name"] == "Gerrit Cole"
      and k["line"] == 6.5 and k["book"] == "DraftKings")
check("american odds parsed (incl. unicode minus)",
      k and k["over_odds"] == -115 and k["under_odds"] == -105,
      f"({k['over_odds']}/{k['under_odds']})" if k else "")
check("future-day event excluded",
      all(r["player_name"] != "Tomorrow Guy" for r in rows))

# scraped rows flow through the existing de-vig consensus unchanged
from betting_math import consensus_p_over
mp = consensus_p_over([(k["over_odds"], k["under_odds"])])
check("scraped row de-vigs to a consensus prob", mp is not None and 0 < mp < 1,
      f"({mp:.3f})")

# ── PrizePicks interface loader ──────────────────────────────────────────────
print("prizepicks interface loader:")

conn = db.connect(Path(tempfile.mkdtemp()) / "iface.db")

# missing table -> [] (lane never ran)
check("missing table -> empty board",
      webapp.load_prizepicks_board(conn, "2026-07-03") == [])

# seed prizepicks_picks directly
import prizepicks as pp
pp.ensure_schema(conn)
seed = [
    ("r1", "Ace Example", "pitcher_strikeouts", 6.5, 0.68, "more", 0.68, 0.577, 0.103, 1),
    ("r1", "Soft Tosser", "pitcher_outs", 14.5, 0.48, "less", 0.52, 0.577, -0.057, 0),
    ("r1", "Ace Example", "pitcher_outs", 16.5, 0.76, "more", 0.76, 0.577, 0.183, 1),
]
for run_id, name, mkt, line, p_over, side, p_pick, be, edge, playable in seed:
    conn.execute(
        """INSERT INTO prizepicks_picks
           (run_id, created_at, game_date, game_pk, player_name, player_id,
            market, line, p_over, pick_side, p_pick, breakeven, edge,
            ev_2, ev_3, playable, mean)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, db.utcnow(), "2026-07-03", 1, name, 1, mkt, line, p_over,
         side, p_pick, be, edge, 0.0, 0.0, playable, 0.0))
conn.commit()

board = webapp.load_prizepicks_board(conn, "2026-07-03")
check("board returns all picks", len(board) == 3, f"({len(board)})")
check("sorted by edge desc",
      [round(p["edge"], 3) for p in board] == sorted(
          [round(p["edge"], 3) for p in board], reverse=True))
check("market_label populated",
      board[0]["market_label"] in ("Strikeouts", "Pitcher Outs"))
playable = webapp.load_prizepicks_board(conn, "2026-07-03", playable_only=True)
check("playable_only filters", len(playable) == 2 and all(p["playable"] for p in playable))
conn.close()

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
