"""Offline tests for the odds-source toggle, the DraftKings scraper parser,
and the PrizePicks interface loader.

No network, no pytest:  python tests/test_interface_and_toggle.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlb import db
from mlb.providers import odds as odds_props
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

rows = odds_props.DraftKingsScraper()._parse(
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

# ── FanDuel scraper parser (offline payload) ─────────────────────────────────
print("fanduel scraper parser:")

FD_PAYLOAD = {
    "attachments": {
        "events": {
            "9001": {"openDate": "2026-07-03T23:05:00Z"},
            "9002": {"openDate": "2026-07-04T18:10:00Z"},  # other day
        },
        "markets": {
            "m1": {"eventId": 9001, "marketType": "PITCHER_STRIKEOUTS",
                   "marketName": "Gerrit Cole - Total Strikeouts", "runners": [
                       {"runnerName": "Over 6.5", "handicap": 6.5,
                        "winRunnerOdds": {"americanDisplayOdds": {"americanOdds": -120}}},
                       {"runnerName": "Under 6.5", "handicap": 6.5,
                        "winRunnerOdds": {"americanDisplayOdds": {"americanOdds": 100}}},
                   ]},
            "m2": {"eventId": 9001, "marketType": "PITCHING_OUTS",
                   "marketName": "Gerrit Cole - Pitching Outs", "runners": [
                       {"runnerName": "Over", "handicap": 17.5,
                        "winRunnerOdds": {"americanDisplayOdds": {"americanOdds": -110}}},
                       {"runnerName": "Under", "handicap": 17.5,
                        "winRunnerOdds": {"americanDisplayOdds": {"americanOdds": -110}}},
                   ]},
            "m3": {"eventId": 9002, "marketType": "PITCHER_STRIKEOUTS",
                   "marketName": "Tomorrow Guy - Total Strikeouts", "runners": [
                       {"runnerName": "Over", "handicap": 5.5,
                        "winRunnerOdds": {"americanDisplayOdds": {"americanOdds": -105}}}]},
            "m4": {"eventId": 9001, "marketType": "TOTAL_BASES",
                   "marketName": "Someone - Total Bases", "runners": []},  # ignored
        },
    },
}

fd = odds_props.FanDuelScraper()._parse(FD_PAYLOAD, "2026-07-03",
                                        "2026-07-03T12:00:00Z")
check("fanduel parsed today's markets (2 rows)", len(fd) == 2, f"({len(fd)})")
fd_by = {r["market"]: r for r in fd}
fk = fd_by.get("pitcher_strikeouts")
check("fanduel player from market name", fk and fk["player_name"] == "Gerrit Cole")
check("fanduel line + odds parsed",
      fk and fk["line"] == 6.5 and fk["over_odds"] == -120 and fk["under_odds"] == 100,
      f"({fk['over_odds']}/{fk['under_odds']})" if fk else "")
check("fanduel book label", fk and fk["book"] == "FanDuel")
check("fanduel future-day market excluded",
      all(r["player_name"] != "Tomorrow Guy" for r in fd))

# ── multi-book consensus ─────────────────────────────────────────────────────
print("multi-book consensus:")
agg = odds_props.SportsbookScraperFetcher()
check("aggregator defaults to DK + FD",
      [s.BOOK for s in agg.scrapers] == ["DraftKings", "FanDuel"])
check("SCRAPE_BOOKS overrides book set",
      [s.BOOK for s in odds_props.SportsbookScraperFetcher(books="fanduel").scrapers]
      == ["FanDuel"])
# two books on the same prop de-vig to a blended consensus between them
from mlb.prediction.betting_math import consensus_p_over, devig_two_way
p_dk, _ = devig_two_way(k["over_odds"], k["under_odds"])       # -115/-105
p_fd, _ = devig_two_way(fk["over_odds"], fk["under_odds"])     # -120/+100
cons = consensus_p_over([(k["over_odds"], k["under_odds"]),
                         (fk["over_odds"], fk["under_odds"])])
check("consensus averages the two books",
      abs(cons - (p_dk + p_fd) / 2) < 1e-9,
      f"(dk={p_dk:.3f}, fd={p_fd:.3f}, cons={cons:.3f})")

# ── PrizePicks interface loader ──────────────────────────────────────────────
print("prizepicks interface loader:")

conn = db.connect(Path(tempfile.mkdtemp()) / "iface.db")

# missing table -> [] (lane never ran)
check("missing table -> empty board",
      webapp.load_prizepicks_board(conn, "2026-07-03") == [])

# seed prizepicks_picks directly
from providers import prizepicks as pp
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

# ── PrizePicks in the lane scorecard ─────────────────────────────────────────
print("prizepicks lane scorecard:")

conn = db.connect(Path(tempfile.mkdtemp()) / "score.db")
pp.ensure_schema(conn)
today = webapp.date.today().isoformat()
BE = pp.power_breakeven(2)  # ~0.5774
# 5 playable graded legs: 4 wins / 1 loss (80% > 57.7% break-even -> +ROI);
# plus a non-playable pick and a push, which must be excluded from the tally.
legs = [("more", 1), ("less", 1), ("more", 1), ("more", 1), ("less", 0)]
for i, (side, won) in enumerate(legs):
    pid = conn.execute(
        """INSERT INTO prizepicks_picks
           (run_id, created_at, game_date, game_pk, player_name, player_id,
            market, line, p_over, pick_side, p_pick, breakeven, edge,
            ev_2, ev_3, playable, mean)
           VALUES ('r1',?,?,1,?,?, 'pitcher_strikeouts',5.5,0.6,?,0.6,?,0.05,0,0,1,0)""",
        (db.utcnow(), today, f"P{i}", i, side, BE)).lastrowid
    conn.execute(
        """INSERT INTO prizepicks_results (pick_id, graded_at, actual, outcome, pick_won)
           VALUES (?,?,?,?,?)""",
        (pid, db.utcnow(), 6, side, won))
# excluded: a non-playable pick, and a playable push (pick_won NULL)
pid = conn.execute(
    """INSERT INTO prizepicks_picks
       (run_id, created_at, game_date, game_pk, player_name, player_id, market,
        line, p_over, pick_side, p_pick, breakeven, edge, ev_2, ev_3, playable, mean)
       VALUES ('r1',?,?,1,'NotPlayable',99,'pitcher_outs',15.5,0.52,'more',0.52,?,-0.05,0,0,0,0)""",
    (db.utcnow(), today, BE)).lastrowid
conn.execute("""INSERT INTO prizepicks_results (pick_id, graded_at, actual, outcome, pick_won)
                VALUES (?,?,?,?,?)""", (pid, db.utcnow(), 16, "more", 1))
pid = conn.execute(
    """INSERT INTO prizepicks_picks
       (run_id, created_at, game_date, game_pk, player_name, player_id, market,
        line, p_over, pick_side, p_pick, breakeven, edge, ev_2, ev_3, playable, mean)
       VALUES ('r1',?,?,1,'Pushy',98,'pitcher_strikeouts',6.0,0.6,'more',0.6,?,0.05,0,0,1,0)""",
    (db.utcnow(), today, BE)).lastrowid
conn.execute("""INSERT INTO prizepicks_results (pick_id, graded_at, actual, outcome, pick_won)
                VALUES (?,?,?,?,?)""", (pid, db.utcnow(), 6, "push", None))
conn.commit()

board = webapp.load_lane_scoreboard(conn, days=14)
pp_lane = [r for r in board if r["source"] == "prizepicks"]
check("prizepicks row present even with no sportsbook lanes", len(pp_lane) == 1)
row = pp_lane[0]
check("only playable, decided legs counted (5)", row["bets"] == 5, f"({row['bets']})")
check("wins counted (4)", row["wins"] == 4, f"({row['wins']})")
check("hit rate 80%", abs(row["hit_rate"] - 0.8) < 1e-9, f"({row['hit_rate']:.3f})")
expected_roi = (4 * (1 / BE - 1) - 1) / 5
check("ROI scored vs break-even", abs(row["roi"] - expected_roi) < 1e-9,
      f"({row['roi']:+.3f})")
check("beating break-even -> positive ROI", row["roi"] > 0)
check("no CLV for pick'em", row["avg_clv"] is None and row["clv_beat_rate"] is None)
conn.close()

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
