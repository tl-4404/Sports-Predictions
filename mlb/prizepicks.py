"""PrizePicks lane for the MLB pitcher-props pipeline.

PrizePicks is a DFS pick'em product, NOT a sportsbook — it posts a single
projection per player+stat and you pick more/less, with the "vig" baked into a
fixed parlay payout structure (2-pick, 3-pick ... power plays). There are no
two-way American prices, so nothing to de-vig and no single-leg price to Kelly
against. This lane therefore evaluates PrizePicks lines a different way:

  * it re-uses the simulator's full outcome distribution (via
    predict.simulate_pitcher) to read P(over) at PrizePicks' EXACT line, which
    is often a different number than the sportsbook line, and
  * it compares that probability against the per-leg break-even implied by the
    power-play payout table: for an N-leg power paying M x, every leg must clear
    (1/M)^(1/N) just to break even. Legs above that bar are +EV to include.

The sportsbook lane stays the sharp/consensus + CLV anchor; this lane is the
soft DFS surface where the picks actually get placed. Output lands in its own
append-only tables (prizepicks_lines / prizepicks_picks / prizepicks_results)
so it never pollutes the two-way-odds board.

Live scraping is opt-in (PRIZEPICKS_LIVE=1) and against PrizePicks' ToS — the
default is the shipped offline stub. Payout multipliers change and run promos;
treat POWER_PAYOUTS as configurable and verify against the current board.

Run: python mlb/prizepicks.py [YYYY-MM-DD]
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
import data_today
import simulator
from factors import park_k_factor, ump_k_factor
from predict import simulate_pitcher, N_SIMS

BASE_DIR = Path(__file__).resolve().parent
# first existing path wins: your own fixture in cache/, else the shipped sample
STUB_PATHS = (
    BASE_DIR / "cache" / "stub_prizepicks_lines.json",
    BASE_DIR / "fixtures" / "stub_prizepicks_lines.sample.json",
)

# PrizePicks stat_type -> our market key. PrizePicks labels vary over time;
# keep this map lenient (lower-cased, trimmed on lookup).
STAT_MAP = {
    "pitcher strikeouts": "pitcher_strikeouts",
    "strikeouts": "pitcher_strikeouts",
    "strikeouts (pitcher)": "pitcher_strikeouts",
    "pitching outs": "pitcher_outs",
    "pitcher outs": "pitcher_outs",
    "outs": "pitcher_outs",
    "outs recorded": "pitcher_outs",
}

# PrizePicks "Power Play" (all legs must hit) payout multipliers by leg count.
# CONFIGURABLE — PrizePicks changes these and runs promos; verify against the
# current board. Flex plays (partial payouts on a miss) are not modelled in v1,
# nor are "demon"/"goblin" lines with boosted/reduced multipliers.
POWER_PAYOUTS = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 37.5}
TARGET_LEGS = 2          # reference play size for the single-leg "playable" bar
PLAYABLE_MARGIN = 0.0    # require p_pick >= breakeven + this margin to flag


# ── payout / break-even math ─────────────────────────────────────────────────

def power_breakeven(n_legs, payouts=None):
    """Per-leg win prob at which an all-hit N-leg power play breaks even.

    An N-leg power paying M x returns M * p**N in units of stake, so
    break-even is p = (1/M)**(1/N).
    """
    payouts = payouts or POWER_PAYOUTS
    return (1.0 / payouts[n_legs]) ** (1.0 / n_legs)


def power_ev(p, n_legs, payouts=None):
    """Net EV per unit stake of an N-leg power whose legs each hit with prob p
    (assumes independence). Positive means +EV."""
    payouts = payouts or POWER_PAYOUTS
    return payouts[n_legs] * (p ** n_legs) - 1.0


# ── fetchers (live scrape + offline stub) ────────────────────────────────────

class PrizePicksFetcher:
    """Scrapes PrizePicks' public projections endpoint (JSON:API format).

    Opt-in via PRIZEPICKS_LIVE — scraping is against PrizePicks' ToS and the
    endpoint sits behind Cloudflare, so this is best-effort. league_id and the
    URL are configurable because both drift over time.
    """

    def __init__(self, league_id=None, base=None):
        # MLB league id on PrizePicks (drifts over time — override via env)
        self.league_id = league_id or os.getenv("PRIZEPICKS_MLB_LEAGUE_ID", "2")
        self.base = base or os.getenv(
            "PRIZEPICKS_URL", "https://api.prizepicks.com/projections")

    def fetch(self, game_date=None):
        game_date = game_date or date.today().isoformat()
        as_of = datetime.now(timezone.utc).isoformat()
        r = requests.get(
            self.base,
            params={"league_id": self.league_id, "per_page": 1000,
                    "single_stat": "true"},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=20,
        )
        r.raise_for_status()
        payload = r.json()
        return _parse_projections(payload, game_date, as_of)


class StubPrizePicksFetcher:
    """Offline stand-in: serves fixture projections from a JSON file.

    Fixture format: [{"player_name": ..., "market"|"stat_type": ...,
    "line": ..., "player_id": ..., "projection_id": ...}, ...]
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
            market = fx.get("market") or STAT_MAP.get(
                (fx.get("stat_type") or "").strip().lower())
            if not market:
                continue
            rows.append({
                "game_date": game_date,
                "player_name": fx["player_name"],
                "player_id": fx.get("player_id"),
                "market": market,
                "line": float(fx["line"]),
                "stat_type": fx.get("stat_type", market),
                "projection_id": fx.get("projection_id"),
                "as_of": as_of,
            })
        return rows


def _parse_projections(payload, game_date, as_of):
    """Turn a PrizePicks JSON:API payload into prizepicks_lines row dicts."""
    # player id -> name from the `included` sidecar
    names = {}
    for inc in payload.get("included", []):
        if inc.get("type") in ("new_player", "player"):
            names[inc.get("id")] = (inc.get("attributes") or {}).get("name")

    rows = []
    for proj in payload.get("data", []):
        attr = proj.get("attributes", {}) or {}
        market = STAT_MAP.get((attr.get("stat_type") or "").strip().lower())
        if not market:
            continue
        # projections can span several days — keep only this game date
        start = attr.get("start_time") or ""
        if start and not start.startswith(game_date):
            continue
        rel = (proj.get("relationships", {}) or {}).get("new_player", {})
        pid = (rel.get("data") or {}).get("id")
        name = names.get(pid)
        line = attr.get("line_score")
        if not name or line is None:
            continue
        rows.append({
            "game_date": game_date,
            "player_name": name,
            "player_id": None,
            "market": market,
            "line": float(line),
            "stat_type": attr.get("stat_type"),
            "projection_id": proj.get("id"),
            "as_of": as_of,
        })
    return rows


def get_fetcher():
    """Live scraper only when explicitly opted in; stub otherwise."""
    if os.getenv("PRIZEPICKS_LIVE"):
        return PrizePicksFetcher()
    return StubPrizePicksFetcher()


# ── schema + storage (all append-only, separate from the two-way tables) ─────

def ensure_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS prizepicks_lines (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      game_date     TEXT NOT NULL,
      player_name   TEXT NOT NULL,
      player_id     INTEGER,
      market        TEXT NOT NULL,          -- pitcher_strikeouts | pitcher_outs
      line          REAL NOT NULL,
      stat_type     TEXT,                   -- raw PrizePicks label
      projection_id TEXT,                   -- PrizePicks projection id
      as_of         TEXT NOT NULL           -- snapshot time, append-only
    );
    CREATE INDEX IF NOT EXISTS idx_pp_lines_lookup
      ON prizepicks_lines(game_date, market, player_name);

    CREATE TABLE IF NOT EXISTS prizepicks_picks (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id        TEXT NOT NULL,
      created_at    TEXT NOT NULL,
      game_date     TEXT NOT NULL,
      game_pk       INTEGER,
      player_name   TEXT NOT NULL,
      player_id     INTEGER,
      market        TEXT NOT NULL,
      line          REAL NOT NULL,          -- PrizePicks projection
      p_over        REAL NOT NULL,          -- sim P(over the PP line)
      pick_side     TEXT NOT NULL,          -- 'more' | 'less'
      p_pick        REAL NOT NULL,          -- sim prob of the chosen side
      breakeven     REAL NOT NULL,          -- per-leg break-even (TARGET_LEGS)
      edge          REAL NOT NULL,          -- p_pick - breakeven
      ev_2          REAL,                   -- per-leg EV if built as a 2-pick
      ev_3          REAL,                   -- per-leg EV if built as a 3-pick
      playable      INTEGER NOT NULL,       -- 1 when edge >= PLAYABLE_MARGIN
      mean          REAL                    -- sim distribution mean
    );
    CREATE INDEX IF NOT EXISTS idx_pp_picks_lookup
      ON prizepicks_picks(game_date, market, player_name);

    CREATE TABLE IF NOT EXISTS prizepicks_results (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      pick_id       INTEGER NOT NULL REFERENCES prizepicks_picks(id),
      graded_at     TEXT NOT NULL,
      actual        REAL,                   -- actual Ks or outs; NULL = void
      outcome       TEXT,                   -- 'more' | 'less' | 'push' | 'void'
      pick_won      INTEGER                 -- 1/0/NULL (NULL on push/void)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_pp_results_pick
      ON prizepicks_results(pick_id);
    """)
    conn.commit()


def _insert_lines(conn, rows):
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO prizepicks_lines
           (game_date, player_name, player_id, market, line, stat_type,
            projection_id, as_of)
           VALUES (:game_date, :player_name, :player_id, :market, :line,
                   :stat_type, :projection_id, :as_of)""",
        rows,
    )
    conn.commit()
    return len(rows)


def _latest_lines(conn, game_date):
    """Most recent snapshot per (player, market, line) for a date."""
    return conn.execute(
        """SELECT * FROM prizepicks_lines pl
           WHERE pl.game_date = ?
             AND pl.as_of = (
               SELECT MAX(as_of) FROM prizepicks_lines
               WHERE game_date = pl.game_date AND player_name = pl.player_name
                 AND market = pl.market AND line = pl.line)""",
        (game_date,)).fetchall()


def _norm(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


# ── stages ───────────────────────────────────────────────────────────────────

def refresh(game_date=None):
    """Fetch today's PrizePicks projections and append them to line history."""
    conn = db.connect()
    ensure_schema(conn)
    fetcher = get_fetcher()
    rows = fetcher.fetch(game_date)
    n = _insert_lines(conn, rows)
    print(f"[prizepicks] {type(fetcher).__name__}: stored {n} projection snapshots")
    conn.close()
    return n


def evaluate_date(game_date=None, n_sims=N_SIMS):
    """Score every posted PrizePicks line off the simulator distribution."""
    game_date = game_date or date.today().isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = db.connect()
    ensure_schema(conn)

    lines_by_player = {}
    for row in _latest_lines(conn, game_date):
        lines_by_player.setdefault(_norm(row["player_name"]), []).append(row)
    if not lines_by_player:
        print("[prizepicks] no projections stored — run refresh first")
        conn.close()
        return 0

    n_logged = 0
    for g in data_today.get_schedule(game_date):
        k_park = park_k_factor(g["home_team"])
        ump_name = data_today.get_home_plate_umpire(g["game_pk"])
        k_ump = ump_k_factor(ump_name)
        for side in ("home", "away"):
            pitcher = g[f"{side}_pitcher"]
            if not pitcher:
                continue
            line_rows = lines_by_player.get(_norm(pitcher["name"]))
            if not line_rows:
                continue
            opp_lineup_ids = g["away_lineup"] if side == "home" else g["home_lineup"]
            try:
                ks, outs, _ = simulate_pitcher(
                    pitcher, opp_lineup_ids, k_park=k_park, k_ump=k_ump,
                    n_sims=n_sims)
            except Exception as e:
                print(f"[prizepicks] {pitcher['name']}: sim failed ({e})")
                continue
            arrays = {"pitcher_strikeouts": ks, "pitcher_outs": outs}
            for r in line_rows:
                arr = arrays.get(r["market"])
                if arr is None:
                    continue
                n_logged += _log_pick(conn, run_id, game_date, g, pitcher, r, arr)
    print(f"[prizepicks] logged {n_logged} picks")
    conn.close()
    return n_logged


def _log_pick(conn, run_id, game_date, game, pitcher, line_row, arr):
    line = line_row["line"]
    p_over, p_under, _ = simulator.prob_over(arr, line)
    if p_over >= 0.5:
        pick_side, p_pick = "more", p_over
    else:
        pick_side, p_pick = "less", p_under
    be = power_breakeven(TARGET_LEGS)
    edge = p_pick - be
    conn.execute(
        """INSERT INTO prizepicks_picks
           (run_id, created_at, game_date, game_pk, player_name, player_id,
            market, line, p_over, pick_side, p_pick, breakeven, edge,
            ev_2, ev_3, playable, mean)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, db.utcnow(), game_date, game["game_pk"], pitcher["name"],
         pitcher.get("id"), line_row["market"], line, p_over, pick_side,
         p_pick, be, edge, power_ev(p_pick, 2), power_ev(p_pick, 3),
         1 if edge >= PLAYABLE_MARGIN else 0, float(arr.mean())))
    conn.commit()
    return 1


def grade_date(game_date=None):
    """Grade ungraded PrizePicks picks against box-score actuals."""
    game_date = game_date or date.today().isoformat()
    conn = db.connect()
    ensure_schema(conn)
    picks = conn.execute(
        """SELECT p.* FROM prizepicks_picks p
           LEFT JOIN prizepicks_results r ON r.pick_id = p.id
           WHERE p.game_date = ? AND r.id IS NULL""",
        (game_date,)).fetchall()
    if not picks:
        print(f"[prizepicks] nothing ungraded for {game_date}")
        conn.close()
        return 0

    actuals_by_game = {}
    n = 0
    for pk in picks:
        gp = pk["game_pk"]
        if gp not in actuals_by_game:
            try:
                actuals_by_game[gp] = data_today.get_starter_actuals(gp) or {}
            except Exception as e:
                print(f"[prizepicks] actuals for game {gp} failed ({e})")
                actuals_by_game[gp] = {}
        stat = "ks" if pk["market"] == "pitcher_strikeouts" else "outs"
        box = actuals_by_game[gp].get(pk["player_id"])
        actual = box.get(stat) if box else None

        if actual is None:
            outcome, won = "void", None
        elif actual > pk["line"]:
            outcome, won = "more", 1 if pk["pick_side"] == "more" else 0
        elif actual < pk["line"]:
            outcome, won = "less", 1 if pk["pick_side"] == "less" else 0
        else:
            outcome, won = "push", None
        conn.execute(
            """INSERT INTO prizepicks_results
               (pick_id, graded_at, actual, outcome, pick_won)
               VALUES (?,?,?,?,?)""",
            (pk["id"], db.utcnow(), actual, outcome, won))
        conn.commit()
        n += 1
    print(f"[prizepicks] graded {n} picks")
    conn.close()
    return n


def run_all(game_date=None):
    """refresh -> evaluate for the given date."""
    refresh(game_date)
    return evaluate_date(game_date)


# ── board renderer (stdout) ──────────────────────────────────────────────────

def print_board(game_date=None):
    game_date = game_date or date.today().isoformat()
    conn = db.connect()
    ensure_schema(conn)
    run = conn.execute(
        "SELECT MAX(run_id) r FROM prizepicks_picks WHERE game_date=?",
        (game_date,)).fetchone()["r"]
    if not run:
        print(f"[prizepicks] no picks for {game_date}")
        conn.close()
        return
    rows = conn.execute(
        """SELECT * FROM prizepicks_picks WHERE game_date=? AND run_id=?
           ORDER BY edge DESC""", (game_date, run)).fetchall()
    mkt = {"pitcher_strikeouts": "Strikeouts", "pitcher_outs": "Pitcher Outs"}
    be = power_breakeven(TARGET_LEGS)
    print(f"\n===== PrizePicks Board — {game_date}  "
          f"(break-even {be:.1%} for a {TARGET_LEGS}-pick power) =====")
    hdr = (f"{'Pitcher':<15}{'Stat':<13}{'PP Line':>8}  {'Pick':<6}"
           f"{'P(pick)':>8}{'Edge':>8}  {'Play?':<6}")
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        flag = "PLAY" if r["playable"] else "—"
        print(f"{r['player_name']:<15}{mkt.get(r['market'], r['market']):<13}"
              f"{r['line']:>8}  {r['pick_side'].upper():<6}"
              f"{r['p_pick']:>8.1%}{r['edge']:>+8.1%}  {flag:<6}")
    conn.close()


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else None
    run_all(d)
    print_board(d)
