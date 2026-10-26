"""PrizePicks lane for the MLB pitcher-props pipeline.

PrizePicks is a DFS pick'em product, NOT a sportsbook. It posts projections and
users choose More/Less, with payout terms determined by the lineup. This lane:

  * evaluates the simulator distribution at PrizePicks' exact current line;
  * keeps standard, goblin, and demon lines distinct using ``odds_type``; and
  * automatically evaluates only STANDARD projections because goblin/demon
    payouts are lineup-specific and cannot safely use the standard break-even.

The sportsbook lane remains the sharp-consensus/CLV anchor. PrizePicks output
lands in separate append-only tables so it never pollutes two-way odds history.

Live fetching is opt-in (PRIZEPICKS_LIVE=1) and uses an undocumented endpoint.
Set PRIZEPICKS_STATE_CODE to the jurisdiction where the board will be used.
Verify the current board and payout shown in the app before submitting a lineup.

Run: python mlb/prizepicks.py [YYYY-MM-DD]
"""
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import db
import data_today
import netfetch
import simulator
from factors import park_k_factor, ump_k_factor
from predict import simulate_pitcher, N_SIMS

BASE_DIR = Path(__file__).resolve().parent
STUB_PATHS = (
    BASE_DIR / "cache" / "stub_prizepicks_lines.json",
    BASE_DIR / "fixtures" / "stub_prizepicks_lines.sample.json",
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

# Hypothetical examples for math tests only. They are not verified live terms
# and must never authorize a pick or supply an entry EV.
POWER_PAYOUTS = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 37.5}
TARGET_LEGS = 2
EVALUATED_ODDS_TYPE = "standard"
PLAYABLE_MARGIN = float(os.getenv("PRIZEPICKS_PLAYABLE_MARGIN", "0.04"))
LINE_MAX_AGE_MINUTES = int(os.getenv("PRIZEPICKS_LINE_MAX_AGE_MINUTES", "30"))


def power_breakeven(n_legs, payouts=None):
    """Per-leg win probability at which an all-hit power play breaks even."""
    payouts = payouts or POWER_PAYOUTS
    return (1.0 / payouts[n_legs]) ** (1.0 / n_legs)


def power_ev(p, n_legs, payouts=None):
    """Net EV per unit stake, assuming independent equal-probability legs."""
    payouts = payouts or POWER_PAYOUTS
    return payouts[n_legs] * (p ** n_legs) - 1.0


def entry_ev(joint_probabilities, gross_returns):
    """EV of an actual joint W/L/P distribution and supplied payout table.

    Gross returns include returned stake. No independence assumption, invented
    companion legs or hardcoded push payout is introduced here. The caller must
    obtain terms and joint probabilities for the exact entry separately.
    """
    import math
    if not joint_probabilities or any(not math.isfinite(p) or p < 0
                                      for p in joint_probabilities.values()):
        raise ValueError("invalid joint distribution")
    if abs(sum(joint_probabilities.values()) - 1.0) > 1e-8:
        raise ValueError("joint probabilities must sum to one")
    if not set(joint_probabilities) <= set(gross_returns):
        raise ValueError("missing settlement terms, including pushes")
    if any(not math.isfinite(v) or v < 0 for v in gross_returns.values()):
        raise ValueError("invalid gross payout")
    return sum(p * gross_returns[state] for state, p in joint_probabilities.items()) - 1.0


def _normalized_odds_type(value):
    value = (value or "standard").strip().lower()
    return value if value in {"standard", "goblin", "demon"} else "unknown"


class PrizePicksFetcher:
    """Best-effort fetcher for PrizePicks' undocumented projection endpoint."""

    def __init__(self, league_id=None, base=None, state_code=None):
        self.league_id = league_id or os.getenv("PRIZEPICKS_MLB_LEAGUE_ID", "2")
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
            payload = netfetch.get_json(
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
            rows.extend(_parse_projections(payload, game_date, as_of))
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
                "odds_type": _normalized_odds_type(fixture.get("odds_type")),
                "stat_type": fixture.get("stat_type", market),
                "projection_id": fixture.get("projection_id"),
                "as_of": as_of,
            })
        return rows


def _parse_projections(payload, game_date, as_of):
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
        market = STAT_MAP.get(
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
            "odds_type": _normalized_odds_type(attributes.get("odds_type")),
            "stat_type": attributes.get("stat_type"),
            "projection_id": projection.get("id"),
            "as_of": as_of,
        })
    return rows


def get_fetcher():
    """Use the live fetcher only for an explicit truthy opt-in value."""
    live = os.getenv("PRIZEPICKS_LIVE", "").strip().lower()
    if live in {"1", "true", "yes", "on"}:
        return PrizePicksFetcher()
    return StubPrizePicksFetcher()


def _ensure_column(conn, table, column, declaration):
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def ensure_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS prizepicks_lines (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      game_date     TEXT NOT NULL,
      player_name   TEXT NOT NULL,
      player_id     INTEGER,
      market        TEXT NOT NULL,
      line          REAL NOT NULL,
      odds_type     TEXT NOT NULL DEFAULT 'standard',
      stat_type     TEXT,
      projection_id TEXT,
      as_of         TEXT NOT NULL
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
      line          REAL NOT NULL,
      odds_type     TEXT NOT NULL DEFAULT 'standard',
      p_over        REAL NOT NULL,
      pick_side     TEXT NOT NULL,
      p_pick        REAL NOT NULL,
      breakeven     REAL NOT NULL,
      edge          REAL NOT NULL,
      ev_2          REAL,
      ev_3          REAL,
      playable      INTEGER NOT NULL,
      mean          REAL
    );
    CREATE INDEX IF NOT EXISTS idx_pp_picks_lookup
      ON prizepicks_picks(game_date, market, player_name);

    CREATE TABLE IF NOT EXISTS prizepicks_results (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      pick_id       INTEGER NOT NULL REFERENCES prizepicks_picks(id),
      graded_at     TEXT NOT NULL,
      actual        REAL,
      outcome       TEXT,
      pick_won      INTEGER
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_pp_results_pick
      ON prizepicks_results(pick_id);
    """)
    _ensure_column(conn, "prizepicks_lines", "odds_type",
                   "TEXT NOT NULL DEFAULT 'standard'")
    _ensure_column(conn, "prizepicks_picks", "odds_type",
                   "TEXT NOT NULL DEFAULT 'standard'")
    for column, declaration in (("p_push", "REAL"), ("p_over_conditional", "REAL"),
                                ("entry_status", "TEXT")):
        _ensure_column(conn, "prizepicks_picks", column, declaration)
    conn.commit()


def _insert_lines(conn, rows):
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO prizepicks_lines
           (game_date, player_name, player_id, market, line, odds_type,
            stat_type, projection_id, as_of)
           VALUES (:game_date, :player_name, :player_id, :market, :line,
                   :odds_type, :stat_type, :projection_id, :as_of)""",
        rows,
    )
    conn.commit()
    return len(rows)


def _latest_lines(conn, game_date, odds_type=EVALUATED_ODDS_TYPE):
    """Return one odds tier from the newest complete PrizePicks snapshot.

    Automatic selection is intentionally limited to ``standard`` projections.
    Goblin and demon rows remain stored for audit, but their lineup-specific
    payout adjustments make the standard 3x break-even invalid.
    """
    return conn.execute(
        """SELECT * FROM prizepicks_lines
           WHERE game_date = ?
             AND as_of = (
               SELECT MAX(as_of) FROM prizepicks_lines WHERE game_date = ?)
             AND odds_type = ?""",
        (game_date, game_date, odds_type)).fetchall()


def _latest_snapshot_time(conn, game_date):
    row = conn.execute(
        "SELECT MAX(as_of) AS as_of FROM prizepicks_lines WHERE game_date=?",
        (game_date,),
    ).fetchone()
    return row["as_of"] if row else None


def _snapshot_is_fresh(conn, game_date):
    """Require a recent live-board snapshot for today's real-money run."""
    if game_date != date.today().isoformat():
        return False
    stamp = _latest_snapshot_time(conn, game_date)
    if not stamp:
        return False
    try:
        observed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    age_minutes = (datetime.now(timezone.utc) - observed).total_seconds() / 60.0
    return 0 <= age_minutes <= LINE_MAX_AGE_MINUTES


def _norm(name):
    from factors import _norm_name
    return _norm_name(name)


def refresh(game_date=None):
    """Fetch today's PrizePicks projections and append them to line history."""
    conn = db.connect()
    ensure_schema(conn)
    fetcher = get_fetcher()
    try:
        rows = fetcher.fetch(game_date)
    except Exception as exc:
        print(f"[prizepicks] refresh failed: {exc}")
        conn.close()
        return 0
    n = _insert_lines(conn, rows)
    print(f"[prizepicks] {type(fetcher).__name__}: stored {n} projection snapshots")
    conn.close()
    return n


def evaluate_date(game_date=None, n_sims=N_SIMS, research_only=False):
    """Score every currently posted standard PrizePicks line."""
    game_date = game_date or date.today().isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = db.connect()
    ensure_schema(conn)

    if not research_only and not _snapshot_is_fresh(conn, game_date):
        print(
            f"[prizepicks] refusing to evaluate {game_date}: "
            f"no projection snapshot within {LINE_MAX_AGE_MINUTES} minutes"
        )
        conn.close()
        return 0

    lines_by_player = {}
    for row in _latest_lines(conn, game_date):
        lines_by_player.setdefault(_norm(row["player_name"]), []).append(row)
    if not lines_by_player:
        print("[prizepicks] no current standard projections — run refresh first")
        conn.close()
        return 0

    n_logged = 0
    for game in data_today.get_schedule(game_date):
        if game.get("status") != "Preview" and not research_only:
            continue
        k_park = park_k_factor(game["home_team"], as_of_date=game_date)
        ump_name = data_today.get_home_plate_umpire(game["game_pk"])
        k_ump = ump_k_factor(ump_name, as_of_date=game_date)
        for side in ("home", "away"):
            pitcher = game[f"{side}_pitcher"]
            if not pitcher:
                continue
            line_rows = lines_by_player.get(_norm(pitcher["name"]))
            if not line_rows:
                continue
            opp_lineup_ids = (
                game["away_lineup"] if side == "home" else game["home_lineup"]
            )
            try:
                ks, outs, meta = simulate_pitcher(
                    pitcher, opp_lineup_ids, k_park=k_park, k_ump=k_ump,
                    n_sims=n_sims)
            except Exception as exc:
                print(f"[prizepicks] {pitcher['name']}: sim failed ({exc})")
                continue
            arrays = {"pitcher_strikeouts": ks, "pitcher_outs": outs}
            for row in line_rows:
                arr = arrays.get(row["market"])
                if arr is None:
                    continue
                n_logged += _log_pick(
                    conn, run_id, game_date, game, pitcher, row, arr,
                    allow_play=bool(meta.get("confirmed")),
                )
    print(f"[prizepicks] logged {n_logged} standard picks")
    conn.close()
    return n_logged


def _log_pick(
    conn, run_id, game_date, game, pitcher, line_row, arr, allow_play=True
):
    line = line_row["line"]
    p_conditional, p_under_conditional, p_push = simulator.prob_over(arr, line)
    p_over = p_conditional * (1 - p_push)
    p_under = p_under_conditional * (1 - p_push)
    if p_over >= p_under:
        pick_side, p_pick = "more", p_over
    else:
        pick_side, p_pick = "less", p_under
    # Legacy NOT NULL columns use neutral placeholders. No entry/payout is
    # recorded in this table, so neither an EV nor a playable flag is valid.
    breakeven, edge, playable = 0.0, 0.0, False
    conn.execute(
        """INSERT INTO prizepicks_picks
           (run_id, created_at, game_date, game_pk, player_name, player_id,
            market, line, odds_type, p_over, pick_side, p_pick, breakeven, edge,
            ev_2, ev_3, playable, mean, p_push, p_over_conditional, entry_status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, db.utcnow(), game_date, game["game_pk"], pitcher["name"],
         pitcher.get("id"), line_row["market"], line, line_row["odds_type"],
         p_over, pick_side, p_pick, breakeven, edge, None,
         None, 0, float(arr.mean()), p_push, p_conditional,
         "shadow_no_verified_entry_terms"))
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
    final_ids = {g["game_pk"] for g in data_today.get_schedule(game_date)
                 if g.get("status") == "Final"}
    graded = 0
    for pick in picks:
        game_pk = pick["game_pk"]
        if game_pk not in final_ids or not pick["player_id"]:
            continue
        if game_pk not in actuals_by_game:
            try:
                actuals_by_game[game_pk] = (
                    data_today.get_starter_actuals(game_pk) or {}
                )
            except Exception as exc:
                print(f"[prizepicks] actuals for game {game_pk} failed ({exc})")
                actuals_by_game[game_pk] = None
        if not actuals_by_game[game_pk] or len(actuals_by_game[game_pk]) != 2:
            continue
        stat = "ks" if pick["market"] == "pitcher_strikeouts" else "outs"
        box = actuals_by_game[game_pk].get(pick["player_id"])
        actual = box.get(stat) if box else None

        if actual is None:
            outcome, won = "void", None
        elif actual > pick["line"]:
            outcome, won = "more", 1 if pick["pick_side"] == "more" else 0
        elif actual < pick["line"]:
            outcome, won = "less", 1 if pick["pick_side"] == "less" else 0
        else:
            outcome, won = "push", None
        conn.execute(
            """INSERT INTO prizepicks_results
               (pick_id, graded_at, actual, outcome, pick_won)
               VALUES (?,?,?,?,?)""",
            (pick["id"], db.utcnow(), actual, outcome, won))
        conn.commit()
        graded += 1
    print(f"[prizepicks] graded {graded} picks")
    conn.close()
    return graded


def run_all(game_date=None):
    """Refresh and evaluate only when a current snapshot was stored."""
    n = refresh(game_date)
    if n <= 0:
        print("[prizepicks] refresh returned no projections; evaluation aborted")
        return 0
    return evaluate_date(game_date)


def print_board(game_date=None):
    game_date = game_date or date.today().isoformat()
    conn = db.connect()
    ensure_schema(conn)
    run = conn.execute(
        "SELECT MAX(run_id) AS run_id FROM prizepicks_picks WHERE game_date=?",
        (game_date,)).fetchone()["run_id"]
    if not run:
        print(f"[prizepicks] no picks for {game_date}")
        conn.close()
        return
    rows = conn.execute(
        """SELECT * FROM prizepicks_picks WHERE game_date=? AND run_id=?
           ORDER BY edge DESC""", (game_date, run)).fetchall()
    market_names = {
        "pitcher_strikeouts": "Strikeouts",
        "pitcher_outs": "Pitcher Outs",
    }
    print(f"\n===== PrizePicks STANDARD Board — {game_date}  "
          "(research only; no verified entry payout) =====")
    header = (f"{'Pitcher':<15}{'Stat':<13}{'PP Line':>8}  {'Pick':<6}"
              f"{'P(pick)':>8}{'Edge':>8}  {'Play?':<6}")
    print(header + "\n" + "-" * len(header))
    for row in rows:
        flag = "PLAY" if row["playable"] else "—"
        print(f"{row['player_name']:<15}"
              f"{market_names.get(row['market'], row['market']):<13}"
              f"{row['line']:>8}  {row['pick_side'].upper():<6}"
              f"{row['p_pick']:>8.1%}{row['edge']:>+8.1%}  {flag:<6}")
    conn.close()


if __name__ == "__main__":
    import sys
    selected_date = sys.argv[1] if len(sys.argv) > 1 else None
    run_all(selected_date)
    print_board(selected_date)
