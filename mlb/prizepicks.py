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
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import db
import data_today
import calibration
import decision
import netfetch
import simulator
from betting_math import consensus_p_over
from factors import park_k_factor, ump_k_factor
from predict import simulate_pitcher, N_SIMS
from timeutils import slate_date, slate_today

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

# Standard PrizePicks power-play multipliers. Only the 2-pick rate is used for
# the automatic play gate; other entries are retained for reporting/tests.
POWER_PAYOUTS = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 37.5}
TARGET_LEGS = 2
EVALUATED_ODDS_TYPE = "standard"
PLAYABLE_MARGIN = float(os.getenv("PRIZEPICKS_PLAYABLE_MARGIN", "0.04"))
LINE_MAX_AGE_MINUTES = int(os.getenv("PRIZEPICKS_LINE_MAX_AGE_MINUTES", "30"))
CARD_MIN_EV = float(os.getenv("PRIZEPICKS_CARD_MIN_EV", "0.08"))
MAX_CARDS = int(os.getenv("PRIZEPICKS_MAX_CARDS", "1"))


def power_breakeven(n_legs, payouts=None):
    """Per-leg win probability at which an all-hit power play breaks even."""
    payouts = payouts or POWER_PAYOUTS
    return (1.0 / payouts[n_legs]) ** (1.0 / n_legs)


def power_ev(p, n_legs, payouts=None):
    """Net EV per unit stake, assuming independent equal-probability legs."""
    payouts = payouts or POWER_PAYOUTS
    return payouts[n_legs] * (p ** n_legs) - 1.0


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
        game_date = game_date or slate_today()
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
        game_date = game_date or slate_today()
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
        if start and slate_date(start) != game_date:
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
      raw_p_over    REAL,
      decision_p_over REAL,
      p_push        REAL NOT NULL DEFAULT 0,
      market_p_over REAL,
      pick_side     TEXT NOT NULL,
      p_pick        REAL NOT NULL,
      breakeven     REAL NOT NULL,
      edge          REAL NOT NULL,
      ev_2          REAL,
      ev_3          REAL,
      candidate     INTEGER NOT NULL DEFAULT 0,
      playable      INTEGER NOT NULL,
      mean          REAL,
      policy_version TEXT NOT NULL DEFAULT 'legacy',
      hold_reason   TEXT
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

    CREATE TABLE IF NOT EXISTS prizepicks_cards (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id         TEXT NOT NULL,
      created_at     TEXT NOT NULL,
      game_date      TEXT NOT NULL,
      pick_1_id      INTEGER NOT NULL REFERENCES prizepicks_picks(id),
      pick_2_id      INTEGER NOT NULL REFERENCES prizepicks_picks(id),
      multiplier     REAL NOT NULL,
      tie_multiplier REAL NOT NULL,
      expected_value REAL NOT NULL,
      playable       INTEGER NOT NULL,
      rationale      TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_pp_cards_pair
      ON prizepicks_cards(run_id, pick_1_id, pick_2_id);

    CREATE TABLE IF NOT EXISTS prizepicks_card_results (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      card_id        INTEGER NOT NULL REFERENCES prizepicks_cards(id),
      graded_at      TEXT NOT NULL,
      outcome        TEXT,
      profit_units   REAL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_pp_card_results_card
      ON prizepicks_card_results(card_id);
    """)
    _ensure_column(conn, "prizepicks_lines", "odds_type",
                   "TEXT NOT NULL DEFAULT 'standard'")
    _ensure_column(conn, "prizepicks_picks", "odds_type",
                   "TEXT NOT NULL DEFAULT 'standard'")
    _ensure_column(conn, "prizepicks_picks", "raw_p_over", "REAL")
    _ensure_column(conn, "prizepicks_picks", "decision_p_over", "REAL")
    _ensure_column(
        conn, "prizepicks_picks", "p_push", "REAL NOT NULL DEFAULT 0"
    )
    _ensure_column(conn, "prizepicks_picks", "market_p_over", "REAL")
    _ensure_column(
        conn, "prizepicks_picks", "candidate", "INTEGER NOT NULL DEFAULT 0"
    )
    _ensure_column(
        conn,
        "prizepicks_picks",
        "policy_version",
        "TEXT NOT NULL DEFAULT 'legacy'",
    )
    _ensure_column(conn, "prizepicks_picks", "hold_reason", "TEXT")
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
    if game_date != slate_today():
        return True
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
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


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


def evaluate_date(game_date=None, n_sims=N_SIMS):
    """Score every currently posted standard PrizePicks line."""
    game_date = game_date or slate_today()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = db.connect()
    db.ensure_schema(conn)
    ensure_schema(conn)

    if not _snapshot_is_fresh(conn, game_date):
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

    sportsbook = {}
    for row in db.active_lines(conn, game_date):
        key = (_norm(row["player_name"]), row["market"], float(row["line"]))
        sportsbook.setdefault(key, []).append(row)
    judge_approved = {}
    judge_run = conn.execute(
        """SELECT MAX(run_id) AS run_id
           FROM prop_predictions
           WHERE game_date=? AND source='judge'""",
        (game_date,),
    ).fetchone()
    if judge_run and judge_run["run_id"]:
        for row in conn.execute(
            """SELECT player_name, market, line, bet_side
               FROM prop_predictions
               WHERE game_date=? AND run_id=? AND source='judge'
                 AND bet_side IN ('over','under')""",
            (game_date, judge_run["run_id"]),
        ):
            judge_approved[
                (_norm(row["player_name"]), row["market"], float(row["line"]))
            ] = row["bet_side"]

    n_logged = 0
    for game in data_today.get_schedule(game_date):
        game_reasons = data_today.game_gate_reasons(game, game_date)
        k_park = park_k_factor(game["home_team"])
        ump_name = data_today.get_home_plate_umpire(game["game_pk"])
        k_ump = ump_k_factor(ump_name)
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
                    n_sims=n_sims, game_date=game_date)
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
                    anchor_rows=sportsbook.get(
                        (
                            _norm(pitcher["name"]),
                            row["market"],
                            float(row["line"]),
                        ),
                        [],
                    ),
                    quality_reasons=game_reasons + meta["quality_reasons"],
                    judge_side=judge_approved.get(
                        (
                            _norm(pitcher["name"]),
                            row["market"],
                            float(row["line"]),
                        )
                    ),
                )
    cards = _build_two_pick_cards(conn, run_id, game_date)
    print(
        f"[prizepicks] logged {n_logged} standard picks; "
        f"{cards} card(s) cleared exact EV"
    )
    conn.close()
    return n_logged


def _log_pick(
    conn,
    run_id,
    game_date,
    game,
    pitcher,
    line_row,
    arr,
    anchor_rows=(),
    quality_reasons=(),
    judge_side=None,
):
    line = line_row["line"]
    raw_p_over, raw_p_under, p_push = simulator.prob_over(arr, line)
    p_over = calibration.calibrated_probability(
        conn,
        raw_p_over,
        line_row["market"],
        "sim",
        as_of_date=game_date,
        policy_version=decision.POLICY_VERSION,
    )
    p_under = 1.0 - p_over
    market_p = consensus_p_over(
        [
            (row["over_odds"], row["under_odds"])
            for row in anchor_rows
        ]
    )
    priced_books = {
        row["book"]
        for row in anchor_rows
        if row["over_odds"] is not None and row["under_odds"] is not None
    }
    validation = calibration.validation_status(
        conn,
        line_row["market"],
        "sim",
        as_of_date=game_date,
        policy_version=decision.POLICY_VERSION,
    )
    quality_reasons = list(quality_reasons)
    if judge_side not in {"over", "under"}:
        quality_reasons.append(
            "independent LLM safety-judge confirmation missing"
        )
    call = decision.decide(
        p_over,
        market_p,
        book_count=len(priced_books),
        validation=validation,
        quality_reasons=quality_reasons,
        required_side=judge_side,
    )

    if call.bet_side == "over":
        pick_side = "more"
    elif call.bet_side == "under":
        pick_side = "less"
    elif p_over >= 0.5:
        pick_side = "more"
    else:
        pick_side = "less"

    no_push_mass = max(0.0, 1.0 - p_push)
    conditional_pick = p_over if pick_side == "more" else p_under
    p_pick = conditional_pick * no_push_mass
    breakeven = power_breakeven(TARGET_LEGS)
    edge = p_pick - breakeven
    candidate = bool(call.bet_side)
    hold_reason = "; ".join(call.hold_reasons)
    conn.execute(
        """INSERT INTO prizepicks_picks
           (run_id, created_at, game_date, game_pk, player_name, player_id,
            market, line, odds_type, p_over, raw_p_over, decision_p_over,
            p_push, market_p_over, pick_side, p_pick, breakeven, edge,
            ev_2, ev_3, candidate, playable, mean, policy_version, hold_reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, db.utcnow(), game_date, game["game_pk"], pitcher["name"],
         pitcher.get("id"), line_row["market"], line, line_row["odds_type"],
         p_over, raw_p_over, call.decision_p_over, p_push, market_p,
         pick_side, p_pick, breakeven, edge, None, None,
         1 if candidate else 0, 0, float(arr.mean()),
         decision.POLICY_VERSION, hold_reason))
    conn.commit()
    return 1


def _configured_card_terms():
    """Displayed PrizePicks terms must be supplied; payouts can vary by card."""
    try:
        multiplier = float(os.getenv("PRIZEPICKS_2_PICK_MULTIPLIER", ""))
        tie_multiplier = float(
            os.getenv("PRIZEPICKS_TIE_WIN_MULTIPLIER", "")
        )
    except ValueError:
        return None
    if multiplier <= 1.0 or tie_multiplier < 1.0:
        return None
    return multiplier, tie_multiplier


def two_pick_ev(first, second, multiplier, tie_multiplier):
    """Exact two-pick EV with pushes/ties, under the independence assumption."""
    w1, t1 = float(first["p_pick"]), float(first["p_push"])
    w2, t2 = float(second["p_pick"]), float(second["p_push"])
    gross = (
        multiplier * w1 * w2
        + tie_multiplier * (w1 * t2 + t1 * w2)
        + t1 * t2
    )
    return gross - 1.0


def _build_two_pick_cards(conn, run_id, game_date):
    """Create only cross-game cards and gate the actual entry-level EV."""
    terms = _configured_card_terms()
    if not terms:
        conn.execute(
            """UPDATE prizepicks_picks
               SET hold_reason = TRIM(
                 COALESCE(hold_reason || '; ', '') ||
                 'displayed 2-pick and tie multipliers not configured'
               )
               WHERE run_id = ?""",
            (run_id,),
        )
        conn.commit()
        return 0

    multiplier, tie_multiplier = terms
    picks = conn.execute(
        """SELECT * FROM prizepicks_picks
           WHERE run_id = ? AND candidate = 1
           ORDER BY id""",
        (run_id,),
    ).fetchall()
    pairs = []
    for first, second in combinations(picks, 2):
        if first["game_pk"] == second["game_pk"]:
            continue
        ev = two_pick_ev(first, second, multiplier, tie_multiplier)
        pairs.append((ev, first, second))

    # Accuracy-first exposure control: select the highest-EV non-overlapping
    # cards instead of reusing one uncertain anchor across the whole slate.
    selected = set()
    used_picks = set()
    for ev, first, second in sorted(pairs, key=lambda item: item[0], reverse=True):
        if ev < CARD_MIN_EV or len(selected) >= max(MAX_CARDS, 0):
            continue
        if first["id"] in used_picks or second["id"] in used_picks:
            continue
        selected.add((first["id"], second["id"]))
        used_picks.update((first["id"], second["id"]))

    playable_pick_ids = set()
    for ev, first, second in pairs:
        playable = (first["id"], second["id"]) in selected
        rationale = (
            f"cross-game independence approximation; displayed payout "
            f"{multiplier:g}x; tie+win {tie_multiplier:g}x; "
            f"required EV {CARD_MIN_EV:.1%}; max cards {MAX_CARDS}"
        )
        conn.execute(
            """INSERT OR IGNORE INTO prizepicks_cards
               (run_id, created_at, game_date, pick_1_id, pick_2_id,
                multiplier, tie_multiplier, expected_value, playable,
                rationale)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                db.utcnow(),
                game_date,
                first["id"],
                second["id"],
                multiplier,
                tie_multiplier,
                ev,
                1 if playable else 0,
                rationale,
            ),
        )
        if playable:
            playable_pick_ids.update((first["id"], second["id"]))

    if playable_pick_ids:
        placeholders = ",".join("?" for _ in playable_pick_ids)
        conn.execute(
            f"UPDATE prizepicks_picks SET playable=1 "
            f"WHERE id IN ({placeholders})",
            tuple(sorted(playable_pick_ids)),
        )
    conn.commit()
    return len(selected)


def grade_date(game_date=None):
    """Grade ungraded PrizePicks picks against box-score actuals."""
    game_date = game_date or slate_today()
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
    graded = 0
    for pick in picks:
        game_pk = pick["game_pk"]
        if game_pk not in actuals_by_game:
            try:
                actuals_by_game[game_pk] = (
                    data_today.get_starter_actuals(game_pk) or {}
                )
            except Exception as exc:
                print(f"[prizepicks] actuals for game {game_pk} failed ({exc})")
                actuals_by_game[game_pk] = None
        if actuals_by_game[game_pk] is None:
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
    cards_graded = _grade_cards(conn, game_date)
    print(f"[prizepicks] graded {graded} picks and {cards_graded} cards")
    conn.close()
    return graded


def _grade_cards(conn, game_date):
    cards = conn.execute(
        """SELECT c.*, r1.pick_won AS won_1, r1.outcome AS outcome_1,
                         r2.pick_won AS won_2, r2.outcome AS outcome_2
           FROM prizepicks_cards c
           JOIN prizepicks_results r1 ON r1.pick_id = c.pick_1_id
           JOIN prizepicks_results r2 ON r2.pick_id = c.pick_2_id
           LEFT JOIN prizepicks_card_results cr ON cr.card_id = c.id
           WHERE c.game_date = ? AND cr.id IS NULL""",
        (game_date,),
    ).fetchall()
    graded = 0
    tie_states = {"push", "void"}
    for card in cards:
        first_tie = card["outcome_1"] in tie_states
        second_tie = card["outcome_2"] in tie_states
        if card["won_1"] == 1 and card["won_2"] == 1:
            outcome = "win"
            profit = float(card["multiplier"]) - 1.0
        elif (
            (card["won_1"] == 1 and second_tie)
            or (card["won_2"] == 1 and first_tie)
        ):
            outcome = "tie_win"
            profit = float(card["tie_multiplier"]) - 1.0
        elif first_tie and second_tie:
            outcome = "refund"
            profit = 0.0
        else:
            outcome = "loss"
            profit = -1.0
        conn.execute(
            """INSERT INTO prizepicks_card_results
               (card_id, graded_at, outcome, profit_units)
               VALUES (?,?,?,?)""",
            (card["id"], db.utcnow(), outcome, profit),
        )
        graded += 1
    conn.commit()
    return graded


def run_all(game_date=None):
    """Refresh and evaluate only when a current snapshot was stored."""
    n = refresh(game_date)
    if n <= 0:
        print("[prizepicks] refresh returned no projections; evaluation aborted")
        return 0
    return evaluate_date(game_date)


def print_board(game_date=None):
    game_date = game_date or slate_today()
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
    breakeven = power_breakeven(TARGET_LEGS)
    print(f"\n===== PrizePicks STANDARD Board — {game_date}  "
          f"(break-even {breakeven:.1%} for a {TARGET_LEGS}-pick power) =====")
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
