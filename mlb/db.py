"""SQLite schema + helpers for the MLB props pipeline.

Core tables are append-only:
  prop_lines       — every odds snapshot we ever fetch (line history, per book)
  prop_predictions — every probability any pipeline (sim / llm / judge) produces
  prop_results     — graded outcomes, filled in the morning after
  calibration_runs — immutable fitted calibration snapshots
  calibration_buckets — empirical probability mapping for each snapshot

Everything lives in the same sportsaid.db the Flask app reads.
"""
import sqlite3
from datetime import datetime, timezone

from mlb.paths import DATABASE_PATH

DB_PATH = DATABASE_PATH
MARKETS = ("pitcher_strikeouts", "pitcher_outs")


def connect(db_path=None):
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def ensure_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS prop_lines (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      game_date     TEXT NOT NULL,
      game_pk       INTEGER,
      event_id      TEXT,
      player_name   TEXT NOT NULL,
      player_id     INTEGER,
      market        TEXT NOT NULL,
      book          TEXT NOT NULL,
      line          REAL NOT NULL,
      over_odds     INTEGER,
      under_odds    INTEGER,
      as_of         TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_prop_lines_lookup
      ON prop_lines(game_date, market, player_name);

    CREATE TABLE IF NOT EXISTS prop_predictions (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id        TEXT NOT NULL,
      created_at    TEXT NOT NULL,
      game_date     TEXT NOT NULL,
      game_pk       INTEGER,
      player_name   TEXT NOT NULL,
      player_id     INTEGER,
      market        TEXT NOT NULL,
      line          REAL NOT NULL,
      source        TEXT NOT NULL,
      p_over        REAL NOT NULL,
      mean          REAL,
      book          TEXT,
      over_odds     INTEGER,
      under_odds    INTEGER,
      market_p_over REAL,
      edge_over     REAL,
      edge_under    REAL,
      bet_side      TEXT,
      kelly_frac    REAL,
      rationale     TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_prop_pred_lookup
      ON prop_predictions(game_date, market, player_name, source);

    CREATE TABLE IF NOT EXISTS prop_results (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      prediction_id INTEGER NOT NULL REFERENCES prop_predictions(id),
      graded_at     TEXT NOT NULL,
      actual        REAL,
      outcome       TEXT,
      bet_won       INTEGER,
      profit_units  REAL,
      closing_p_over REAL,
      clv           REAL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_prop_results_pred
      ON prop_results(prediction_id);

    CREATE TABLE IF NOT EXISTS calibration_runs (
      id                INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id            TEXT NOT NULL,
      created_at        TEXT NOT NULL,
      through_date      TEXT NOT NULL,
      market            TEXT NOT NULL,
      source            TEXT NOT NULL,
      sample_size       INTEGER NOT NULL,
      mode              TEXT NOT NULL,
      brier_score       REAL,
      log_loss          REAL,
      calibration_error REAL,
      metadata_json     TEXT,
      UNIQUE(run_id, market, source)
    );
    CREATE INDEX IF NOT EXISTS idx_calibration_runs_lookup
      ON calibration_runs(market, source, through_date, created_at);

    CREATE TABLE IF NOT EXISTS calibration_buckets (
      id                     INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id                 TEXT NOT NULL,
      market                 TEXT NOT NULL,
      source                 TEXT NOT NULL,
      bucket_low             REAL NOT NULL,
      bucket_high            REAL NOT NULL,
      sample_size            INTEGER NOT NULL,
      mean_predicted         REAL NOT NULL,
      actual_rate            REAL NOT NULL,
      calibrated_probability REAL NOT NULL,
      UNIQUE(run_id, market, source, bucket_low)
    );
    CREATE INDEX IF NOT EXISTS idx_calibration_buckets_lookup
      ON calibration_buckets(run_id, market, source, bucket_low);

    CREATE TABLE IF NOT EXISTS pitcher_arsenal_games (
      game_pk       INTEGER NOT NULL,
      game_date     TEXT NOT NULL,
      pitcher_id    INTEGER NOT NULL,
      pitch_type    TEXT NOT NULL,
      pitches       INTEGER NOT NULL,
      swings        INTEGER NOT NULL,
      whiffs        INTEGER NOT NULL,
      balls_in_play INTEGER NOT NULL,
      PRIMARY KEY (game_pk, pitcher_id, pitch_type)
    );
    CREATE INDEX IF NOT EXISTS idx_pitcher_arsenal_lookup
      ON pitcher_arsenal_games(pitcher_id, game_date);

    CREATE TABLE IF NOT EXISTS batter_pitch_games (
      game_pk       INTEGER NOT NULL,
      game_date     TEXT NOT NULL,
      batter_id     INTEGER NOT NULL,
      pitch_type    TEXT NOT NULL,
      pitches       INTEGER NOT NULL,
      swings        INTEGER NOT NULL,
      whiffs        INTEGER NOT NULL,
      PRIMARY KEY (game_pk, batter_id, pitch_type)
    );
    CREATE INDEX IF NOT EXISTS idx_batter_pitch_lookup
      ON batter_pitch_games(batter_id, pitch_type, game_date);

    CREATE TABLE IF NOT EXISTS pitcher_batted_ball_games (
      game_pk       INTEGER NOT NULL,
      game_date     TEXT NOT NULL,
      pitcher_id    INTEGER NOT NULL,
      ground_balls  INTEGER NOT NULL,
      balls_in_play INTEGER NOT NULL,
      PRIMARY KEY (game_pk, pitcher_id)
    );
    CREATE INDEX IF NOT EXISTS idx_pitcher_batted_ball_lookup
      ON pitcher_batted_ball_games(pitcher_id, game_date);

    CREATE TABLE IF NOT EXISTS team_fielding_games (
      game_pk         INTEGER NOT NULL,
      game_date       TEXT NOT NULL,
      team_name       TEXT NOT NULL,
      balls_in_play   INTEGER NOT NULL,
      dp_opportunities INTEGER NOT NULL,
      double_plays    INTEGER NOT NULL,
      errors          INTEGER NOT NULL,
      PRIMARY KEY (game_pk, team_name)
    );
    CREATE INDEX IF NOT EXISTS idx_team_fielding_lookup
      ON team_fielding_games(team_name, game_date);
    """)
    conn.commit()


def replace_game_artifacts(conn, arsenal_rows, batter_rows,
                           batted_ball_rows, fielding_rows):
    """Idempotently persist postgame simulation inputs."""
    all_rows = arsenal_rows + batter_rows + batted_ball_rows + fielding_rows
    game_pks = {row["game_pk"] for row in all_rows}
    for game_pk in game_pks:
        for table in (
            "pitcher_arsenal_games", "batter_pitch_games",
            "pitcher_batted_ball_games", "team_fielding_games",
        ):
            conn.execute(f"DELETE FROM {table} WHERE game_pk = ?", (game_pk,))
    conn.executemany(
        """INSERT OR REPLACE INTO batter_pitch_games
           (game_pk, game_date, batter_id, pitch_type, pitches, swings, whiffs)
           VALUES (:game_pk, :game_date, :batter_id, :pitch_type, :pitches,
                   :swings, :whiffs)""",
        batter_rows,
    )
    conn.executemany(
        """INSERT OR REPLACE INTO pitcher_arsenal_games
           (game_pk, game_date, pitcher_id, pitch_type, pitches, swings,
            whiffs, balls_in_play)
           VALUES (:game_pk, :game_date, :pitcher_id, :pitch_type, :pitches,
                   :swings, :whiffs, :balls_in_play)""",
        arsenal_rows,
    )
    conn.executemany(
        """INSERT OR REPLACE INTO pitcher_batted_ball_games
           (game_pk, game_date, pitcher_id, ground_balls, balls_in_play)
           VALUES (:game_pk, :game_date, :pitcher_id, :ground_balls,
                   :balls_in_play)""",
        batted_ball_rows,
    )
    conn.executemany(
        """INSERT OR REPLACE INTO team_fielding_games
           (game_pk, game_date, team_name, balls_in_play, dp_opportunities,
            double_plays, errors)
           VALUES (:game_pk, :game_date, :team_name, :balls_in_play,
                   :dp_opportunities, :double_plays, :errors)""",
        fielding_rows,
    )
    conn.commit()


def pitcher_arsenal(conn, pitcher_id, as_of_date=None):
    """Cumulative pitch-mix/whiff artifact strictly before as_of_date."""
    where, args = "pitcher_id = ?", [pitcher_id]
    if as_of_date:
        where += " AND game_date < ?"
        args.append(as_of_date)
    rows = conn.execute(
        f"""SELECT pitch_type, SUM(pitches) pitches, SUM(swings) swings,
                   SUM(whiffs) whiffs, SUM(balls_in_play) balls_in_play
            FROM pitcher_arsenal_games WHERE {where}
            GROUP BY pitch_type ORDER BY pitches DESC""",
        args,
    ).fetchall()
    total = sum(row["pitches"] for row in rows)
    return [
        {
            **dict(row),
            "usage": row["pitches"] / total if total else 0.0,
            "whiff_rate": row["whiffs"] / row["swings"] if row["swings"] else 0.0,
        }
        for row in rows
    ]


def arsenal_matchup_factor(conn, pitcher_id, batter_ids, as_of_date=None,
                           swing_threshold=100):
    """Pitch-usage-weighted lineup whiff factor; neutral without evidence."""
    arsenal = pitcher_arsenal(conn, pitcher_id, as_of_date)
    batter_ids = list(dict.fromkeys(batter_ids or []))
    if not arsenal or not batter_ids:
        return 1.0
    date_clause = " AND game_date < ?" if as_of_date else ""
    placeholders = ",".join("?" for _ in batter_ids)
    args = batter_ids + ([as_of_date] if as_of_date else [])
    lineup_rows = conn.execute(
        f"""SELECT pitch_type, SUM(swings) swings, SUM(whiffs) whiffs
            FROM batter_pitch_games
            WHERE batter_id IN ({placeholders}){date_clause}
            GROUP BY pitch_type""",
        args,
    ).fetchall()
    league_rows = conn.execute(
        f"""SELECT pitch_type, SUM(swings) swings, SUM(whiffs) whiffs
            FROM batter_pitch_games WHERE 1=1{date_clause}
            GROUP BY pitch_type""",
        [as_of_date] if as_of_date else [],
    ).fetchall()
    lineup = {row["pitch_type"]: row for row in lineup_rows}
    league = {row["pitch_type"]: row for row in league_rows}
    weighted_factor = 0.0
    covered_usage = 0.0
    for pitch in arsenal:
        code = pitch["pitch_type"]
        observed = lineup.get(code)
        baseline = league.get(code)
        if not observed or not baseline or not baseline["swings"]:
            continue
        league_rate = baseline["whiffs"] / baseline["swings"]
        if league_rate <= 0:
            continue
        sample = observed["swings"]
        raw = observed["whiffs"] / sample if sample else league_rate
        weight = min(sample / float(swing_threshold), 1.0)
        credible = weight * raw + (1.0 - weight) * league_rate
        weighted_factor += pitch["usage"] * (credible / league_rate)
        covered_usage += pitch["usage"]
    if covered_usage <= 0:
        return 1.0
    # Uncovered pitch types are explicitly neutral.
    factor = weighted_factor + (1.0 - covered_usage)
    return min(max(factor, 0.85), 1.15)


def defense_context(conn, pitcher_id, team_name, as_of_date=None):
    """Pregame pitcher ground rate and fielding-team DP/error rates."""
    date_clause = " AND game_date < ?" if as_of_date else ""
    pitcher_args = [pitcher_id] + ([as_of_date] if as_of_date else [])
    team_args = [team_name] + ([as_of_date] if as_of_date else [])
    pitcher = conn.execute(
        f"""SELECT SUM(ground_balls) ground_balls,
                   SUM(balls_in_play) balls_in_play
            FROM pitcher_batted_ball_games
            WHERE pitcher_id = ?{date_clause}""",
        pitcher_args,
    ).fetchone()
    team = conn.execute(
        f"""SELECT SUM(balls_in_play) balls_in_play,
                   SUM(dp_opportunities) dp_opportunities,
                   SUM(double_plays) double_plays, SUM(errors) errors
            FROM team_fielding_games
            WHERE team_name = ?{date_clause}""",
        team_args,
    ).fetchone()
    league_pitcher = conn.execute(
        f"""SELECT SUM(ground_balls) ground_balls,
                   SUM(balls_in_play) balls_in_play
            FROM pitcher_batted_ball_games WHERE 1=1{date_clause}""",
        [as_of_date] if as_of_date else [],
    ).fetchone()
    league_team = conn.execute(
        f"""SELECT SUM(balls_in_play) balls_in_play,
                   SUM(dp_opportunities) dp_opportunities,
                   SUM(double_plays) double_plays, SUM(errors) errors
            FROM team_fielding_games WHERE 1=1{date_clause}""",
        [as_of_date] if as_of_date else [],
    ).fetchone()
    pitcher_bip = (pitcher["balls_in_play"] or 0) if pitcher else 0
    team_bip = (team["balls_in_play"] or 0) if team else 0
    dp_opps = (team["dp_opportunities"] or 0) if team else 0
    league_pitcher_bip = league_pitcher["balls_in_play"] or 0
    league_team_bip = league_team["balls_in_play"] or 0
    league_dp_opps = league_team["dp_opportunities"] or 0
    return {
        "pitcher_ground_rate": (
            pitcher["ground_balls"] / pitcher_bip if pitcher_bip else None
        ),
        "pitcher_bip": pitcher_bip,
        "team_dp_rate": team["double_plays"] / dp_opps if dp_opps else None,
        "team_dp_opportunities": dp_opps,
        "team_error_rate": team["errors"] / team_bip if team_bip else None,
        "team_bip": team_bip,
        "league_ground_rate": (
            league_pitcher["ground_balls"] / league_pitcher_bip
            if league_pitcher_bip else None
        ),
        "league_dp_rate": (
            league_team["double_plays"] / league_dp_opps
            if league_dp_opps else None
        ),
        "league_error_rate": (
            league_team["errors"] / league_team_bip
            if league_team_bip else None
        ),
    }


def insert_lines(conn, rows):
    """rows: iterable of dicts matching prop_lines columns (minus id)."""
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """INSERT INTO prop_lines
           (game_date, game_pk, event_id, player_name, player_id, market,
            book, line, over_odds, under_odds, as_of)
           VALUES (:game_date, :game_pk, :event_id, :player_name, :player_id,
                   :market, :book, :line, :over_odds, :under_odds, :as_of)""",
        rows,
    )
    conn.commit()
    return len(rows)


def insert_prediction(conn, row):
    cur = conn.execute(
        """INSERT INTO prop_predictions
           (run_id, created_at, game_date, game_pk, player_name, player_id,
            market, line, source, p_over, mean, book, over_odds, under_odds,
            market_p_over, edge_over, edge_under, bet_side, kelly_frac, rationale)
           VALUES (:run_id, :created_at, :game_date, :game_pk, :player_name,
                   :player_id, :market, :line, :source, :p_over, :mean, :book,
                   :over_odds, :under_odds, :market_p_over, :edge_over,
                   :edge_under, :bet_side, :kelly_frac, :rationale)""",
        row,
    )
    conn.commit()
    return cur.lastrowid


def latest_lines(conn, game_date, market=None):
    """Return rows from each book's newest complete board snapshot.

    Every fetcher stamps all rows from one book with the same ``as_of`` value.
    Selecting the latest timestamp per book drops moved thresholds and players
    whose markets disappeared, while preserving all ladders offered in that
    current snapshot. Partitioning by player or numeric line would retain stale
    rows indefinitely.
    """
    q = """
      SELECT * FROM prop_lines pl
      WHERE pl.game_date = ?
        AND pl.as_of = (
          SELECT MAX(as_of) FROM prop_lines
          WHERE game_date = pl.game_date AND book = pl.book
        )
    """
    args = [game_date]
    if market:
        q += " AND pl.market = ?"
        args.append(market)
    return conn.execute(q, args).fetchall()


def ungraded_predictions(conn, game_date):
    return conn.execute(
        """SELECT p.* FROM prop_predictions p
           LEFT JOIN prop_results r ON r.prediction_id = p.id
           WHERE p.game_date = ? AND r.id IS NULL""",
        (game_date,),
    ).fetchall()


if __name__ == "__main__":
    conn = connect()
    ensure_schema(conn)
    print(f"Schema ensured in {DB_PATH}")
