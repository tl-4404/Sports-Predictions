import os
import sqlite3
import pandas as pd
from shared.data_fetching.sportsbooks import BASE_URL, OddsAPIClient
from shared.paths import DATABASE_PATH
from datetime import datetime, timezone
from dotenv import load_dotenv
from shared.paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")
# api
API_KEY = os.getenv("ODDS_API_KEY")
# odds_client route
BASE = BASE_URL
DB_PATH = DATABASE_PATH

PREFERRED_BOOKS = ("DraftKings", "FanDuel", "Caesars", "BetMGM")

# fetch current odds
def fetch_nba_odds():
    # check for API key
    if not API_KEY:
        raise RuntimeError("Set ODDS_API_KEY in your environment.")
    data = OddsAPIClient(API_KEY, base_url=BASE).get(
        "/sports/basketball_nba/odds", regions="us", markets="h2h",
        oddsFormat="american", dateFormat="iso",
    )
    # check for empty data
    if not isinstance(data, list) or not data:
        print("Warning: API returned no events.")
    return data

# format json -> pd frames
def normalize_to_frames(events_json):
    as_of = datetime.now(timezone.utc).isoformat()
    event_rows, price_rows = [], []

    for ev in events_json:
        event_rows.append({
            "event_id": ev["id"],
            "league": "NBA",
            "commence_time": ev.get("commence_time"),
            "home": ev.get("home_team"),
            "away": ev.get("away_team"),
        })

        # set bk to the the first book in preferred book with available odds
        bk = None
        for name in PREFERRED_BOOKS:
            bk = next((b for b in ev.get("bookmakers", []) if b.get("title") == name), None)
            if bk:
                break
        if not bk:
            continue

        # get all markets for bk
        markets = {m["key"]: m for m in bk.get("markets", [])}

        # moneyline (h2h)
        ml = markets.get("h2h")
        if ml:
            mm = {o["name"]: o["price"] for o in ml.get("outcomes", [])}
            if ev["home_team"] in mm:
                price_rows.append({
                    "event_id": ev["id"], 
                    "book": bk["title"], 
                    "market": "h2h",
                    "outcome": "Home", 
                    "line": None,
                    "odds_american": int(mm[ev["home_team"]]),
                    "as_of": as_of
                })
            if ev["away_team"] in mm:
                price_rows.append({
                    "event_id": ev["id"], 
                    "book": bk["title"], 
                    "market": "h2h",
                    "outcome": "Away", 
                    "line": None,
                    "odds_american": int(mm[ev["away_team"]]),
                    "as_of": as_of
                })

    return pd.DataFrame(event_rows), pd.DataFrame(price_rows)

# ensure events table format/exists
def ensure_schema(db):
    db.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS events (
      event_id TEXT PRIMARY KEY,
      league   TEXT NOT NULL,
      commence_time TEXT,
      home     TEXT NOT NULL,
      away     TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS prices (
      event_id TEXT NOT NULL,
      book     TEXT NOT NULL,
      market   TEXT NOT NULL,   
      outcome  TEXT NOT NULL,     
      line     REAL,
      odds_american INTEGER NOT NULL,
      as_of    TEXT NOT NULL,
      PRIMARY KEY (event_id, book, market, outcome),
      FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
    );
    """)

# update/insert events & odds
def upsert(df_events, df_prices):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    try:
        ensure_schema(db)
        # remove old data
        db.execute("DELETE FROM prices;")
        db.execute("DELETE FROM events;")
        # insert new events/odds to db
        if not df_events.empty:
            df_events.to_sql("events", db, if_exists="append", index=False)
        if not df_prices.empty:
            df_prices.to_sql("prices", db, if_exists="append", index=False)

        db.commit()

        # Debug
        ev = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        pr = db.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        print(f"WROTE → events: {ev}  prices: {pr}  DB: {DB_PATH}")
    finally:
        db.close()

if __name__ == "__main__":
    print("Using DB at:", DB_PATH)
    data = fetch_nba_odds()
    df_ev, df_pr = normalize_to_frames(data)
    print("events:", len(df_ev), "prices:", len(df_pr))
    upsert(df_ev, df_pr)
    print("Done.")
