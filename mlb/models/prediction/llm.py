"""LLM lanes for the MLB props pipeline (Claude).

Three stages, run after the simulator lane:

  facts    — one web-search call per game for QUALITATIVE context only
             (injury chatter, velocity dips, skipped starts, bullpen state,
             roof/ump notes). Hard numbers (lines, odds, stats) always come
             from APIs — never from the LLM.
  predict  — an independent Claude probability per posted line. Sees the
             fact packet and the LINE, but not the odds price and not the
             simulator's output, so the lane stays uncorrelated.
  judge    — combines sim + llm + de-vigged market. Bounded authority:
             the final probability is clamped to the config blend ± JUDGE_BAND,
             and the judge may veto (pass) but not invent bets below MIN_EDGE.

All rows land in prop_predictions under source='llm' / source='judge' with
the SAME run_id as the sim rows they consumed, so grading and the UI can
compare lanes per prop.

Requires ANTHROPIC_API_KEY (skips gracefully when absent).
"""
import json
import os
import re
from datetime import date

from mlb import db
from mlb.models.prediction import form
from shared.betting.math import consensus_p_over, kelly_fraction
from mlb.models.prediction.factors import park_k_factor, ump_k_factor
from mlb.data_fetching.providers import mlb_api

MODEL = os.getenv("SPORTSAID_LLM_MODEL", "claude-opus-4-8")
MIN_EDGE = 0.03
W_SIM = 0.70          # prior blend weight on the simulator
JUDGE_BAND = 0.05     # judge may move the blended prob at most this far
MAX_SEARCHES_PER_GAME = 5

_anthropic_client = None


def get_client():
    """Lazy import so serving boxes without the SDK can still run sim lanes."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install anthropic to enable the LLM lanes") from e
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to enable the LLM lanes")
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def set_client(client):
    """Test hook: inject a fake client."""
    global _anthropic_client
    _anthropic_client = client


def _text_of(response):
    return next((b.text for b in response.content if b.type == "text"), "")


def _lenient_json(text):
    """Parse JSON from a response that may wrap it in prose/code fences."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ── stage 1: facts ───────────────────────────────────────────────────────────

FACTS_SYSTEM = (
    "You are a baseball research assistant for a betting pipeline. You gather "
    "QUALITATIVE context only — never quote betting lines, odds, or statistics "
    "as facts (the pipeline gets those from APIs). Search for recent, "
    "credible reporting (beat writers, official injury reports). If you find "
    "nothing noteworthy, say so — an empty flags list is a good answer. "
    "Never invent information; every flag must trace to something you read."
)

FACTS_PROMPT = """Game today ({game_date}): {away} at {home}.
Probable starters: {away_pitcher} (away), {home_pitcher} (home).

Search for news from the last 3 days that could affect how long each starter
pitches or how many strikeouts they get: injury/illness chatter, velocity or
mechanics concerns, skipped or extra rest, planned short outings or pitch
limits, opposing bullpen fatigue (affects how deep the starter is pushed),
expected lineup rest days for key hitters, roof/weather notes, announced
home-plate umpire.

Respond with ONLY this JSON:
{{"pitchers": [{{"name": "...", "flags": ["short factual flag", ...],
"outlook": "neutral|positive|negative"}}],
"game_notes": ["..."], "sources": ["url or outlet", ...]}}"""


def ensure_facts_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS llm_facts (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      game_date  TEXT NOT NULL,
      game_pk    INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      facts_json TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_llm_facts ON llm_facts(game_date, game_pk);
    """)
    conn.commit()


def fetch_facts(game, client=None):
    """One web-search call per game; returns the parsed facts dict."""
    client = client or get_client()
    prompt = FACTS_PROMPT.format(
        game_date=game["game_date"], away=game["away_team"], home=game["home_team"],
        away_pitcher=(game["away_pitcher"] or {}).get("name", "TBD"),
        home_pitcher=(game["home_pitcher"] or {}).get("name", "TBD"),
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=FACTS_SYSTEM,
        tools=[{"type": "web_search_20260209", "name": "web_search",
                "max_uses": MAX_SEARCHES_PER_GAME}],
        messages=[{"role": "user", "content": prompt}],
    )
    facts = _lenient_json(_text_of(resp))
    return facts or {"pitchers": [], "game_notes": [], "sources": []}


def facts_stage(game_date=None):
    game_date = game_date or date.today().isoformat()
    conn = db.connect()
    ensure_facts_schema(conn)
    games = mlb_api.get_schedule(game_date)
    n = 0
    for g in games:
        # skip games we already researched this date (facts are append-only;
        # re-running the stage refreshes only games without a snapshot)
        row = conn.execute(
            "SELECT 1 FROM llm_facts WHERE game_date=? AND game_pk=?",
            (game_date, g["game_pk"])).fetchone()
        if row:
            continue
        try:
            facts = fetch_facts(g)
        except Exception as e:
            print(f"[llm.facts] game {g['game_pk']}: {e}")
            continue
        conn.execute(
            "INSERT INTO llm_facts (game_date, game_pk, created_at, facts_json)"
            " VALUES (?, ?, ?, ?)",
            (game_date, g["game_pk"], db.utcnow(), json.dumps(facts)))
        conn.commit()
        n += 1
    print(f"[llm.facts] researched {n} games")
    conn.close()
    return n


def _facts_for(conn, game_date, game_pk):
    row = conn.execute(
        """SELECT facts_json FROM llm_facts
           WHERE game_date=? AND game_pk=? ORDER BY id DESC LIMIT 1""",
        (game_date, game_pk)).fetchone()
    return json.loads(row["facts_json"]) if row else None


def _pitcher_flags(facts, pitcher_name):
    if not facts:
        return {"flags": [], "outlook": "neutral", "game_notes": []}
    def norm(s):
        return "".join(c for c in (s or "").lower() if c.isalpha())
    for p in facts.get("pitchers", []):
        if norm(p.get("name")) == norm(pitcher_name):
            return {"flags": p.get("flags", []),
                    "outlook": p.get("outlook", "neutral"),
                    "game_notes": facts.get("game_notes", [])}
    return {"flags": [], "outlook": "neutral",
            "game_notes": facts.get("game_notes", [])}


# ── stage 2: independent Claude predictor ───────────────────────────────────

PREDICT_SYSTEM = (
    "You are an expert MLB handicapper producing calibrated probabilities for "
    "pitcher props. You are given per-plate-appearance rates and context; you "
    "see the posted line but NOT the betting odds, and you must reason from "
    "the data given plus general baseball knowledge. Be honest about "
    "uncertainty: most props resolve near 50/50, and probabilities outside "
    "0.35-0.65 need strong justification. Think about how many batters the "
    "starter will face (pitch efficiency, leash, bullpen context) and the "
    "matchup K rates."
)

PREDICT_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "props": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "market": {"type": "string",
                                   "enum": ["pitcher_strikeouts", "pitcher_outs"]},
                        "line": {"type": "number"},
                        "p_over": {"type": "number"},
                        "confidence": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["market", "line", "p_over", "confidence", "rationale"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["props"],
        "additionalProperties": False,
    },
}

PREDICT_PROMPT = """{game_date}: {pitcher} ({hand}HP, {team_side} starter) faces this lineup.

Pitcher season per-PA rates (shrunk to league avg): K {pk:.3f}, BB {pbb:.3f}, hit {phit:.3f}
Recent form (stuff & velocity): {form_summary}
Recent start pitch counts: {pitch_counts}
Opposing lineup per-PA K rates vs {hand}HP (slots 1-9): {lineup_ks}
(league avg K rate: 0.223)
Park K factor: {k_park:.2f}   Home-plate umpire K factor: {k_ump:.2f} ({ump_name})
(a K factor above 1.0 means more strikeouts than average)
Lineup status: {lineup_status}

Recent news flags for this pitcher: {flags}
Game notes: {game_notes}

Estimate P(over) for each posted prop line:
{lines_block}

Return one entry per line above. p_over is the probability the OVER hits
(pushes excluded). confidence is 0-1 for how much you trust your own number."""


def llm_predict_stage(game_date=None, run_id=None):
    """Log an independent Claude probability for every posted line."""
    game_date = game_date or date.today().isoformat()
    conn = db.connect()
    db.ensure_schema(conn)
    ensure_facts_schema(conn)
    client = get_client()

    run_id = run_id or _latest_sim_run(conn, game_date)
    if not run_id:
        print("[llm.predict] no sim run found for date — run predict stage first")
        conn.close()
        return 0

    lines_by_player = {}
    for row in db.latest_lines(conn, game_date):
        lines_by_player.setdefault(_norm(row["player_name"]), []).append(row)

    n = 0
    for g in mlb_api.get_schedule(game_date):
        facts = _facts_for(conn, game_date, g["game_pk"])
        k_park = park_k_factor(g["home_team"])
        ump_name = mlb_api.get_home_plate_umpire(g["game_pk"])
        k_ump = ump_k_factor(ump_name)
        for side in ("home", "away"):
            pitcher = g[f"{side}_pitcher"]
            if not pitcher:
                continue
            line_rows = lines_by_player.get(_norm(pitcher["name"]), [])
            if not line_rows:
                continue
            try:
                n += _llm_predict_pitcher(
                    conn, client, run_id, game_date, g, side, pitcher,
                    line_rows, _pitcher_flags(facts, pitcher["name"]),
                    k_park, k_ump, ump_name)
            except Exception as e:
                print(f"[llm.predict] {pitcher['name']}: {e}")
    print(f"[llm.predict] logged {n} predictions")
    conn.close()
    return n


def _llm_predict_pitcher(conn, client, run_id, game_date, game, side, pitcher,
                         line_rows, flags, k_park, k_ump, ump_name):
    profile = mlb_api.get_pitcher_profile(pitcher["id"])
    opp_ids = game["away_lineup"] if side == "home" else game["home_lineup"]
    if opp_ids:
        lineup_ks = [round(mlb_api.get_batter_rates(b, pitcher["hand"])["k"], 3)
                     for b in opp_ids[:9]]
        lineup_status = "confirmed lineup"
    else:
        lineup_ks = [mlb_api.LEAGUE["k"]] * 9
        lineup_status = "lineup not posted — league-average assumed"

    distinct = sorted({(r["market"], r["line"]) for r in line_rows})
    lines_block = "\n".join(
        f"- {m}: line {ln}" for m, ln in distinct)

    prompt = PREDICT_PROMPT.format(
        game_date=game_date, pitcher=pitcher["name"], hand=pitcher["hand"],
        team_side=side, pk=profile["k"], pbb=profile["bb"], phit=profile["hit"],
        form_summary=form.form_summary(pitcher["id"]) or "n/a",
        pitch_counts=profile["recent_pitch_counts"] or "unknown",
        lineup_ks=lineup_ks, k_park=k_park, k_ump=k_ump,
        ump_name=ump_name or "TBD", lineup_status=lineup_status,
        flags=flags["flags"] or "none found",
        game_notes=flags["game_notes"] or "none",
        lines_block=lines_block,
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=PREDICT_SYSTEM,
        output_config={"format": PREDICT_SCHEMA},
        messages=[{"role": "user", "content": prompt}],
    )
    data = json.loads(_text_of(resp))

    by_key = {}
    for r in line_rows:
        by_key.setdefault((r["market"], r["line"]), []).append(r)

    logged = 0
    for prop in data.get("props", []):
        key = (prop["market"], float(prop["line"]))
        books = by_key.get(key)
        if not books:
            continue
        p_over = min(max(float(prop["p_over"]), 0.02), 0.98)
        market_p = consensus_p_over(
            [(b["over_odds"], b["under_odds"]) for b in books])
        row = _priced_row(
            game_date, game, pitcher, key[0], key[1], p_over, books, market_p,
            run_id=run_id, source="llm",
            rationale=json.dumps({"rationale": prop.get("rationale"),
                                  "confidence": prop.get("confidence")}),
        )
        db.insert_prediction(conn, row)
        logged += 1
    return logged


# ── stage 3: bounded judge ───────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are the final decision layer of a betting pipeline. Two independent "
    "estimates exist for each prop: a plate-appearance Monte Carlo simulator "
    "(strong on rates and distributions, blind to news) and an LLM handicapper "
    "(sees news, weaker calibration). You also see the de-vigged market "
    "consensus — treat it as a sharp opinion that is usually right. Your job: "
    "adjust the blended probability only when the news/flags justify it, and "
    "veto bets when something the simulator cannot see (injury, short leash, "
    "lineup rest) undermines the edge. You cannot move the probability far — "
    "the pipeline clamps you to a narrow band. Passing is a good decision."
)

JUDGE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "market": {"type": "string",
                                   "enum": ["pitcher_strikeouts", "pitcher_outs"]},
                        "line": {"type": "number"},
                        "final_p_over": {"type": "number"},
                        "action": {"type": "string", "enum": ["bet", "pass"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["market", "line", "final_p_over", "action",
                                 "rationale"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    },
}

JUDGE_PROMPT = """{game_date}: props for {pitcher}.

News flags: {flags}
Game notes: {game_notes}

Per prop — simulator P(over), LLM P(over) (with its confidence), de-vigged
market P(over), and the blended prior (weights: sim {w_sim:.0%} / llm {w_llm:.0%}):
{props_block}

For each prop return final_p_over (your best calibrated estimate; it will be
clamped to blend ± {band:.0%}) and action: "bet" if you endorse betting when
the edge clears the threshold, "pass" to veto regardless of edge (say why)."""


def judge_stage(game_date=None, run_id=None):
    game_date = game_date or date.today().isoformat()
    conn = db.connect()
    db.ensure_schema(conn)
    ensure_facts_schema(conn)
    client = get_client()

    run_id = run_id or _latest_sim_run(conn, game_date)
    if not run_id:
        print("[llm.judge] no sim run found for date")
        conn.close()
        return 0

    rows = conn.execute(
        """SELECT * FROM prop_predictions
           WHERE game_date=? AND run_id=? AND source IN ('sim','llm')""",
        (game_date, run_id)).fetchall()

    # group per pitcher → per (market, line) with both lanes
    by_pitcher = {}
    for r in rows:
        by_pitcher.setdefault(r["player_name"], {}) \
                  .setdefault((r["market"], r["line"]), {})[r["source"]] = r

    lines_by_player = {}
    for row in db.latest_lines(conn, game_date):
        key = (_norm(row["player_name"]), row["market"], row["line"])
        lines_by_player.setdefault(key, []).append(row)

    n = 0
    for pitcher_name, props in by_pitcher.items():
        # judge only props that have BOTH lanes and a market price
        candidates = {}
        for (market, line), lanes in props.items():
            sim, llm_row = lanes.get("sim"), lanes.get("llm")
            if not sim or not llm_row or sim["market_p_over"] is None:
                continue
            candidates[(market, line)] = (sim, llm_row)
        if not candidates:
            continue

        # facts lookup by game_pk of any sim row
        any_sim = list(candidates.values())[0][0]
        facts = _facts_for(conn, game_date, any_sim["game_pk"])
        flags = _pitcher_flags(facts, pitcher_name)

        try:
            n += _judge_pitcher(conn, client, run_id, game_date, pitcher_name,
                                candidates, flags, lines_by_player)
        except Exception as e:
            print(f"[llm.judge] {pitcher_name}: {e}")
    print(f"[llm.judge] logged {n} decisions")
    conn.close()
    return n


def _judge_pitcher(conn, client, run_id, game_date, pitcher_name, candidates,
                   flags, lines_by_player):
    props_block = []
    blends = {}
    for (market, line), (sim, llm_row) in sorted(candidates.items()):
        blend = W_SIM * sim["p_over"] + (1 - W_SIM) * llm_row["p_over"]
        blends[(market, line)] = blend
        llm_meta = _lenient_json(llm_row["rationale"] or "") or {}
        props_block.append(
            f"- {market} line {line}: sim {sim['p_over']:.3f}, "
            f"llm {llm_row['p_over']:.3f} (conf {llm_meta.get('confidence', '?')}), "
            f"market {sim['market_p_over']:.3f}, blend {blend:.3f}")

    prompt = JUDGE_PROMPT.format(
        game_date=game_date, pitcher=pitcher_name,
        flags=flags["flags"] or "none found",
        game_notes=flags["game_notes"] or "none",
        w_sim=W_SIM, w_llm=1 - W_SIM, band=JUDGE_BAND,
        props_block="\n".join(props_block),
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=JUDGE_SYSTEM,
        output_config={"format": JUDGE_SCHEMA},
        messages=[{"role": "user", "content": prompt}],
    )
    data = json.loads(_text_of(resp))

    logged = 0
    for dec in data.get("decisions", []):
        key = (dec["market"], float(dec["line"]))
        if key not in candidates:
            continue
        sim, llm_row = candidates[key]
        blend = blends[key]
        # bounded authority: clamp to blend ± band, no matter what it said
        p_final = min(max(float(dec["final_p_over"]), blend - JUDGE_BAND),
                      blend + JUDGE_BAND)
        p_final = min(max(p_final, 0.02), 0.98)

        books = lines_by_player.get((_norm(pitcher_name), key[0], key[1]), [])
        market_p = sim["market_p_over"]
        veto = dec["action"] == "pass"
        row = _priced_row(
            game_date, {"game_pk": sim["game_pk"]},
            {"name": pitcher_name, "id": sim["player_id"]},
            key[0], key[1], p_final, books, market_p,
            run_id=run_id, source="judge",
            rationale=json.dumps({"action": dec["action"],
                                  "rationale": dec.get("rationale"),
                                  "blend": round(blend, 4)}),
            suppress_bet=veto,
        )
        db.insert_prediction(conn, row)
        logged += 1
    return logged


# ── shared helpers ───────────────────────────────────────────────────────────

def _norm(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalpha())


def _latest_sim_run(conn, game_date):
    row = conn.execute(
        "SELECT MAX(run_id) r FROM prop_predictions WHERE game_date=? AND source='sim'",
        (game_date,)).fetchone()
    return row["r"] if row else None


def _priced_row(game_date, game, pitcher, market, line, p_over, books,
                market_p, run_id, source, rationale, suppress_bet=False):
    """Build a prop_predictions row with edge / bet side / Kelly stake."""
    from shared.betting.math import american_to_decimal

    edge_over = p_over - market_p if market_p is not None else None
    edge_under = (1 - p_over) - (1 - market_p) if market_p is not None else None

    bet_side, kelly, best = None, 0.0, None
    if not suppress_bet and edge_over is not None:
        if edge_over >= MIN_EDGE and edge_over >= (edge_under or 0):
            priced = [b for b in books if b["over_odds"] is not None]
            best = max(priced, key=lambda b: american_to_decimal(b["over_odds"]),
                       default=None)
            if best:
                bet_side = "over"
                kelly = kelly_fraction(p_over, best["over_odds"])
        elif edge_under is not None and edge_under >= MIN_EDGE:
            priced = [b for b in books if b["under_odds"] is not None]
            best = max(priced, key=lambda b: american_to_decimal(b["under_odds"]),
                       default=None)
            if best:
                bet_side = "under"
                kelly = kelly_fraction(1 - p_over, best["under_odds"])
    if best is None and books:
        best = books[0]

    return {
        "run_id": run_id, "created_at": db.utcnow(), "game_date": game_date,
        "game_pk": game.get("game_pk"), "player_name": pitcher["name"],
        "player_id": pitcher.get("id"), "market": market, "line": line,
        "source": source, "p_over": p_over, "mean": None,
        "book": best["book"] if best else None,
        "over_odds": best["over_odds"] if best else None,
        "under_odds": best["under_odds"] if best else None,
        "market_p_over": market_p, "edge_over": edge_over,
        "edge_under": edge_under, "bet_side": bet_side, "kelly_frac": kelly,
        "rationale": rationale,
    }


def run_all(game_date=None):
    """facts → llm predict → judge, in order."""
    facts_stage(game_date)
    llm_predict_stage(game_date)
    return judge_stage(game_date)


if __name__ == "__main__":
    import sys
    run_all(sys.argv[1] if len(sys.argv) > 1 else None)
