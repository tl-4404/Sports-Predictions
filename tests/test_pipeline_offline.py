"""Offline end-to-end test of the MLB props pipeline.

No network, no pytest needed:  python tests/test_pipeline_offline.py

Covers: schema, stub odds ingestion, de-vig/Kelly math, the PA simulator,
prediction logging with edges/bets, and next-day grading with CLV.
StatsAPI calls are monkeypatched with fixtures.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlb"))

import numpy as np

import db
import data_today
import grade
import odds_props
import predict
import simulator
from betting_math import (american_to_prob, consensus_p_over, devig_two_way,
                          kelly_fraction)

FAILURES = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


# ── betting math ─────────────────────────────────────────────────────────────
print("betting_math:")
check("american -110 implied", abs(american_to_prob(-110) - 0.5238) < 1e-3)
p_o, p_u = devig_two_way(-110, -110)
check("devig symmetric market", abs(p_o - 0.5) < 1e-9 and abs(p_u - 0.5) < 1e-9)
p_o, p_u = devig_two_way(-130, 110)
check("devig sums to 1", abs(p_o + p_u - 1.0) < 1e-9)
check("devig favors juiced side", p_o > 0.5)
check("consensus averages books",
      abs(consensus_p_over([(-110, -110), (-130, 110)]) - (0.5 + p_o) / 2) < 1e-9)
check("kelly zero without edge", kelly_fraction(0.50, -110) == 0.0)
k = kelly_fraction(0.58, -110)
check("kelly positive with edge", 0 < k <= 0.02, f"(k={k:.4f})")

# ── simulator ────────────────────────────────────────────────────────────────
print("simulator:")
avg = dict(simulator.LEAGUE)
lineup = [dict(simulator.LEAGUE) for _ in range(9)]
ks, outs = simulator.simulate_starter(avg, lineup, [92, 88, 95], n_sims=4000, seed=7)
check("league-avg K mean sane", 4.2 < ks.mean() < 5.8, f"({ks.mean():.2f})")
check("league-avg outs mean sane", 14.5 < outs.mean() < 17.5, f"({outs.mean():.2f})")
check("outs never exceed 27", outs.max() <= 27)

high_k = {"k": 0.32, "bb": 0.07, "hit": 0.19}
ks_hi, _ = simulator.simulate_starter(high_k, lineup, [95, 100, 98], n_sims=4000, seed=7)
check("high-K pitcher strikes out more", ks_hi.mean() > ks.mean() + 1.5,
      f"({ks_hi.mean():.2f} vs {ks.mean():.2f})")

p_over, p_under, _ = simulator.prob_over(ks, 4.5)
check("prob_over sums to 1", abs(p_over + p_under - 1.0) < 1e-9)
check("monotonic in line",
      simulator.prob_over(ks, 3.5)[0] > simulator.prob_over(ks, 7.5)[0])

# ── pipeline end-to-end on a temp DB with mocked StatsAPI ────────────────────
print("pipeline (temp db, mocked network):")
tmp = tempfile.mkdtemp()
db.DB_PATH = Path(tmp) / "test.db"

conn = db.connect()
db.ensure_schema(conn)

# 1. stub odds ingestion (uses the shipped sample fixture)
rows = odds_props.StubPropsFetcher().fetch("2026-07-01")
n = db.insert_lines(conn, rows)
check("stub lines stored", n == 5, f"(n={n})")

# 2. mock the StatsAPI layer
def fake_schedule(game_date=None):
    return [{
        "game_pk": 999001, "game_date": "2026-07-01", "status": "Preview",
        "home_team": "New York Mets", "away_team": "Atlanta Braves",
        "venue": "Citi Field",
        "home_pitcher": {"id": 111, "name": "Ace Example", "hand": "R"},
        "away_pitcher": {"id": 222, "name": "Soft Tosser", "hand": "L"},
        "home_lineup": [], "away_lineup": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }]

def fake_profile(pid, season=None):
    if pid == 111:  # strikeout artist
        return {"k": 0.30, "bb": 0.06, "hit": 0.19, "batters_faced": 400,
                "recent_pitch_counts": [98, 95, 101, 97, 99]}
    return {"k": 0.16, "bb": 0.09, "hit": 0.24, "batters_faced": 350,
            "recent_pitch_counts": [82, 78, 85, 80, 79]}

def fake_batter(bid, hand, season=None):
    return {"k": 0.22 + (bid % 3) * 0.02, "bb": 0.08, "hit": 0.21}

data_today.get_schedule = fake_schedule
data_today.get_pitcher_profile = fake_profile
data_today.get_batter_rates = fake_batter

# 3. predict
n_pred = predict.predict_date("2026-07-01", n_sims=4000)
check("predictions logged", n_pred >= 4, f"(n={n_pred})")

preds = conn.execute(
    "SELECT * FROM prop_predictions WHERE game_date='2026-07-01'").fetchall()
ace_k = [p for p in preds if p["player_name"] == "Ace Example"
         and p["market"] == "pitcher_strikeouts"]
check("ace matched to posted line", len(ace_k) == 1 and ace_k[0]["line"] == 6.5)
check("ace edge computed vs market", ace_k[0]["market_p_over"] is not None)
check("ace favors over 6.5", ace_k[0]["p_over"] > 0.5, f"(p={ace_k[0]['p_over']:.3f})")
bets = [p for p in preds if p["bet_side"]]
check("at least one bet flagged", len(bets) >= 1,
      f"({[(b['player_name'], b['market'], b['bet_side']) for b in bets]})")
check("kelly stake capped", all((b["kelly_frac"] or 0) <= 0.02 for b in bets))

# 4. grade with mocked box score
def fake_actuals(game_pk):
    return {
        111: {"name": "Ace Example", "ks": 8, "outs": 19},
        222: {"name": "Soft Tosser", "ks": 3, "outs": 14},
    }

data_today.get_starter_actuals = fake_actuals
n_graded = grade.grade_date("2026-07-01")
check("all predictions graded", n_graded == len(preds), f"({n_graded})")

res = conn.execute(
    """SELECT p.player_name, p.market, p.line, r.actual, r.outcome, r.clv
       FROM prop_results r JOIN prop_predictions p ON p.id = r.prediction_id"""
).fetchall()
ace_res = [r for r in res if r["player_name"] == "Ace Example"
           and r["market"] == "pitcher_strikeouts"][0]
check("ace 8 Ks graded over 6.5", ace_res["actual"] == 8 and ace_res["outcome"] == "over")
soft_out = [r for r in res if r["player_name"] == "Soft Tosser"
            and r["market"] == "pitcher_outs"][0]
check("soft tosser 14 outs under 15.5", soft_out["outcome"] == "under")
clvs = [r["clv"] for r in res if r["clv"] is not None]
check("clv computed for bets", len(clvs) == len(bets), f"({len(clvs)})")

check("innings_to_outs 5.2 = 17", data_today.innings_to_outs("5.2") == 17)
check("innings_to_outs 7.0 = 21", data_today.innings_to_outs("7.0") == 21)

# ── LLM lanes with a fake Anthropic client ───────────────────────────────────
print("llm lanes (fake client):")
import json as _json
import llm


class FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeBlock(text)]


class FakeMessages:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kwargs):
        self.outer.calls.append(kwargs)
        if any(t.get("name") == "web_search" for t in kwargs.get("tools", [])):
            # facts call: note wrapped in prose to exercise lenient parsing
            return FakeResponse(
                'Here is what I found:\n```json\n'
                + _json.dumps({
                    "pitchers": [
                        {"name": "Ace Example", "flags": [], "outlook": "neutral"},
                        {"name": "Soft Tosser",
                         "flags": ["manager hinted at a ~75 pitch limit"],
                         "outlook": "negative"},
                    ],
                    "game_notes": ["roof closed"],
                    "sources": ["beat writer"],
                }) + "\n```")
        schema = kwargs.get("output_config", {}).get("format", {}).get("schema", {})
        props = schema.get("properties", {})
        if "props" in props:
            # predictor call: parse requested lines out of the prompt
            prompt = kwargs["messages"][0]["content"]
            out = []
            for m in __import__("re").finditer(r"- (\w+): line ([\d.]+)", prompt):
                out.append({"market": m.group(1), "line": float(m.group(2)),
                            "p_over": 0.62, "confidence": 0.5,
                            "rationale": "test rationale"})
            return FakeResponse(_json.dumps({"props": out}))
        # judge call: extreme final_p_over (tests clamping) + one veto
        prompt = kwargs["messages"][0]["content"]
        decisions = []
        for m in __import__("re").finditer(
                r"- (\w+) line ([\d.]+): sim ([\d.]+)", prompt):
            market, line = m.group(1), float(m.group(2))
            decisions.append({
                "market": market, "line": line,
                "final_p_over": 0.99,           # absurd — must be clamped
                "action": "pass" if market == "pitcher_outs" else "bet",
                "rationale": "test judge",
            })
        return FakeResponse(_json.dumps({"decisions": decisions}))


class FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = FakeMessages(self)


fake = FakeClient()
llm.set_client(fake)

n_facts = llm.facts_stage("2026-07-01")
check("facts researched per game", n_facts == 1)
conn = db.connect()
facts_row = conn.execute("SELECT facts_json FROM llm_facts").fetchone()
check("facts stored + lenient-parsed",
      facts_row and "pitch limit" in facts_row["facts_json"])

n_llm = llm.llm_predict_stage("2026-07-01")
check("llm predictions logged", n_llm == 4, f"(n={n_llm})")
llm_rows = conn.execute(
    "SELECT * FROM prop_predictions WHERE source='llm'").fetchall()
check("llm rows carry edge vs market",
      all(r["market_p_over"] is not None for r in llm_rows))
check("llm run_id matches sim run",
      {r["run_id"] for r in llm_rows} ==
      {r["run_id"] for r in conn.execute(
          "SELECT run_id FROM prop_predictions WHERE source='sim'").fetchall()})

n_judge = llm.judge_stage("2026-07-01")
check("judge decisions logged", n_judge == 4, f"(n={n_judge})")
judge_rows = conn.execute(
    "SELECT * FROM prop_predictions WHERE source='judge'").fetchall()
for jr in judge_rows:
    sim_r = conn.execute(
        """SELECT p_over FROM prop_predictions
           WHERE source='sim' AND player_name=? AND market=? AND line=?""",
        (jr["player_name"], jr["market"], jr["line"])).fetchone()
    llm_r = conn.execute(
        """SELECT p_over FROM prop_predictions
           WHERE source='llm' AND player_name=? AND market=? AND line=?""",
        (jr["player_name"], jr["market"], jr["line"])).fetchone()
    blend = llm.W_SIM * sim_r["p_over"] + (1 - llm.W_SIM) * llm_r["p_over"]
    check(f"judge clamped ({jr['player_name'][:4]} {jr['market'][8:]} {jr['line']})",
          abs(jr["p_over"] - blend) <= llm.JUDGE_BAND + 1e-9,
          f"(p={jr['p_over']:.3f}, blend={blend:.3f})")
vetoed = [r for r in judge_rows if r["market"] == "pitcher_outs"]
check("veto suppresses bet", all(r["bet_side"] is None for r in vetoed))
meta = _json.loads(vetoed[0]["rationale"])
check("judge rationale stored", meta["action"] == "pass" and "blend" in meta)

# grade the judge/llm rows too (sim rows already graded above)
n_graded2 = grade.grade_date("2026-07-01")
check("llm+judge rows graded", n_graded2 == n_llm + n_judge, f"({n_graded2})")
per_source = conn.execute(
    """SELECT p.source, COUNT(*) n FROM prop_predictions p
       JOIN prop_results r ON r.prediction_id = p.id GROUP BY p.source"""
).fetchall()
check("all three lanes graded", {r["source"] for r in per_source} ==
      {"sim", "llm", "judge"})

conn.close()

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
