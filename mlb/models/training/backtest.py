"""Backtest the PA-level simulator against historical starts.

Answers the only question that matters before real money: is the model
CALIBRATED (when it says 60%, does it hit ~60%?) and, where historical odds
exist, is it PROFITABLE after vig?

A backtest dataset is a list of game rows:
  {
    "game_date": "2024-06-01", "pitcher_name": "...",
    "pitcher": {"k":.., "bb":.., "hit":..},          # per-PA, as-of the start
    "lineup":  [{"k":..,"bb":..,"hit":..} * 9],       # opponent, vs the hand
    "recent_pitch_counts": [..],
    "k_park": 1.0, "k_ump": 1.0, "k_arsenal": 1.0,
    "defense_context": {...}, "third_time_penalty": 0.05,
    "actual_ks": 7, "actual_outs": 18,
    "lines": [
       {"market":"pitcher_strikeouts","line":5.5,
        "over_odds":-115,"under_odds":-105,          # optional (enables ROI)
        "closing_over_odds":-120,"closing_under_odds":100}  # optional (CLV)
    ]
  }

Build a real dataset with `mlb.data_fetching.build_backtest_data` (StatsAPI, run on your
machine). Validate the engine itself offline with synthetic_dataset(), where
the model IS the data-generating process, so calibration should be near
perfect — that proves the scoring math, not the model's real edge.

    python -m mlb.models.training.backtest path/to/dataset.json
    python -m mlb.models.training.backtest --synthetic
"""
import argparse
import csv
import json
import math

import numpy as np

from mlb.models.prediction import factors, simulator
from shared.betting.math import (
    american_to_decimal,
    consensus_p_over,
    kelly_fraction,
)

MIN_EDGE = 0.03

# accept a few common spellings of the two markets in an odds archive
MARKET_ALIASES = {
    "pitcher_strikeouts": "pitcher_strikeouts", "strikeouts": "pitcher_strikeouts",
    "ks": "pitcher_strikeouts", "k": "pitcher_strikeouts", "so": "pitcher_strikeouts",
    "pitcher_outs": "pitcher_outs", "outs": "pitcher_outs", "ip_outs": "pitcher_outs",
}


# ── engine ───────────────────────────────────────────────────────────────────

def run_backtest(dataset, n_sims=4000, seed=0):
    """Simulate every game/line and collect scored records."""
    records = []
    for i, row in enumerate(dataset):
        ks, outs = simulator.simulate_starter(
            row["pitcher"], row["lineup"],
            recent_pitch_counts=row.get("recent_pitch_counts"),
            k_park=row.get("k_park", 1.0), k_ump=row.get("k_ump", 1.0),
            k_form=row.get("k_form", 1.0),
            k_arsenal=row.get("k_arsenal", 1.0),
            defense_context=row.get("defense_context"),
            third_time_penalty=row.get(
                "third_time_penalty", simulator.THIRD_TIME_PENALTY
            ),
            n_sims=n_sims, seed=seed + i,
        )
        samples = {"pitcher_strikeouts": ks, "pitcher_outs": outs}
        actual = {"pitcher_strikeouts": row["actual_ks"],
                  "pitcher_outs": row["actual_outs"]}
        for ln in row.get("lines", []):
            arr = samples.get(ln["market"])
            if arr is None or actual[ln["market"]] is None:
                continue
            records.append(_score_line(arr, actual[ln["market"]], ln))
    return records


def _score_line(arr, actual_value, ln):
    line = ln["line"]
    p_over, p_under, _ = simulator.prob_over(arr, line)

    # realized outcome (pushes -> None, excluded from calibration/settlement)
    if actual_value > line:
        over_hit = True
    elif actual_value < line:
        over_hit = False
    else:
        over_hit = None

    rec = {"market": ln["market"], "line": line, "p_over": p_over,
           "over_hit": over_hit, "market_p_over": None, "edge": None,
           "bet_side": None, "odds": None, "profit": None, "clv": None,
           "kelly": 0.0}

    market_p = consensus_p_over([(ln.get("over_odds"), ln.get("under_odds"))])
    if market_p is not None:
        rec["market_p_over"] = market_p
        edge_over = p_over - market_p
        edge_under = p_under - (1 - market_p)
        if edge_over >= MIN_EDGE and edge_over >= edge_under:
            rec.update(bet_side="over", edge=edge_over,
                       odds=ln.get("over_odds"),
                       kelly=kelly_fraction(p_over, ln.get("over_odds")))
        elif edge_under >= MIN_EDGE:
            rec.update(bet_side="under", edge=edge_under,
                       odds=ln.get("under_odds"),
                       kelly=kelly_fraction(p_under, ln.get("under_odds")))

    if rec["bet_side"] and over_hit is not None and rec["odds"] is not None:
        won = (over_hit and rec["bet_side"] == "over") or \
              (not over_hit and rec["bet_side"] == "under")
        dec = american_to_decimal(rec["odds"])
        rec["profit"] = (dec - 1.0) if won else -1.0  # flat 1u
        rec["won"] = won

    # CLV vs the closing de-vigged consensus, if a closing price was recorded
    close_p = consensus_p_over(
        [(ln.get("closing_over_odds"), ln.get("closing_under_odds"))])
    if close_p is not None and rec["bet_side"]:
        ours = p_over if rec["bet_side"] == "over" else p_under
        close_side = close_p if rec["bet_side"] == "over" else 1 - close_p
        rec["clv"] = ours - close_side
    return rec


# ── odds archive merge ───────────────────────────────────────────────────────

def _to_num(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return None


def load_odds_archive(path):
    """Load historical prop odds keyed by (game_date, normalized pitcher).

    Accepts JSON (list of records) or CSV with columns:
      game_date, player_name, market, line, over_odds, under_odds,
      [closing_over_odds, closing_under_odds]
    'market' may be pitcher_strikeouts/pitcher_outs or a common alias.
    Closing prices are optional (needed only for CLV); when absent, the
    opening price is reused so CLV is ~0 rather than missing.
    """
    if path.lower().endswith(".json"):
        with open(path) as f:
            records = json.load(f)
    else:
        with open(path, newline="") as f:
            records = list(csv.DictReader(f))

    archive = {}
    for r in records:
        market = MARKET_ALIASES.get(str(r.get("market", "")).strip().lower())
        line = _to_num(r.get("line"))
        if market is None or line is None:
            continue
        over = _to_num(r.get("over_odds"))
        under = _to_num(r.get("under_odds"))
        key = (str(r.get("game_date", "")).strip(),
               factors._norm_name(r.get("player_name")))
        archive.setdefault(key, []).append({
            "market": market, "line": float(line),
            "over_odds": over, "under_odds": under,
            "closing_over_odds": _to_num(r.get("closing_over_odds")) if r.get("closing_over_odds") not in (None, "") else over,
            "closing_under_odds": _to_num(r.get("closing_under_odds")) if r.get("closing_under_odds") not in (None, "") else under,
        })
    return archive


def merge_odds(dataset, archive, replace=True):
    """Attach real odds to a dataset. Returns (dataset, n_matched_rows).

    replace=True swaps each start's lines for the archive's actual posted
    lines+odds (the honest ROI/CLV evaluation — the model is judged on the
    lines the market really offered). replace=False keeps existing lines and
    only attaches odds where (market, line) match.
    """
    matched = 0
    for row in dataset:
        key = (row.get("game_date", ""), factors._norm_name(row.get("pitcher_name")))
        arch_lines = archive.get(key)
        if not arch_lines:
            if replace:
                row["lines"] = []
            continue
        matched += 1
        if replace:
            row["lines"] = [dict(ln) for ln in arch_lines]
        else:
            by_key = {(a["market"], a["line"]): a for a in arch_lines}
            for ln in row.get("lines", []):
                a = by_key.get((ln["market"], ln["line"]))
                if a:
                    ln.update({k: a[k] for k in
                               ("over_odds", "under_odds",
                                "closing_over_odds", "closing_under_odds")})
    return dataset, matched


# ── metrics ──────────────────────────────────────────────────────────────────

def calibration(records, n_bins=10):
    """Reliability table over resolved props. Returns (bins, ece)."""
    resolved = [(r["p_over"], 1.0 if r["over_hit"] else 0.0)
                for r in records if r["over_hit"] is not None]
    n = len(resolved)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        pts = [(p, y) for p, y in resolved
               if (p >= lo and p < hi) or (b == n_bins - 1 and p == 1.0)]
        if not pts:
            bins.append({"lo": lo, "hi": hi, "n": 0, "pred": None, "emp": None})
            continue
        pred = sum(p for p, _ in pts) / len(pts)
        emp = sum(y for _, y in pts) / len(pts)
        bins.append({"lo": lo, "hi": hi, "n": len(pts), "pred": pred, "emp": emp})
        ece += (len(pts) / n) * abs(pred - emp)
    return bins, ece


def log_loss(records):
    resolved = [(r["p_over"], 1.0 if r["over_hit"] else 0.0)
                for r in records if r["over_hit"] is not None]
    if not resolved:
        return None
    tot = 0.0
    for p, y in resolved:
        p = min(max(p, 1e-6), 1 - 1e-6)
        tot += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return tot / len(resolved)


def brier(records):
    resolved = [(r["p_over"], 1.0 if r["over_hit"] else 0.0)
                for r in records if r["over_hit"] is not None]
    if not resolved:
        return None
    return sum((p - y) ** 2 for p, y in resolved) / len(resolved)


def betting_summary(records):
    bets = [r for r in records if r["bet_side"] and r.get("profit") is not None]
    if not bets:
        return {"n_bets": 0}
    staked_flat = len(bets)
    profit_flat = sum(r["profit"] for r in bets)
    wins = sum(1 for r in bets if r.get("won"))
    kelly_staked = sum(r["kelly"] for r in bets if r["kelly"] > 0)
    kelly_profit = sum(r["profit"] * r["kelly"] for r in bets if r["kelly"] > 0)
    clvs = [r["clv"] for r in bets if r["clv"] is not None]
    return {
        "n_bets": len(bets),
        "hit_rate": wins / len(bets),
        "roi_flat": profit_flat / staked_flat,
        "roi_kelly": (kelly_profit / kelly_staked) if kelly_staked else None,
        "avg_clv": (sum(clvs) / len(clvs)) if clvs else None,
        "clv_positive_rate": (sum(1 for c in clvs if c > 0) / len(clvs))
                             if clvs else None,
    }


def report(records):
    bins, ece = calibration(records)
    ll = log_loss(records)
    br = brier(records)
    bet = betting_summary(records)
    n_resolved = sum(1 for r in records if r["over_hit"] is not None)

    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST REPORT")
    lines.append("=" * 60)
    lines.append(f"props scored: {len(records)}  (resolved: {n_resolved})")
    lines.append(f"log-loss: {ll:.4f}   brier: {br:.4f}   "
                 f"ECE: {ece:.4f}   (lower is better; 0.693/0.25 = coinflip)")
    lines.append("")
    lines.append("by market:")
    for market in sorted({record["market"] for record in records}):
        market_records = [
            record for record in records if record["market"] == market
        ]
        _, market_ece = calibration(market_records)
        market_ll = log_loss(market_records)
        market_br = brier(market_records)
        lines.append(
            f"  {market:<22} n={len(market_records):>5}  "
            f"log-loss={market_ll:.4f}  brier={market_br:.4f}  "
            f"ECE={market_ece:.4f}"
        )
    lines.append("")
    lines.append("calibration (predicted P(over) vs realized over-rate):")
    lines.append(f"  {'bin':>12}  {'n':>5}  {'pred':>6}  {'actual':>6}  {'gap':>6}")
    for b in bins:
        if b["n"] == 0:
            continue
        gap = b["emp"] - b["pred"]
        lines.append(f"  {b['lo']:.1f}-{b['hi']:.1f} {'':>3}  {b['n']:>5}  "
                     f"{b['pred']:>6.3f}  {b['emp']:>6.3f}  {gap:>+6.3f}")
    lines.append("")
    if bet.get("n_bets"):
        lines.append(f"bets flagged (edge >= {MIN_EDGE:.0%}): {bet['n_bets']}")
        lines.append(f"  hit rate : {bet['hit_rate']:.3f}")
        lines.append(f"  ROI flat : {bet['roi_flat']:+.3f} (profit per 1u staked)")
        if bet.get("roi_kelly") is not None:
            lines.append(f"  ROI kelly: {bet['roi_kelly']:+.3f}")
        if bet.get("avg_clv") is not None:
            lines.append(f"  avg CLV  : {bet['avg_clv']:+.4f}  "
                         f"(beat close {bet['clv_positive_rate']:.0%} of bets)")
    else:
        lines.append("no bets flagged (no odds in dataset, or no edges cleared "
                     "the threshold)")
    lines.append("=" * 60)
    return "\n".join(lines)


# ── synthetic self-consistency dataset ───────────────────────────────────────

def synthetic_dataset(n_games=250, seed=1, flat_market=False):
    """Generate games where the simulator is the true process.

    Each game's 'actual' is one draw from the same simulator, so a correct
    engine must come out well-calibrated. With flat_market=True every line is
    priced at -110/-110 (de-vigged 50/50) regardless of the true probability —
    an unrealistically dumb book — so the model's real edges should turn a
    clear profit, exercising the betting/ROI path.
    """
    rng = np.random.default_rng(seed)
    dataset = []
    for i in range(n_games):
        pitcher = {"k": float(rng.uniform(0.15, 0.32)),
                   "bb": float(rng.uniform(0.06, 0.11)),
                   "hit": float(rng.uniform(0.18, 0.25))}
        lineup = [{"k": float(rng.uniform(0.15, 0.30)),
                   "bb": float(rng.uniform(0.06, 0.11)),
                   "hit": float(rng.uniform(0.18, 0.25))} for _ in range(9)]
        pitch_counts = [int(rng.normal(90, 6)) for _ in range(4)]

        # true 'actual' outcome: a single game drawn from these rates
        ks, outs = simulator.simulate_starter(
            pitcher, lineup, recent_pitch_counts=pitch_counts,
            n_sims=1, seed=10_000 + i)
        actual_ks, actual_outs = int(ks[0]), int(outs[0])

        # anchor lines on an independent coarse estimate of the center
        est_ks, est_outs = simulator.simulate_starter(
            pitcher, lineup, recent_pitch_counts=pitch_counts,
            n_sims=400, seed=20_000 + i)
        lines = []
        for market, est in (("pitcher_strikeouts", est_ks.mean()),
                            ("pitcher_outs", est_outs.mean())):
            center = round(est)
            for off in (-1.5, -0.5, 0.5, 1.5):
                ln = {"market": market, "line": center + off}
                if flat_market:
                    ln["over_odds"] = -110
                    ln["under_odds"] = -110
                lines.append(ln)

        dataset.append({
            "game_date": "synthetic", "pitcher_name": f"sim_{i}",
            "pitcher": pitcher, "lineup": lineup,
            "recent_pitch_counts": pitch_counts,
            "k_park": 1.0, "k_ump": 1.0,
            "actual_ks": actual_ks, "actual_outs": actual_outs,
            "lines": lines,
        })
    return dataset


def main():
    ap = argparse.ArgumentParser(description="Backtest the MLB props simulator")
    ap.add_argument("dataset", nargs="?", help="path to a backtest dataset JSON")
    ap.add_argument("--synthetic", action="store_true",
                    help="run the self-consistency validation instead")
    ap.add_argument("--flat-market", action="store_true",
                    help="synthetic only: price every line at -110 to test ROI")
    ap.add_argument("--odds", help="odds archive (CSV/JSON) to merge for ROI/CLV")
    ap.add_argument("--keep-lines", action="store_true",
                    help="with --odds: attach odds to existing lines instead of "
                         "replacing them with the archive's posted lines")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--games", type=int, default=250)
    args = ap.parse_args()

    if args.synthetic or not args.dataset:
        print("Running synthetic self-consistency backtest "
              "(model == data-generating process)...")
        data = synthetic_dataset(n_games=args.games, flat_market=args.flat_market)
    else:
        with open(args.dataset) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} games from {args.dataset}")

    if args.odds:
        archive = load_odds_archive(args.odds)
        data, matched = merge_odds(data, archive, replace=not args.keep_lines)
        print(f"Merged odds archive: {matched}/{len(data)} starts matched a line")

    records = run_backtest(data, n_sims=args.sims)
    print(report(records))


if __name__ == "__main__":
    main()
