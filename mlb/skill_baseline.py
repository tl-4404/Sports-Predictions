"""Transparent, shadow-only pitcher skill challenger.

Inspired by Marcel's recency weighting and regression, not a reproduction of
Marcel, Steamer or THE BAT X. No aging/stuff/injury/workload model is claimed.
Fit each event separately on past seasons; retain a Beta pseudo-posterior for
sampling uncertainty conditional on stationary talent and the chosen weights.
It does not include future talent drift or estimate a daily prop probability.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import requests

EVENTS = {"k": ("strikeOuts",), "bb": ("baseOnBalls", "hitByPitch"), "hit": ("hits",)}
DECAYS = (0.0, 0.25, 0.5, 0.75, 1.0)
STRENGTHS = (25, 50, 100, 200, 400, 800)


def counts(payload):
    """One MLB total per player. Reject duplicate aggregates rather than sum."""
    out = {}
    for group in payload.get("stats", []):
        for split in group.get("splits", []):
            pid = split["player"]["id"]
            if pid in out:
                raise ValueError(f"duplicate player aggregate: {pid}")
            s = split["stat"]
            out[pid] = {"bf": int(s.get("battersFaced", 0)),
                        "gs": int(s.get("gamesStarted", 0)),
                        **{event: sum(int(s.get(k, 0)) for k in keys)
                           for event, keys in EVENTS.items()}}
    return out


def league_rate(seasons, year, event):
    previous = seasons[year - 1].values()
    return sum(s[event] for s in previous) / sum(s["bf"] for s in previous)


def project(seasons, year, pid, event, decay, strength):
    prior = league_rate(seasons, year, event)
    successes, exposure = prior * strength, float(strength)
    for lag in range(1, 4):
        sample = seasons.get(year - lag, {}).get(pid, {})
        weight = decay ** (lag - 1)
        successes += weight * sample.get(event, 0)
        exposure += weight * sample.get("bf", 0)
    mean = successes / exposure
    return {"mean": mean, "alpha": successes, "beta": exposure - successes,
            "sd": (mean * (1 - mean) / (exposure + 1)) ** 0.5}


def evaluate(seasons, years, event, config):
    records = []
    for year in years:
        for pid, actual in sorted(seasons[year].items()):
            # Evaluation cohort is stated explicitly; BF weights do not enter
            # the forecast. Include rookies; absent history means league prior.
            if actual["bf"] < 100 or actual["gs"] < 1:
                continue
            p = project(seasons, year, pid, event, **config)["mean"]
            y, n = actual[event], actual["bf"]
            loss = -y * np.log(p) - (n - y) * np.log1p(-p)
            brier = y * (1 - p) ** 2 + (n - y) * p ** 2
            records.append({"year": year, "player_id": pid, "bf": n,
                            "gs": actual["gs"],
                            "prior_bf": sum(seasons.get(year - lag, {}).get(pid, {}).get("bf", 0)
                                            for lag in range(1, 4)),
                            "predicted": p, "actual": y / n,
                            "loss_sum": float(loss), "brier_sum": brier})
    return records


def metrics(records):
    n = sum(r["bf"] for r in records)
    return {"pitcher_seasons": len(records), "batters_faced": n,
            "log_loss_per_bf": sum(r["loss_sum"] for r in records) / n,
            "brier_per_bf": sum(r["brier_sum"] for r in records) / n,
            "weighted_rate_rmse": (sum(r["bf"] * (r["predicted"] - r["actual"]) ** 2
                                       for r in records) / n) ** 0.5}


def benchmark(seasons, tuning=(2023, 2024), holdout=2025, bootstraps=2000):
    if max(tuning) >= holdout:
        raise ValueError("tuning must precede held-out season")
    result = {"scope": "annual_pitcher_event_rates_not_daily_props_or_betting_roi",
              "model_status": "shadow", "tuning_years": list(tuning), "holdout_year": holdout,
              "cohort": "all pitchers with >=100 BF and >=1 GS in evaluation year, including rookies",
              "baseline": "previous season only, 50 BF prior; prior league rate from previous year",
              "uncertainty": "pitcher-block paired bootstrap; conditional on selected hyperparameters",
              "events": {}}
    for event in EVENTS:
        candidates = []
        for decay, strength in itertools.product(DECAYS, STRENGTHS):
            config = {"decay": decay, "strength": strength}
            score = metrics(evaluate(seasons, tuning, event, config))["log_loss_per_bf"]
            candidates.append((score, decay, strength))
        _, decay, strength = min(candidates)
        config = {"decay": decay, "strength": strength}
        # Never use the holdout to choose the configuration.
        old = evaluate(seasons, (holdout,), event, {"decay": 0.0, "strength": 50})
        new = evaluate(seasons, (holdout,), event, config)
        assert [(r["year"], r["player_id"]) for r in old] == [(r["year"], r["player_id"]) for r in new]
        rng = np.random.default_rng(20261026)
        differences = []
        loss_delta = np.array([a["loss_sum"] - b["loss_sum"] for a, b in zip(new, old)])
        exposures = np.array([a["bf"] for a in old])
        for _ in range(bootstraps):
            idx = rng.integers(len(old), size=len(old))
            differences.append(float(loss_delta[idx].sum() / exposures[idx].sum()))
        result["events"][event] = {
            "selected_on_tuning_only": config,
            "baseline_holdout": metrics(old), "challenger_holdout": metrics(new),
            "log_loss_delta_95pct_pitcher_bootstrap": np.quantile(differences, [0.025, 0.975]).tolist(),
            "delta_negative_is_better": True,
            "year_by_year_tuning": {str(y): metrics(evaluate(seasons, (y,), event, config)) for y in tuning},
            "holdout_stress_cohorts": {
                name: {"baseline": metrics([r for r in old if predicate(r)]),
                       "challenger": metrics([r for r in new if predicate(r)])}
                for name, predicate in {
                    "at_least_10_starts": lambda r: r["gs"] >= 10,
                    "at_least_300_BF": lambda r: r["bf"] >= 300,
                    "limited_prior_history_under_200_BF": lambda r: r["prior_bf"] < 200,
                }.items() if any(predicate(r) for r in old)
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).parent / "cache/audit_20261026")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen-counts", type=Path,
                        help="reproduce the audit from the checked-in MLB aggregate snapshot")
    args = parser.parse_args()
    seasons, hashes = {}, {}
    if args.frozen_counts:
        snapshot = json.loads(args.frozen_counts.read_text())
        seasons = {int(year): {r[0]: dict(zip(("bf", "gs", "k", "bb", "hit"), r[1:]))
                              for r in rows} for year, rows in snapshot["seasons"].items()}
        hashes = snapshot["source_sha256"]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for year in range(2020, 2026):
        if year in seasons:
            continue
        path = args.cache_dir / f"pitching_{year}.json"
        if not path.exists():
            response = requests.get("https://statsapi.mlb.com/api/v1/stats", timeout=60,
                params={"stats": "season", "group": "pitching", "season": year,
                        "sportIds": 1, "playerPool": "ALL", "limit": 10000})
            response.raise_for_status()
            path.write_text(json.dumps(response.json()))
        raw = path.read_bytes()
        seasons[year] = counts(json.loads(raw))
        hashes[str(year)] = hashlib.sha256(raw).hexdigest()
    result = benchmark(seasons)
    result["source"] = "MLB StatsAPI /api/v1/stats; season/pitching; sportIds=1; playerPool=ALL; limit=10000"
    result["source_sha256"] = hashes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
