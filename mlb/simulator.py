"""State-mixture plate-appearance simulator for MLB starter props.

The merged v4.2/v5 engine separates planned workload from bad-outing hooks,
models short/normal/extended states, applies true times-through-order decay,
and exposes batters-faced stress lanes for strikeout unders.
"""
from __future__ import annotations

import numpy as np

LEAGUE = {"k": 0.223, "bb": 0.085, "hit": 0.212}
PITCHES_PER_PA = {"K": 4.8, "WALK": 5.5, "HIT": 3.5, "OUT": 3.6}
PITCH_SD = 1.4
DP_PROB = 0.11
MAX_OUTS = 27
TTO_K_FACTORS = (1.00, 0.97, 0.92, 0.88)


def odds_ratio(p_pitcher, p_batter, p_league):
    p_pitcher = min(max(float(p_pitcher), 1e-4), 1 - 1e-4)
    p_batter = min(max(float(p_batter), 1e-4), 1 - 1e-4)
    p_league = min(max(float(p_league), 1e-4), 1 - 1e-4)
    num = (p_pitcher * p_batter) / p_league
    den = num + ((1 - p_pitcher) * (1 - p_batter)) / (1 - p_league)
    return num / den


def build_matchup_probs(pitcher, lineup, k_park=1.0, k_ump=1.0, k_form=1.0):
    probs = np.zeros((len(lineup), 4))
    for i, batter in enumerate(lineup):
        p_k = odds_ratio(pitcher["k"], batter["k"], LEAGUE["k"])
        odds_k = p_k / (1 - p_k) * k_park * k_ump * k_form
        p_k = odds_k / (1 + odds_k)
        p_bb = odds_ratio(pitcher["bb"], batter["bb"], LEAGUE["bb"])
        p_hit = odds_ratio(pitcher["hit"], batter["hit"], LEAGUE["hit"])
        rem = 1.0 - p_k
        other = p_bb + p_hit
        if other >= rem:
            scale = (rem * 0.9) / max(other, 1e-9)
            p_bb, p_hit = p_bb * scale, p_hit * scale
        probs[i] = (p_k, p_bb, p_hit, 1.0 - p_k - p_bb - p_hit)
    return probs


def workload_states(
    recent_pitch_counts=None,
    recent_starts=None,
    verified_planned_limit=None,
    manager_hook_factor=1.0,
    bullpen_pressure=0.0,
):
    """Return explicit short/normal/extended states and auditable weights."""
    starts = list(recent_starts or [])[-8:]
    counts = [
        float(row["pitches"]) for row in starts if row.get("pitches") is not None
    ]
    if not counts:
        counts = [float(value) for value in (recent_pitch_counts or [])][-8:]

    if verified_planned_limit is not None:
        normal = float(verified_planned_limit)
        evidence = "verified_planned_limit"
    elif counts:
        arr = np.asarray(counts, dtype=float)
        normal = float(np.quantile(arr, 0.75) + 3.0)
        evidence = "recent_upper_usage"
    else:
        normal = 88.0
        evidence = "league_prior"
    normal = float(np.clip(normal, 25.0, 115.0))

    bad_short = []
    aligned = starts[-len(counts):] if starts else []
    for index, count in enumerate(counts):
        if count >= normal - 10:
            continue
        if aligned and index < len(aligned):
            row = aligned[index]
            outs = max(int(row.get("outs") or 0), 1)
            traffic = (int(row.get("hits") or 0) + int(row.get("walks") or 0)) / outs
            if int(row.get("earned_runs") or 0) >= 4 or traffic >= 0.55:
                bad_short.append(count)
        else:
            bad_short.append(count)

    short_weight = (len(bad_short) + 1.0) / (len(counts) + 5.0)
    short_weight *= max(0.6, min(1.6, float(manager_hook_factor)))
    short_weight *= 1.0 + 0.25 * max(-1.0, min(1.0, float(bullpen_pressure)))
    short_weight = float(np.clip(short_weight, 0.08, 0.45))
    extended_weight = float(np.clip(0.18 - 0.08 * bullpen_pressure, 0.08, 0.28))
    normal_weight = max(0.20, 1.0 - short_weight - extended_weight)
    total = short_weight + normal_weight + extended_weight
    short_weight, normal_weight, extended_weight = (
        short_weight / total,
        normal_weight / total,
        extended_weight / total,
    )

    short_mean = float(np.mean(bad_short)) if bad_short else normal - 17.0
    states = [
        {"name": "short", "weight": short_weight,
         "pitch_mean": float(np.clip(short_mean, 25.0, normal - 3.0)), "pitch_sd": 7.0},
        {"name": "normal", "weight": normal_weight,
         "pitch_mean": normal, "pitch_sd": 6.0},
        {"name": "extended", "weight": extended_weight,
         "pitch_mean": float(np.clip(normal + 10.0, normal, 118.0)), "pitch_sd": 5.0},
    ]
    return {
        "states": states,
        "planned_mean": normal,
        "early_hook_rate": short_weight,
        "n_starts": len(counts),
        "evidence": evidence,
        "verified": verified_planned_limit is not None or len(counts) >= 3,
    }


def workload_parameters(recent_pitch_counts=None, recent_starts=None):
    mix = workload_states(recent_pitch_counts, recent_starts)
    short = next(state for state in mix["states"] if state["name"] == "short")
    return {
        "planned_mean": mix["planned_mean"],
        "planned_sd": 6.0,
        "early_hook_rate": mix["early_hook_rate"],
        "hook_mean": short["pitch_mean"],
        "n_starts": mix["n_starts"],
    }


def _sample_state(mix, rng):
    weights = [state["weight"] for state in mix["states"]]
    state = mix["states"][int(rng.choice(len(mix["states"]), p=weights))]
    limit = float(np.clip(rng.normal(state["pitch_mean"], state["pitch_sd"]), 20.0, 118.0))
    return state["name"], limit


def sample_pitch_limit(recent_pitch_counts, rng, recent_starts=None):
    _, limit = _sample_state(workload_states(recent_pitch_counts, recent_starts), rng)
    return limit


def _tto_cdfs(probs):
    cdfs = []
    for factor in TTO_K_FACTORS:
        adjusted = np.array(probs, copy=True)
        for index, row in enumerate(adjusted):
            p_k = row[0]
            odds = p_k / max(1.0 - p_k, 1e-9)
            new_k = (odds * factor) / (1.0 + odds * factor)
            other = row[1:].sum()
            row[0] = new_k
            if other > 0:
                row[1:] *= (1.0 - new_k) / other
            adjusted[index] = row
        cdfs.append(np.cumsum(adjusted, axis=1))
    return cdfs


def simulate_starter_detailed(
    pitcher,
    lineup,
    recent_pitch_counts=None,
    recent_starts=None,
    verified_planned_limit=None,
    manager_hook_factor=1.0,
    bullpen_pressure=0.0,
    score_environment=1.0,
    pitches_per_pa_factor=1.0,
    k_park=1.0,
    k_ump=1.0,
    k_form=1.0,
    n_sims=10000,
    seed=None,
):
    rng = np.random.default_rng(seed)
    probs = build_matchup_probs(pitcher, lineup, k_park, k_ump, k_form)
    cdfs = _tto_cdfs(probs)
    n_batters = len(probs)
    mix = workload_states(
        recent_pitch_counts, recent_starts, verified_planned_limit,
        manager_hook_factor, bullpen_pressure,
    )

    ks_out = np.zeros(n_sims, dtype=np.int64)
    outs_out = np.zeros(n_sims, dtype=np.int64)
    bf_out = np.zeros(n_sims, dtype=np.int64)
    pitches_out = np.zeros(n_sims, dtype=float)
    state_out = np.empty(n_sims, dtype="U8")

    ppa_factor = float(np.clip(pitches_per_pa_factor, 0.85, 1.18))
    run_env = float(np.clip(score_environment, 0.75, 1.35))

    for s in range(n_sims):
        state_name, pitch_limit = _sample_state(mix, rng)
        ks = outs = bf = 0
        pitches = 0.0
        batter_idx = 0
        baserunners = runs = 0
        runner_on_first = False
        outs_in_inning = 0

        while outs < MAX_OUTS and pitches < pitch_limit:
            trip = min(batter_idx // n_batters, len(cdfs) - 1)
            row = cdfs[trip][batter_idx % n_batters]
            u = rng.random()
            if u < row[0]:
                outcome = "K"
            elif u < row[1]:
                outcome = "WALK"
            elif u < row[2]:
                outcome = "HIT"
            else:
                outcome = "OUT"

            pitches += max(
                rng.normal(PITCHES_PER_PA[outcome] * ppa_factor, PITCH_SD), 1.0
            )
            bf += 1
            if outcome == "K":
                ks += 1
                outs += 1
                outs_in_inning += 1
            elif outcome == "OUT":
                if runner_on_first and outs_in_inning < 2 and rng.random() < DP_PROB:
                    outs += 2
                    outs_in_inning += 2
                    runner_on_first = False
                else:
                    outs += 1
                    outs_in_inning += 1
            else:
                baserunners += 1
                # Coarse score-state hook proxy. It is conditional on traffic and
                # therefore does not turn every short recent start into a pitch cap.
                if runner_on_first and rng.random() < 0.24 * run_env:
                    runs += 1
                runner_on_first = True

            if outs_in_inning >= 3:
                outs_in_inning = 0
                runner_on_first = False
                traffic = baserunners / max(bf, 1)
                bad_outing = runs >= 4 or (runs >= 3 and traffic >= 0.32)
                if state_name == "short" and outs >= 9 and bad_outing:
                    break
                if bad_outing and outs >= 12 and pitches >= pitch_limit - 16:
                    break
                if pitches >= pitch_limit - 8:
                    break
            batter_idx += 1

        ks_out[s] = ks
        outs_out[s] = min(outs, MAX_OUTS)
        bf_out[s] = bf
        pitches_out[s] = pitches
        state_out[s] = state_name

    return {
        "ks": ks_out,
        "outs": outs_out,
        "batters_faced": bf_out,
        "pitches": pitches_out,
        "state": state_out,
        "workload": mix,
    }


def simulate_starter(
    pitcher, lineup, recent_pitch_counts=None, recent_starts=None,
    k_park=1.0, k_ump=1.0, k_form=1.0, n_sims=10000, seed=None, **kwargs
):
    result = simulate_starter_detailed(
        pitcher, lineup, recent_pitch_counts, recent_starts,
        k_park=k_park, k_ump=k_ump, k_form=k_form,
        n_sims=n_sims, seed=seed, **kwargs,
    )
    return result["ks"], result["outs"]


def prob_over(samples, line):
    samples = np.asarray(samples)
    over = float(np.mean(samples > line))
    under = float(np.mean(samples < line))
    push = max(0.0, 1.0 - over - under)
    if over + under > 0:
        return over / (over + under), under / (over + under), push
    return 0.5, 0.5, push


def k_under_stress(ks, batters_faced, line, extras=(2, 4), seed=1988):
    """Mechanical base/+2/+4 BF under probabilities.

    Extra strikeouts are drawn from the simulation's aggregate K-per-BF rate.
    The result distinguishes a skill under (survives +4 BF) from a workload under.
    """
    ks = np.asarray(ks, dtype=int)
    bf = np.asarray(batters_faced, dtype=int)
    rng = np.random.default_rng(seed)
    k_per_bf = float(np.clip(ks.sum() / max(bf.sum(), 1), 0.03, 0.45))
    _, base_under, _ = prob_over(ks, line)
    result = {"base": base_under, "k_per_bf": k_per_bf}
    for extra in extras:
        stressed = ks + rng.binomial(int(extra), k_per_bf, size=len(ks))
        _, p_under, _ = prob_over(stressed, line)
        result[f"plus_{extra}"] = p_under
    result["basis"] = (
        "skill_under"
        if result.get("plus_4", base_under) >= max(0.55, base_under - 0.06)
        else "workload_under"
    )
    return result


def summarize(samples):
    samples = np.asarray(samples)
    return {
        "mean": float(samples.mean()),
        "p10": float(np.percentile(samples, 10)),
        "p50": float(np.percentile(samples, 50)),
        "p90": float(np.percentile(samples, 90)),
    }
