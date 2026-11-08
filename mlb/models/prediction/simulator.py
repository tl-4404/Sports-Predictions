"""Plate-appearance-level Monte Carlo simulator for starting pitcher props.

The Bat-X-style core: instead of regressing to a point estimate, simulate the
game one plate appearance at a time and read P(over line) straight off the
resulting distribution of strikeouts and outs.

Per PA, the outcome is one of: K, WALK (incl. HBP), HIT, OUT_IN_PLAY.
Each per-PA probability blends the pitcher's rate with the batter's rate via
the standard odds-ratio (log5) method against the league baseline, then the
K probability is scaled by park and umpire factors.

The starter leaves via a stochastic hook model centered on recent workload.
Pitcher-created traffic raises the removal hazard; efficient performance lowers
it. Defensive errors consume pitches without being charged as poor pitcher
performance. Pitches per PA are sampled per outcome, while team fielding
artifacts drive double-play and error probabilities.

Known simplifications (documented on purpose):
  * traffic, rather than runs allowed, drives performance-based early hooks
  * base state remains coarse (runner on first only)
  * sacrifice outs are folded into OUT_IN_PLAY
"""
import math
import os

import numpy as np

MODEL_VERSION = "mc_pa_v2_artifacts_hook"
LEAGUE = {"k": 0.223, "bb": 0.085, "hit": 0.212}

# mean pitches per PA by outcome (Statcast-era averages)
PITCHES_PER_PA = {"K": 4.8, "WALK": 5.5, "HIT": 3.5, "OUT": 3.6}
PITCH_SD = 1.4          # per-PA noise
DP_PROB = 0.11          # P(double play | ball in play out, runner on 1st, <2 outs)
MAX_OUTS = 27
MAX_PITCHES = 125
DEFAULT_GROUND_RATE = 0.43
DEFAULT_ERROR_RATE = 0.015
GROUND_RATE_SAMPLE = 100
TEAM_DP_SAMPLE = 50
TEAM_ERROR_SAMPLE = 500
THIRD_TIME_PENALTY = float(os.getenv("MLB_TTO_PENALTY", "0.05"))


def odds_ratio(p_pitcher, p_batter, p_league):
    """log5 / odds-ratio blend of pitcher and batter event rates."""
    # math logic bounds
    p_pitcher = min(max(p_pitcher, 1e-4), 1 - 1e-4)
    p_batter = min(max(p_batter, 1e-4), 1 - 1e-4)
    p_league = min(max(p_league, 1e-4), 1 - 1e-4)
    num = (p_pitcher * p_batter) / p_league
    den = num + ((1 - p_pitcher) * (1 - p_batter)) / (1 - p_league)
    return num / den


def build_matchup_probs(pitcher, lineup, k_park=1.0, k_ump=1.0, k_form=1.0,
                        k_arsenal=1.0):
    """Per-batter outcome probabilities for the 9 lineup slots.

    pitcher: {"k":, "bb":, "hit":} per-PA rates
    lineup:  list of 9 dicts {"k":, "bb":, "hit":} (batter vs pitcher hand)
    k_park/k_ump/k_form: multiplicative K-odds scalers (park, home-plate
        umpire, and pitcher recent form/velocity). Returns array (9, 4):
        columns K, WALK, HIT, OUT.
    """
    probs = np.zeros((len(lineup), 4))
    for i, batter in enumerate(lineup):
        p_k = odds_ratio(pitcher["k"], batter["k"], LEAGUE["k"])
        # park + ump + form scale K odds, not raw prob (keeps it in [0,1])
        odds_k = p_k / (1 - p_k) * k_park * k_ump * k_form * k_arsenal
        p_k = odds_k / (1 + odds_k)
        p_bb = odds_ratio(pitcher["bb"], batter["bb"], LEAGUE["bb"])
        p_hit = odds_ratio(pitcher["hit"], batter["hit"], LEAGUE["hit"])
        # normalize non-K mass so the 4 outcomes sum to 1
        rem = 1.0 - p_k
        other = p_bb + p_hit
        if other >= rem:  # degenerate: squeeze proportionally, keep some outs
            scale = (rem * 0.9) / other
            p_bb, p_hit = p_bb * scale, p_hit * scale
        p_out = 1.0 - p_k - p_bb - p_hit
        probs[i] = (p_k, p_bb, p_hit, p_out)
    return probs


def sample_pitch_limit(recent_pitch_counts, rng):
    """Backward-compatible name: draw a workload center, not a hard limit."""
    if recent_pitch_counts:
        mean = float(np.mean(recent_pitch_counts))
        sd = max(float(np.std(recent_pitch_counts)), 6.0)
    else:
        mean, sd = 88.0, 10.0
    return float(np.clip(rng.normal(mean, sd), 55.0, 118.0))


def hook_probability(pitches, workload_center, batters_faced,
                     pitcher_reaches, inning_pitcher_reaches,
                     expected_reach_rate, third_time=False,
                     inning_end=False):
    """Probability the manager removes the starter at this decision point.

    Hits and walks increment pitcher_reaches; defensive errors do not. This
    lets errors extend the inning through pitches without being treated as
    pitcher-created poor performance.
    """
    workload_pressure = (pitches - workload_center) / 7.0
    expected_reaches = batters_faced * expected_reach_rate
    performance = 0.25 * (pitcher_reaches - expected_reaches)
    inning_trouble = 0.8 * max(inning_pitcher_reaches - 2, 0)
    tto_pressure = 0.35 if third_time else 0.0
    score = workload_pressure + performance + inning_trouble + tto_pressure
    score = min(max(score, -20.0), 20.0)
    logistic = 1.0 / (1.0 + math.exp(-score))
    if inning_end and pitches >= 40:
        return float(logistic)
    if pitches >= workload_center:
        return float(0.20 * logistic)
    if pitches >= 45 and inning_pitcher_reaches >= 3:
        return float(0.20 * logistic)
    return 0.0


def _credible_rate(rate, sample, prior, threshold):
    if rate is None or sample <= 0:
        return prior
    weight = min(sample / float(threshold), 1.0)
    return weight * rate + (1.0 - weight) * prior


def defense_rates(context=None):
    """Resolve credibility-weighted DP and error rates from stored artifacts."""
    context = context or {}
    league_ground = context.get("league_ground_rate") or DEFAULT_GROUND_RATE
    league_dp = context.get("league_dp_rate")
    league_error = context.get("league_error_rate") or DEFAULT_ERROR_RATE
    ground = _credible_rate(
        context.get("pitcher_ground_rate"), context.get("pitcher_bip", 0),
        league_ground, GROUND_RATE_SAMPLE,
    )
    team_dp = _credible_rate(
        context.get("team_dp_rate"), context.get("team_dp_opportunities", 0),
        league_dp, TEAM_DP_SAMPLE,
    ) if league_dp is not None else None
    error = _credible_rate(
        context.get("team_error_rate"), context.get("team_bip", 0),
        league_error, TEAM_ERROR_SAMPLE,
    )
    ground_factor = ground / league_ground if league_ground else 1.0
    dp_factor = team_dp / league_dp if team_dp is not None and league_dp else 1.0
    return {
        "dp": float(np.clip(DP_PROB * ground_factor * dp_factor, 0.02, 0.30)),
        "error": float(np.clip(error, 0.0, 0.08)),
    }


def third_time_probs(base_probs, penalty=THIRD_TIME_PENALTY):
    """Fixed third-time-through penalty: less K, more BB/hit probability."""
    if penalty <= 0:
        return base_probs
    weights = np.array([1.0 - penalty, 1.0 + penalty,
                        1.0 + penalty, 1.0], dtype=float)
    adjusted = np.asarray(base_probs, dtype=float) * weights
    return adjusted / adjusted.sum()


def simulate_starter(pitcher, lineup, recent_pitch_counts=None,
                     k_park=1.0, k_ump=1.0, k_form=1.0, k_arsenal=1.0,
                     defense_context=None, third_time_penalty=THIRD_TIME_PENALTY,
                     n_sims=10000, seed=None):
    """Simulate n games; return (ks, outs) integer arrays of length n_sims."""
    rng = np.random.default_rng(seed)
    probs = build_matchup_probs(
        pitcher, lineup, k_park, k_ump, k_form, k_arsenal
    )
    n_batters = len(probs)
    cdf = np.cumsum(probs, axis=1)  # per-batter outcome CDF
    tto_cdf = np.cumsum(
        np.array([third_time_probs(row, third_time_penalty) for row in probs]),
        axis=1,
    )
    fielding = defense_rates(defense_context)

    ks_out = np.zeros(n_sims, dtype=np.int64)
    outs_out = np.zeros(n_sims, dtype=np.int64)

    for s in range(n_sims):
        workload_center = sample_pitch_limit(recent_pitch_counts, rng)
        ks = outs = 0
        pitches = 0.0
        batter_idx = 0
        runner_on_first = False
        outs_in_inning = 0
        pitcher_reaches = 0
        inning_pitcher_reaches = 0
        expected_reach_rate = min(pitcher["bb"] + pitcher["hit"], 0.95)

        while outs < MAX_OUTS and pitches < MAX_PITCHES:
            u = rng.random()
            # The third trip begins after 18 completed lineup plate appearances.
            row = (tto_cdf if batter_idx >= 2 * n_batters else cdf)[
                batter_idx % n_batters
            ]
            if u < row[0]:
                outcome = "K"
            elif u < row[1]:
                outcome = "WALK"
            elif u < row[2]:
                outcome = "HIT"
            else:
                outcome = "OUT"

            pitches += max(rng.normal(PITCHES_PER_PA[outcome], PITCH_SD), 1.0)

            if outcome == "K":
                ks += 1
                outs += 1
                outs_in_inning += 1
            elif outcome == "OUT":
                if rng.random() < fielding["error"]:
                    # An error extends the inning and costs pitches, but is not
                    # pitcher-created traffic for the future hook model.
                    runner_on_first = True
                elif (runner_on_first and outs_in_inning < 2
                      and rng.random() < fielding["dp"]):
                    outs += 2
                    outs_in_inning += 2
                    runner_on_first = False
                else:
                    outs += 1
                    outs_in_inning += 1
            elif outcome == "WALK":
                runner_on_first = True
                pitcher_reaches += 1
                inning_pitcher_reaches += 1
            elif outcome == "HIT":
                runner_on_first = True  # coarse: batter reaches, keep 1st occupied
                pitcher_reaches += 1
                inning_pitcher_reaches += 1

            batter_idx += 1
            inning_end = outs_in_inning >= 3
            hook_p = hook_probability(
                pitches, workload_center, batter_idx, pitcher_reaches,
                inning_pitcher_reaches, expected_reach_rate,
                third_time=batter_idx >= 2 * n_batters,
                inning_end=inning_end,
            )
            if rng.random() < hook_p:
                break
            if outs_in_inning >= 3:
                outs_in_inning = 0
                runner_on_first = False
                inning_pitcher_reaches = 0

        ks_out[s] = ks
        outs_out[s] = min(outs, MAX_OUTS)

    return ks_out, outs_out


def prob_over(samples, line):
    """P(over) with pushes excluded (standard grading for .0 lines).

    Returns (p_over, p_under, p_push).
    """
    samples = np.asarray(samples)
    over = float(np.mean(samples > line))
    under = float(np.mean(samples < line))
    push = max(0.0, 1.0 - over - under)
    if over + under > 0:  # condition on no-push, matches how books grade
        return over / (over + under), under / (over + under), push
    return 0.5, 0.5, push


def summarize(samples):
    samples = np.asarray(samples)
    return {
        "mean": float(samples.mean()),
        "p10": float(np.percentile(samples, 10)),
        "p50": float(np.percentile(samples, 50)),
        "p90": float(np.percentile(samples, 90)),
    }


if __name__ == "__main__":
    # smoke test: league-average pitcher vs league-average lineup
    avg_pitcher = dict(LEAGUE)
    avg_lineup = [dict(LEAGUE) for _ in range(9)]
    ks, outs = simulate_starter(avg_pitcher, avg_lineup,
                                recent_pitch_counts=[92, 88, 95, 90, 85],
                                n_sims=20000, seed=42)
    print("Ks  :", summarize(ks))
    print("Outs:", summarize(outs))
    for line in (4.5, 5.5, 6.5):
        po, pu, push = prob_over(ks, line)
        print(f"P(K > {line}) = {po:.3f}")
    for line in (15.5, 16.5, 17.5):
        po, pu, push = prob_over(outs, line)
        print(f"P(outs > {line}) = {po:.3f}")
