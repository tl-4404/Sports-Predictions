"""Plate-appearance-level Monte Carlo simulator for starting pitcher props.

The Bat-X-style core: instead of regressing to a point estimate, simulate the
game one plate appearance at a time and read P(over line) straight off the
resulting distribution of strikeouts and outs.

Per PA, the outcome is one of: K, WALK (incl. HBP), HIT, OUT_IN_PLAY.
Each per-PA probability blends the pitcher's rate with the batter's rate via
the standard odds-ratio (log5) method against the league baseline, then the
K probability is scaled by park and umpire factors.

The starter's workload is a mixture rather than a single average. Recent
high/clean starts estimate the planned ceiling; recent short, poor starts
estimate an early-hook component. The latter only activates when the simulated
starter is allowing traffic. This avoids treating every bad four-inning start
as a new manager-imposed pitch limit while still preserving downside.

A modest times-through-order K penalty is applied on later lineup turns. The
base-state remains deliberately small: errors/ROE are folded into hits and
sacrifice outs into balls in play.
"""
import numpy as np

LEAGUE = {"k": 0.223, "bb": 0.085, "hit": 0.212}

# mean pitches per PA by outcome (Statcast-era averages)
PITCHES_PER_PA = {"K": 4.8, "WALK": 5.5, "HIT": 3.5, "OUT": 3.6}
PITCH_SD = 1.4          # per-PA noise
DP_PROB = 0.11          # P(double play | ball in play out, runner on 1st, <2 outs)
MAX_OUTS = 27
TTO_K_FACTORS = (1.00, 0.97, 0.92, 0.88)


def odds_ratio(p_pitcher, p_batter, p_league):
    """log5 / odds-ratio blend of pitcher and batter event rates."""
    p_pitcher = min(max(p_pitcher, 1e-4), 1 - 1e-4)
    p_batter = min(max(p_batter, 1e-4), 1 - 1e-4)
    p_league = min(max(p_league, 1e-4), 1 - 1e-4)
    num = (p_pitcher * p_batter) / p_league
    den = num + ((1 - p_pitcher) * (1 - p_batter)) / (1 - p_league)
    return num / den


def build_matchup_probs(pitcher, lineup, k_park=1.0, k_ump=1.0, k_form=1.0):
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
        odds_k = p_k / (1 - p_k) * k_park * k_ump * k_form
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


def workload_parameters(recent_pitch_counts=None, recent_starts=None):
    """Estimate planned ceiling and bad-outing hook risk from recent starts."""
    starts = list(recent_starts or [])
    counts = [
        float(row["pitches"])
        for row in starts
        if row.get("pitches") is not None
    ]
    if not counts:
        counts = [float(value) for value in (recent_pitch_counts or [])]
    counts = counts[-8:]
    if not counts:
        return {
            "planned_mean": 88.0,
            "planned_sd": 10.0,
            "early_hook_rate": 0.25,
            "hook_mean": 70.0,
            "n_starts": 0,
        }

    arr = np.asarray(counts, dtype=float)
    # Upper recent usage is a better proxy for the manager's intended ceiling
    # than the raw mean, which is contaminated by bad-outing hooks.
    planned_mean = float(np.quantile(arr, 0.75) + 3.0)
    planned_mean = float(np.clip(planned_mean, 20.0, 115.0))
    planned_sd = float(np.clip(np.std(arr), 4.0, 10.0))
    low_cutoff = planned_mean - 10.0

    bad_short = []
    for index, count in enumerate(counts):
        if count >= low_cutoff:
            continue
        if starts and index < len(starts):
            start = starts[-len(counts):][index]
            outs = max(int(start.get("outs") or 0), 1)
            traffic = (
                int(start.get("hits") or 0) + int(start.get("walks") or 0)
            ) / outs
            bad = int(start.get("earned_runs") or 0) >= 4 or traffic >= 0.55
            if bad:
                bad_short.append(count)
        else:
            bad_short.append(count)

    # Beta(1, 3) prior: early hooks are real but not the default state.
    early_hook_rate = (len(bad_short) + 1.0) / (len(counts) + 4.0)
    hook_mean = (
        float(np.mean(bad_short))
        if bad_short
        else max(20.0, planned_mean - 16.0)
    )
    return {
        "planned_mean": planned_mean,
        "planned_sd": planned_sd,
        "early_hook_rate": float(np.clip(early_hook_rate, 0.05, 0.65)),
        "hook_mean": hook_mean,
        "n_starts": len(counts),
    }


def _sample_workload(recent_pitch_counts, recent_starts, rng):
    params = workload_parameters(recent_pitch_counts, recent_starts)
    planned = float(
        np.clip(
            rng.normal(params["planned_mean"], params["planned_sd"]),
            15.0,
            118.0,
        )
    )
    early = bool(rng.random() < params["early_hook_rate"])
    hook_at = float(
        np.clip(rng.normal(params["hook_mean"], 6.0), 15.0, planned)
    )
    return planned, early, hook_at


def sample_pitch_limit(recent_pitch_counts, rng, recent_starts=None):
    """Backward-compatible realized workload draw, including opener ranges."""
    planned, early, hook_at = _sample_workload(
        recent_pitch_counts, recent_starts, rng
    )
    return hook_at if early else planned


def _tto_cdfs(probs):
    """Build outcome CDFs with conservative K decay on later lineup turns."""
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


def simulate_starter(
    pitcher,
    lineup,
    recent_pitch_counts=None,
    recent_starts=None,
    k_park=1.0,
    k_ump=1.0,
    k_form=1.0,
    n_sims=10000,
    seed=None,
):
    """Simulate n games; return (ks, outs) integer arrays of length n_sims."""
    rng = np.random.default_rng(seed)
    probs = build_matchup_probs(pitcher, lineup, k_park, k_ump, k_form)
    n_batters = len(probs)
    cdfs = _tto_cdfs(probs)

    ks_out = np.zeros(n_sims, dtype=np.int64)
    outs_out = np.zeros(n_sims, dtype=np.int64)

    for s in range(n_sims):
        pitch_limit, early_hook, hook_at = _sample_workload(
            recent_pitch_counts, recent_starts, rng
        )
        ks = outs = 0
        pitches = 0.0
        batter_idx = 0
        batters_faced = 0
        baserunners = 0
        runner_on_first = False
        outs_in_inning = 0

        while outs < MAX_OUTS and pitches < pitch_limit:
            u = rng.random()
            trip = min(batter_idx // n_batters, len(cdfs) - 1)
            row = cdfs[trip][batter_idx % n_batters]
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
                if runner_on_first and outs_in_inning < 2 and rng.random() < DP_PROB:
                    outs += 2
                    outs_in_inning += 2
                    runner_on_first = False
                else:
                    outs += 1
                    outs_in_inning += 1
            elif outcome == "WALK":
                runner_on_first = True
                baserunners += 1
            elif outcome == "HIT":
                runner_on_first = True  # coarse: batter reaches, keep 1st occupied
                baserunners += 1

            batters_faced += 1
            if outs_in_inning >= 3:
                outs_in_inning = 0
                runner_on_first = False
                traffic_rate = baserunners / max(batters_faced, 1)
                if (
                    early_hook
                    and outs >= 9
                    and pitches >= hook_at
                    and (traffic_rate >= 0.25 or pitches >= hook_at + 8)
                ):
                    break
                # managers rarely send a starter back out deep over the limit
                if pitches >= pitch_limit - 8:
                    break

            batter_idx += 1

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
