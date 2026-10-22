"""Serving-side pitcher 'stuff & form' adjustments (CSW% + velocity trend).

Two signals the raw season K% can't capture on its own:

  CSW%  (called strikes + whiffs, per pitch) is a more stable, earlier
        predictor of strikeout ability than past K% — so we blend the
        season K rate with a CSW-implied K rate for a better estimate.

  Velocity trend (recent fastball velo vs the pitcher's own baseline) is
        NEW information not in the season line: a velo dip signals fatigue
        or injury and reliably suppresses both strikeouts and outings.
        Applied as a multiplicative K-odds factor in the simulator.

Both read cache/pitcher_form.json (built by mlb.data.form.build_pitcher_form on
your machine — heavyweight Statcast). Missing pitcher / missing cache → no
adjustment, exactly like the park & umpire factors. This module never imports
pybaseball, so it is safe on the serving box.
"""
import json

from mlb.paths import CACHE_DIR

FORM_PATH = CACHE_DIR / "pitcher_form.json"

# weight on the CSW-implied K rate when blending with the season rate.
# CSW% stabilizes in ~100 pitches vs ~hundreds of PA for K%, so it earns a
# meaningful but not dominant share of the estimate.
W_CSW = 0.40

# fastball-velocity effect on K odds: ~3% relative K change per mph off
# baseline, clamped so a single noisy reading can't swing a projection.
VELO_SLOPE = 0.03
VELO_CLAMP = (0.85, 1.10)

_cache = None


def _load():
    global _cache
    if _cache is None:
        if FORM_PATH.exists():
            try:
                with open(FORM_PATH) as f:
                    _cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                _cache = {}
        else:
            _cache = {}
    return _cache


def _reset_cache():
    global _cache
    _cache = None


def get_form(pitcher_id):
    """Cached form dict for a pitcher, or None."""
    return _load().get(str(pitcher_id))


def blended_k_rate(pitcher_id, k_season):
    """Blend the live season K rate with the CSW-implied K rate.

    Returns k_season unchanged when there's no form data — so behavior is
    identical to before once the cache is absent."""
    form = get_form(pitcher_id)
    if not form:
        return k_season
    implied = form.get("k_implied")
    if implied is None:
        return k_season
    return W_CSW * float(implied) + (1.0 - W_CSW) * k_season


def velo_k_factor(pitcher_id):
    """Multiplicative K-odds factor from the recent-vs-baseline velo trend.

    1.0 (neutral) when form data is missing. > 1.0 when throwing harder than
    baseline, < 1.0 on a dip."""
    form = get_form(pitcher_id)
    if not form:
        return 1.0
    recent = form.get("velo_recent")
    baseline = form.get("velo_baseline")
    if recent is None or baseline is None or not baseline:
        return 1.0
    delta = float(recent) - float(baseline)
    factor = 1.0 + VELO_SLOPE * delta
    return max(VELO_CLAMP[0], min(factor, VELO_CLAMP[1]))


def form_summary(pitcher_id):
    """Short human/LLM-facing description of a pitcher's form, or None."""
    form = get_form(pitcher_id)
    if not form:
        return None
    parts = []
    if form.get("csw_pct") is not None:
        parts.append(f"CSW% {form['csw_pct']:.1%}")
    if form.get("velo_recent") is not None and form.get("velo_baseline") is not None:
        delta = form["velo_recent"] - form["velo_baseline"]
        parts.append(f"velo {form['velo_recent']:.1f} vs {form['velo_baseline']:.1f} "
                     f"baseline ({delta:+.1f} mph)")
    return ", ".join(parts) if parts else None
