"""Training-side builder for cache/pitcher_form.json (CSW% + velocity trend).

Run on your own machine, periodically in-season (weekly is plenty) — it pulls
Statcast and is far too heavy for the daily serving run:

    python -m mlb.data_fetching.form 2026          # current season through today
    python -m mlb.data_fetching.form 2026 --end 2026-07-01

For each pitcher it records:
  csw_pct        called strikes + whiffs, per pitch (stabilizes fast)
  k_implied      season K% predicted from CSW% via a league regression fit on
                 this same pull (a CSW-based K-rate estimate, no leakage)
  velo_baseline  season mean fastball (FF, else SI) velocity
  velo_recent    mean fastball velocity over the pitcher's last RECENT_STARTS

Serving reads it via form.py; a missing pitcher falls back to neutral.
"""
import argparse
import json
import os
from datetime import date

import numpy as np
import pandas as pd
from pybaseball import statcast

from mlb.paths import CACHE_DIR

FORM_PATH = CACHE_DIR / "pitcher_form.json"

CSW_DESCS = {"called_strike", "swinging_strike", "swinging_strike_blocked",
             "swinging_strike_blocked"}
FB_TYPES = ("FF", "SI")
MIN_PITCHES = 200        # ignore pitchers with tiny samples for the regression
RECENT_STARTS = 3
KEEP_COLS = ["game_date", "pitcher", "pitch_type", "release_speed",
             "description", "events"]


def _csw_and_k(g):
    pitches = len(g)
    csw = g["description"].isin(CSW_DESCS).sum() / pitches if pitches else None
    pa = g["events"].notna().sum()
    ks = (g["events"] == "strikeout").sum() + \
         (g["events"] == "strikeout_double_play").sum()
    k_rate = ks / pa if pa else None
    return csw, k_rate, pitches, pa


def _fb_velo(g):
    fb = g[g["pitch_type"].isin(FB_TYPES)]
    return float(fb["release_speed"].mean()) if len(fb) else None


def _recent_fb_velo(g):
    dates = sorted(g["game_date"].dropna().unique())
    recent = g[g["game_date"].isin(dates[-RECENT_STARTS:])]
    return _fb_velo(recent)


def build_pitcher_form(season, end=None):
    end = end or date.today().isoformat()
    start = f"{season}-03-20"
    print(f"[data_form] fetching statcast {start}..{end} (heavy)...")
    df = statcast(start, end)[KEEP_COLS]
    if df.empty:
        print("[data_form] no data")
        return {}
    df["game_date"] = pd.to_datetime(df["game_date"])

    rows = {}
    for pid, g in df.groupby("pitcher"):
        csw, k_rate, pitches, pa = _csw_and_k(g)
        if pitches < MIN_PITCHES:
            continue
        rows[int(pid)] = {
            "csw_pct": round(csw, 4) if csw is not None else None,
            "k_rate": round(k_rate, 4) if k_rate is not None else None,
            "velo_baseline": round(_fb_velo(g), 2) if _fb_velo(g) is not None else None,
            "velo_recent": round(_recent_fb_velo(g), 2) if _recent_fb_velo(g) is not None else None,
            "pitches": int(pitches),
        }

    # fit league K% ~ CSW% on this pull, then store each pitcher's implied K%
    pts = [(r["csw_pct"], r["k_rate"]) for r in rows.values()
           if r["csw_pct"] is not None and r["k_rate"] is not None]
    if len(pts) >= 20:
        xs, ys = zip(*pts)
        b, a = np.polyfit(xs, ys, 1)  # k = a + b*csw
        for r in rows.values():
            if r["csw_pct"] is not None:
                r["k_implied"] = round(float(np.clip(a + b * r["csw_pct"], 0.10, 0.40)), 4)
        print(f"[data_form] CSW->K fit: k = {a:.3f} + {b:.3f}*csw (n={len(pts)})")
    else:
        print("[data_form] too few pitchers to fit CSW->K; k_implied omitted")

    out = {str(pid): r for pid, r in rows.items()}
    os.makedirs(FORM_PATH.parent, exist_ok=True)
    with open(FORM_PATH, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"[data_form] wrote {len(out)} pitchers → {FORM_PATH}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("season", type=int)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    build_pitcher_form(args.season, args.end)
