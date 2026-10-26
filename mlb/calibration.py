"""Persistent calibration and performance reporting for MLB prop predictions.

Calibration uses only one historically valid, priced prediction per unique
game/player/market/line/source. Repeated intraday runs and synthetic observability
rows are deliberately excluded so they cannot create false sample size or
overconfidence.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from typing import Optional

from db import connect, ensure_schema
from integrity import MODEL_VERSION

MIN_AUTO_SAMPLE = 200
FULL_CALIBRATION_SAMPLE = 200
MIN_BUCKET_SAMPLE = 30
BUCKET_WIDTH = 0.05
EPS = 1e-6


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(p: float) -> float:
    return min(1.0 - EPS, max(EPS, float(p)))


def _bucket_floor(p: float) -> float:
    p = _clip(p)
    return math.floor(p / BUCKET_WIDTH) * BUCKET_WIDTH


def _eligible_rows(conn, through_date: Optional[str] = None):
    """Return deduplicated graded predictions suitable for calibration."""
    date_clause = "AND p.game_date <= ?" if through_date else ""
    args = [through_date] if through_date else []
    return conn.execute(
        f"""
        WITH ranked AS (
          SELECT
            p.id, p.game_date, p.market, p.source, p.raw_p_over AS p_over,
            p.market_p_over, p.bet_side, r.outcome, r.closing_p_over,
            ROW_NUMBER() OVER (
              PARTITION BY
                p.game_date,
                COALESCE(p.game_pk, -1),
                COALESCE(p.player_id, -1),
                p.market,
                p.source
              ORDER BY julianday(p.created_at), p.id
            ) AS recency_rank
          FROM prop_predictions p
          JOIN prop_results r ON r.prediction_id = p.id
          WHERE r.outcome IN ('over','under')
            AND r.actual IS NOT NULL
            AND p.market_p_over IS NOT NULL
            AND p.raw_p_over IS NOT NULL
            AND p.price_verified = 1
            AND p.model_version = '{MODEL_VERSION}'
            AND p.game_pk IS NOT NULL AND p.player_id IS NOT NULL
            AND julianday(p.created_at) < julianday(p.game_start_time)
            {date_clause}
        )
        SELECT id, game_date, market, source, p_over, market_p_over,
               bet_side, outcome, closing_p_over
        FROM ranked
        WHERE recency_rank = 1
        ORDER BY game_date, id
        """,
        args,
    ).fetchall()


def _brier(rows) -> Optional[float]:
    if not rows:
        return None
    return sum(
        (float(r["p_over"]) - (1.0 if r["outcome"] == "over" else 0.0)) ** 2
        for r in rows
    ) / len(rows)


def _log_loss(rows) -> Optional[float]:
    if not rows:
        return None
    total = 0.0
    for r in rows:
        p = _clip(r["p_over"])
        y = 1.0 if r["outcome"] == "over" else 0.0
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / len(rows)


def _calibration_error(buckets) -> Optional[float]:
    n = sum(b["n"] for b in buckets)
    if not n:
        return None
    return sum(
        b["n"] * abs(b["hit_rate"] - b["mean_pred"]) for b in buckets
    ) / n


def _build_buckets(rows):
    grouped = {}
    for r in rows:
        p = float(r["p_over"])
        lo = round(_bucket_floor(p), 2)
        g = grouped.setdefault(lo, {"n": 0, "pred_sum": 0.0, "wins": 0})
        g["n"] += 1
        g["pred_sum"] += p
        g["wins"] += int(r["outcome"] == "over")

    result = []
    for lo, g in sorted(grouped.items()):
        n = g["n"]
        mean_pred = g["pred_sum"] / n
        raw_rate = g["wins"] / n
        prior_strength = 50.0 if n < FULL_CALIBRATION_SAMPLE else 20.0
        calibrated = (
            g["wins"] + prior_strength * mean_pred
        ) / (n + prior_strength)
        result.append({
            "bucket_low": lo,
            "bucket_high": min(1.0, lo + BUCKET_WIDTH),
            "n": n,
            "mean_pred": mean_pred,
            "hit_rate": raw_rate,
            "calibrated_p": calibrated,
        })
    # Pool adjacent violators: an increased raw probability must never map to
    # a lower forecast. Apply to the shrunk event counts, not calibrated inputs.
    blocks = []
    for i, bucket in enumerate(result):
        weight = bucket["n"] + (50.0 if bucket["n"] < FULL_CALIBRATION_SAMPLE else 20.0)
        blocks.append([i, i, weight, weight * bucket["calibrated_p"]])
        while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    for start, end, weight, total in blocks:
        for i in range(start, end + 1):
            result[i]["calibrated_p"] = total / weight
    return result


def fit(through_date: Optional[str] = None) -> dict:
    """Fit and persist calibration snapshots through a historical cutoff."""
    conn = connect()
    ensure_schema(conn)
    rows = list(_eligible_rows(conn, through_date))
    cutoff = through_date or max(
        (r["game_date"] for r in rows), default=date.today().isoformat()
    )
    run_id = (
        f"cal-{cutoff}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )

    segments = [("all", "all", rows)]
    markets = sorted({r["market"] for r in rows})
    sources = sorted({r["source"] for r in rows})
    segments += [
        (market, "all", [r for r in rows if r["market"] == market])
        for market in markets
    ]
    segments += [
        ("all", source, [r for r in rows if r["source"] == source])
        for source in sources
    ]
    segments += [
        (
            market,
            source,
            [
                r for r in rows
                if r["market"] == market and r["source"] == source
            ],
        )
        for market in markets
        for source in sources
    ]

    inserted = 0
    summary = []
    for market, source, seg_rows in segments:
        if not seg_rows:
            continue
        buckets = _build_buckets(seg_rows)
        n = len(seg_rows)
        mode = (
            "diagnostic" if n < MIN_AUTO_SAMPLE
            else "full" if n >= FULL_CALIBRATION_SAMPLE
            else "light"
        )
        conn.execute(
            """INSERT INTO calibration_runs
               (run_id, created_at, through_date, market, source, sample_size,
                mode, brier_score, log_loss, calibration_error, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, utcnow(), cutoff, market, source, n, mode,
                _brier(seg_rows), _log_loss(seg_rows),
                _calibration_error(buckets),
                json.dumps({
                    "bucket_width": BUCKET_WIDTH,
                    "minimum_auto_sample": MIN_AUTO_SAMPLE,
                    "deduplicated": True,
                    "priced_rows_only": True,
                    "model_version": MODEL_VERSION,
                    "input": "raw_p_over",
                    "decision_protocol": "first_pregame_contract_per_game_player_market_source",
                    "minimum_bucket_sample": MIN_BUCKET_SAMPLE,
                    "fit_metrics_are_in_sample": True,
                }),
            ),
        )
        for bucket in buckets:
            conn.execute(
                """INSERT INTO calibration_buckets
                   (run_id, market, source, bucket_low, bucket_high, sample_size,
                    mean_predicted, actual_rate, calibrated_probability)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, market, source,
                    bucket["bucket_low"], bucket["bucket_high"], bucket["n"],
                    bucket["mean_pred"], bucket["hit_rate"],
                    bucket["calibrated_p"],
                ),
            )
            inserted += 1
        summary.append({
            "market": market, "source": source, "n": n, "mode": mode
        })

    conn.commit()
    conn.close()
    return {
        "run_id": run_id,
        "through_date": cutoff,
        "rows": len(rows),
        "buckets": inserted,
        "segments": summary,
    }


def calibrated_probability(
    conn,
    raw_p: float,
    market: str,
    source: str,
    as_of_date: Optional[str] = None,
    return_snapshot: bool = False,
) -> float:
    """Return the latest historically valid calibrated probability."""
    raw_p = _clip(raw_p)
    result = lambda probability, snapshot=None: (probability, snapshot) if return_snapshot else probability
    date_filter = ("AND cr.through_date < ? AND julianday(cr.created_at) < julianday(?)"
                   if as_of_date else "")
    args = [market, source]
    if as_of_date:
        args += [as_of_date, as_of_date]
    row = conn.execute(
        f"""
        SELECT cr.run_id, cr.sample_size, cr.metadata_json
        FROM calibration_runs cr
        WHERE cr.market = ? AND cr.source = ? {date_filter}
        ORDER BY cr.through_date DESC, cr.created_at DESC
        LIMIT 1
        """,
        args,
    ).fetchone()
    if not row or row["sample_size"] < MIN_AUTO_SAMPLE:
        return result(raw_p)
    metadata = json.loads(row["metadata_json"] or "{}")
    if metadata.get("model_version") != MODEL_VERSION or metadata.get("input") != "raw_p_over":
        return result(raw_p)
    buckets = conn.execute(
        """SELECT mean_predicted, calibrated_probability
           FROM calibration_buckets
           WHERE run_id = ? AND market = ? AND source = ?
             AND sample_size >= ? ORDER BY mean_predicted""",
        (row["run_id"], market, source, MIN_BUCKET_SAMPLE),
    ).fetchall()
    if len(buckets) < 2:
        return result(raw_p)
    knots = [(0.0, 0.0)] + [(b["mean_predicted"], b["calibrated_probability"]) for b in buckets] + [(1.0, 1.0)]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= raw_p <= x1:
            return result(_clip(y0 + (y1 - y0) * (raw_p - x0) / max(EPS, x1 - x0)), row["run_id"])
    return result(raw_p)


def report(through_date: Optional[str] = None) -> str:
    conn = connect()
    ensure_schema(conn)
    rows = list(_eligible_rows(conn, through_date))
    buckets = _build_buckets(rows)
    lines = [
        f"MLB calibration report through {through_date or 'latest'}",
        f"graded unique priced predictions: {len(rows)}",
        f"Brier score: {_brier(rows):.4f}" if rows else "Brier score: n/a",
        f"log loss: {_log_loss(rows):.4f}" if rows else "log loss: n/a",
        (
            f"expected calibration error: {_calibration_error(buckets):.4f}"
            if rows else "expected calibration error: n/a"
        ),
        "",
        "bucket       n    mean p   actual   calibrated",
    ]
    for bucket in buckets:
        lines.append(
            f"{bucket['bucket_low']:.2f}-{bucket['bucket_high']:.2f}  "
            f"{bucket['n']:4d}  {bucket['mean_pred']:.3f}    "
            f"{bucket['hit_rate']:.3f}    {bucket['calibrated_p']:.3f}"
        )
    conn.close()
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--through-date", default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        print(json.dumps(fit(args.through_date), indent=2))
    print(report(args.through_date))


if __name__ == "__main__":
    main()
