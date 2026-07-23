"""Persistent calibration and performance reporting for MLB prop predictions.

Calibration uses only one historically valid, priced prediction per unique
game/player/market/source, selecting the closest-to-even main line. Correlated
alternate-line ladders, repeated intraday runs, and synthetic observability rows
are excluded so they cannot create false sample size or overconfidence.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from typing import Optional

from db import connect, ensure_schema
from decision import MIN_VALIDATION_ROWS, POLICY_VERSION

MIN_AUTO_SAMPLE = MIN_VALIDATION_ROWS
FULL_CALIBRATION_SAMPLE = 200
BUCKET_WIDTH = 0.05
EPS = 1e-6


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(p: float) -> float:
    return min(1.0 - EPS, max(EPS, float(p)))


def _bucket_floor(p: float) -> float:
    p = _clip(p)
    return math.floor(p / BUCKET_WIDTH) * BUCKET_WIDTH


def _eligible_rows(
    conn,
    through_date: Optional[str] = None,
    policy_version: Optional[str] = None,
):
    """Return deduplicated graded predictions suitable for calibration."""
    date_clause = "AND p.game_date <= ?" if through_date else ""
    policy_clause = "AND p.policy_version = ?" if policy_version else ""
    args = []
    if through_date:
        args.append(through_date)
    if policy_version:
        args.append(policy_version)
    return conn.execute(
        f"""
        WITH ranked AS (
          SELECT
            p.id, p.game_date, p.market, p.source, p.p_over,
            COALESCE(p.raw_p_over, p.p_over) AS raw_p_over,
            COALESCE(p.decision_p_over, p.p_over) AS decision_p_over,
            p.market_p_over, p.bet_side, p.policy_version, r.outcome,
            r.closing_p_over, r.clv,
            ROW_NUMBER() OVER (
              PARTITION BY
                p.game_date,
                COALESCE(p.game_pk, -1),
                COALESCE(p.player_id, -1),
                LOWER(p.player_name),
                p.market,
                p.source
              ORDER BY ABS(p.market_p_over - 0.5) ASC,
                       p.created_at DESC, p.id DESC
            ) AS recency_rank
          FROM prop_predictions p
          JOIN prop_results r ON r.prediction_id = p.id
          WHERE r.outcome IN ('over','under')
            AND r.actual IS NOT NULL
            AND p.market_p_over IS NOT NULL
            {date_clause}
            {policy_clause}
        )
        SELECT id, game_date, market, source, p_over, market_p_over,
               raw_p_over, decision_p_over, bet_side, policy_version,
               outcome, closing_p_over, clv
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


def _brier_for(rows, column) -> Optional[float]:
    if not rows:
        return None
    return sum(
        (float(r[column]) - (1.0 if r["outcome"] == "over" else 0.0)) ** 2
        for r in rows
    ) / len(rows)


def _log_loss(rows, column="p_over") -> Optional[float]:
    if not rows:
        return None
    total = 0.0
    for r in rows:
        p = _clip(r[column])
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


def _build_buckets(rows, column="p_over"):
    grouped = {}
    for r in rows:
        p = float(r[column])
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
    return result


def fit(
    through_date: Optional[str] = None,
    policy_version: str = POLICY_VERSION,
) -> dict:
    """Fit and persist calibration snapshots through a historical cutoff."""
    conn = connect()
    ensure_schema(conn)
    rows = list(_eligible_rows(conn, through_date, policy_version))
    cutoff = through_date or max(
        (r["game_date"] for r in rows), default=date.today().isoformat()
    )
    run_id = (
        f"cal-{cutoff}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
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
        buckets = _build_buckets(seg_rows, "raw_p_over")
        n = len(seg_rows)
        mode = (
            "diagnostic" if n < MIN_AUTO_SAMPLE
            else "full" if n >= FULL_CALIBRATION_SAMPLE
            else "light"
        )
        conn.execute(
            """INSERT INTO calibration_runs
               (run_id, created_at, through_date, market, source,
                policy_version, sample_size, mode, brier_score, log_loss,
                calibration_error, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, utcnow(), cutoff, market, source, policy_version,
                n, mode,
                _brier_for(seg_rows, "raw_p_over"),
                _log_loss(seg_rows, "raw_p_over"),
                _calibration_error(buckets),
                json.dumps({
                    "bucket_width": BUCKET_WIDTH,
                    "minimum_auto_sample": MIN_AUTO_SAMPLE,
                    "deduplicated": True,
                    "priced_rows_only": True,
                    "one_observation_per_pitcher_market_game": True,
                    "policy_version": policy_version,
                }),
            ),
        )
        for bucket in buckets:
            conn.execute(
                """INSERT INTO calibration_buckets
                   (run_id, market, source, policy_version, bucket_low,
                    bucket_high, sample_size, mean_predicted, actual_rate,
                    calibrated_probability)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, market, source, policy_version,
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
        "policy_version": policy_version,
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
    policy_version: str = POLICY_VERSION,
) -> float:
    """Return the latest historically valid calibrated probability."""
    raw_p = _clip(raw_p)
    date_filter = "AND cr.through_date < ?" if as_of_date else ""
    args = [market, source, policy_version]
    if as_of_date:
        args.append(as_of_date)
    row = conn.execute(
        f"""
        SELECT cr.run_id, cr.sample_size
        FROM calibration_runs cr
        WHERE cr.market = ? AND cr.source = ?
          AND cr.policy_version = ? {date_filter}
        ORDER BY cr.through_date DESC, cr.created_at DESC
        LIMIT 1
        """,
        args,
    ).fetchone()
    if not row or row["sample_size"] < MIN_AUTO_SAMPLE:
        return raw_p
    bucket = conn.execute(
        """SELECT calibrated_probability
           FROM calibration_buckets
           WHERE run_id = ? AND market = ? AND source = ?
             AND policy_version = ?
             AND ? >= bucket_low AND ? < bucket_high
           ORDER BY bucket_low DESC LIMIT 1""",
        (row["run_id"], market, source, policy_version, raw_p, raw_p),
    ).fetchone()
    return _clip(bucket["calibrated_probability"]) if bucket else raw_p


def validation_status(
    conn,
    market: str,
    source: str = "sim",
    as_of_date: Optional[str] = None,
    policy_version: str = POLICY_VERSION,
) -> dict:
    """Out-of-sample proof required before automatic selection is enabled."""
    rows = [
        row
        for row in _eligible_rows(conn, as_of_date, policy_version)
        if row["market"] == market
        and row["source"] == source
        and row["clv"] is not None
    ]
    return {
        "policy_version": policy_version,
        "sample_size": len(rows),
        "model_brier": _brier_for(rows, "raw_p_over"),
        "market_brier": _brier_for(rows, "market_p_over"),
        "average_clv": (
            sum(float(row["clv"]) for row in rows) / len(rows)
            if rows
            else None
        ),
    }


def report(
    through_date: Optional[str] = None,
    policy_version: str = POLICY_VERSION,
) -> str:
    conn = connect()
    ensure_schema(conn)
    rows = list(_eligible_rows(conn, through_date, policy_version))
    buckets = _build_buckets(rows, "raw_p_over")
    lines = [
        (
            f"MLB calibration report ({policy_version}) through "
            f"{through_date or 'latest'}"
        ),
        f"graded unique priced predictions: {len(rows)}",
        (
            f"raw-model Brier score: {_brier_for(rows, 'raw_p_over'):.4f}"
            if rows
            else "raw-model Brier score: n/a"
        ),
        (
            f"raw-model log loss: {_log_loss(rows, 'raw_p_over'):.4f}"
            if rows
            else "raw-model log loss: n/a"
        ),
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
    parser.add_argument("--policy-version", default=POLICY_VERSION)
    args = parser.parse_args()
    if not args.report_only:
        print(json.dumps(fit(args.through_date, args.policy_version), indent=2))
    print(report(args.through_date, args.policy_version))


if __name__ == "__main__":
    main()
