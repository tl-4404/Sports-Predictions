"""Chronological, policy-versioned calibration for MLB prop forecasts.

Only immutable, pregame, exact-line, priced observations are eligible. Repeated
intraday reruns cannot replace the first eligible decision snapshot. Platt scaling
is used while samples are limited; bucket tables remain diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from typing import Optional

from db import connect, ensure_schema
from decision import (
    FULL_SEGMENT_ROWS,
    MIN_DIAGNOSTIC_ROWS,
    MIN_VALIDATION_ROWS,
    POLICY_VERSION,
)

BUCKET_WIDTH = 0.05
EPS = 1e-6


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def _clip(p):
    return min(1.0 - EPS, max(EPS, float(p)))


def _logit(p):
    p = _clip(p)
    return math.log(p / (1.0 - p))


def _sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _eligible_rows(
    conn,
    through_date: Optional[str] = None,
    policy_version: Optional[str] = None,
    *,
    exclusive: bool = False,
):
    date_clause = ""
    args = []
    if through_date:
        date_clause = "AND p.game_date < ?" if exclusive else "AND p.game_date <= ?"
        args.append(through_date)
    policy_clause = ""
    if policy_version:
        policy_clause = "AND p.policy_version = ?"
        args.append(policy_version)

    # First eligible timestamp is selected before the main-line choice so a later
    # rerun cannot overwrite the historical decision snapshot.
    return conn.execute(
        f"""
        WITH eligible AS (
          SELECT p.*, r.outcome, r.clv, r.clv_available,
                 r.settlement_status, r.roi_eligible
          FROM prop_predictions p
          JOIN prop_results r ON r.prediction_id=p.id
          WHERE r.outcome IN ('over','under')
            AND r.actual IS NOT NULL
            AND r.settlement_status IN ('win','loss','graded')
            AND p.market_p_over IS NOT NULL
            AND p.line_verified=1
            AND p.line_tier='standard'
            AND p.line_as_of IS NOT NULL
            AND p.event_start IS NOT NULL
            AND p.created_at < p.event_start
            AND p.lineup_status='confirmed'
            AND p.starter_verified=1
            AND p.workload_verified=1
            AND p.provenance NOT IN ('synthetic','manual','reconstructed','estimated','user_reported')
            {date_clause}
            {policy_clause}
        ), first_snapshot AS (
          SELECT game_date, COALESCE(game_pk,-1) game_key,
                 COALESCE(player_id,-1) player_key, LOWER(player_name) player_key_name,
                 market, source, MIN(created_at) first_created
          FROM eligible
          GROUP BY game_date, game_key, player_key, player_key_name, market, source
        ), canonical AS (
          SELECT e.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY e.game_date, COALESCE(e.game_pk,-1),
                                COALESCE(e.player_id,-1), LOWER(e.player_name),
                                e.market, e.source
                   ORDER BY e.official DESC,
                            ABS(e.market_p_over - 0.5) ASC,
                            e.line ASC, e.id ASC
                 ) AS canonical_rank
          FROM eligible e
          JOIN first_snapshot f
            ON f.game_date=e.game_date
           AND f.game_key=COALESCE(e.game_pk,-1)
           AND f.player_key=COALESCE(e.player_id,-1)
           AND f.player_key_name=LOWER(e.player_name)
           AND f.market=e.market AND f.source=e.source
           AND f.first_created=e.created_at
        )
        SELECT id, game_date, market, source, policy_version,
               COALESCE(raw_p_over,p_over) raw_p_over,
               COALESCE(decision_p_over,p_over) decision_p_over,
               market_p_over, candidate_side, bet_side, official,
               external_mode, outcome, clv, clv_available, roi_eligible
        FROM canonical WHERE canonical_rank=1
        ORDER BY game_date,id
        """,
        args,
    ).fetchall()


def _target_rows(rows, target):
    out = []
    for row in rows:
        raw = _clip(row["raw_p_over"])
        y_over = 1.0 if row["outcome"] == "over" else 0.0
        if target == "forecast_over":
            p, y = raw, y_over
        else:
            side = row["candidate_side"] or row["bet_side"]
            if side not in {"over", "under"}:
                side = "over" if raw >= float(row["market_p_over"]) else "under"
            p = raw if side == "over" else 1.0 - raw
            y = y_over if side == "over" else 1.0 - y_over
        out.append((p, y, row))
    return out


def _fit_platt(pairs):
    """L2-shrunk Newton fit of y ~ sigmoid(a + b*logit(p))."""
    if not pairs:
        return 0.0, 1.0
    a, b = 0.0, 1.0
    ridge = 2.0 if len(pairs) < FULL_SEGMENT_ROWS else 0.5
    for _ in range(60):
        g0 = g1 = 0.0
        h00 = h01 = h11 = 0.0
        for p, y, _ in pairs:
            x = _logit(p)
            q = _sigmoid(a + b * x)
            diff = q - y
            weight = max(q * (1.0 - q), 1e-8)
            g0 += diff
            g1 += diff * x
            h00 += weight
            h01 += weight * x
            h11 += weight * x * x
        g0 += ridge * a
        g1 += ridge * (b - 1.0)
        h00 += ridge
        h11 += ridge
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        step_a = (h11 * g0 - h01 * g1) / det
        step_b = (-h01 * g0 + h00 * g1) / det
        a -= step_a
        b -= step_b
        if max(abs(step_a), abs(step_b)) < 1e-8:
            break
    return float(max(-3.0, min(3.0, a))), float(max(0.20, min(2.50, b)))


def _apply(p, a, b):
    return _clip(_sigmoid(a + b * _logit(p)))


def _metrics(pairs, a=0.0, b=1.0):
    if not pairs:
        return {"brier": None, "log_loss": None, "ece": None}
    values = [(_apply(p, a, b), y) for p, y, _ in pairs]
    brier = sum((p - y) ** 2 for p, y in values) / len(values)
    log_loss = sum(-(y * math.log(_clip(p)) + (1-y) * math.log(_clip(1-p)))
                   for p, y in values) / len(values)
    buckets = {}
    for p, y in values:
        key = min(0.95, math.floor(p / BUCKET_WIDTH) * BUCKET_WIDTH)
        group = buckets.setdefault(key, [0, 0.0, 0.0])
        group[0] += 1; group[1] += p; group[2] += y
    ece = sum(n * abs(y_sum/n - p_sum/n) for n,p_sum,y_sum in buckets.values()) / len(values)
    return {"brier": brier, "log_loss": log_loss, "ece": ece}


def _bucket_rows(pairs, a, b):
    grouped = {}
    for p, y, _ in pairs:
        calibrated = _apply(p, a, b)
        lo = round(math.floor(p / BUCKET_WIDTH) * BUCKET_WIDTH, 2)
        g = grouped.setdefault(lo, [0, 0.0, 0.0, 0.0])
        g[0] += 1; g[1] += p; g[2] += y; g[3] += calibrated
    return [
        {"lo": lo, "hi": min(1.0, lo + BUCKET_WIDTH), "n": vals[0],
         "mean": vals[1]/vals[0], "actual": vals[2]/vals[0],
         "calibrated": vals[3]/vals[0]}
        for lo, vals in sorted(grouped.items())
    ]


def fit(through_date=None, policy_version=POLICY_VERSION):
    conn = connect(); ensure_schema(conn)
    rows = list(_eligible_rows(conn, through_date, policy_version))
    cutoff = through_date or max((r["game_date"] for r in rows), default=date.today().isoformat())
    run_id = f"cal-{cutoff}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    markets = sorted({r["market"] for r in rows})
    sources = sorted({r["source"] for r in rows})
    segments = [("all","all",rows)]
    segments += [(m,"all",[r for r in rows if r["market"]==m]) for m in markets]
    segments += [("all",s,[r for r in rows if r["source"]==s]) for s in sources]
    segments += [(m,s,[r for r in rows if r["market"]==m and r["source"]==s])
                 for m in markets for s in sources]

    summary=[]
    for market, source, segment in segments:
        if not segment:
            continue
        for target in ("forecast_over","chosen_side"):
            pairs = _target_rows(segment, target)
            a,b = _fit_platt(pairs)
            metrics = _metrics(pairs,a,b)
            mode = "diagnostic" if len(pairs)<MIN_VALIDATION_ROWS else (
                "platt" if len(pairs)<FULL_SEGMENT_ROWS else "platt_mature")
            conn.execute(
                """INSERT INTO calibration_models
                   (run_id,created_at,through_date,market,source,policy_version,target,
                    sample_size,method,intercept,slope,brier_score,log_loss,ece,metadata_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id,utcnow(),cutoff,market,source,policy_version,target,len(pairs),
                 mode,a,b,metrics["brier"],metrics["log_loss"],metrics["ece"],
                 json.dumps({"chronological":True,"first_eligible_snapshot":True,
                             "exact_priced_rows_only":True,"refunds_excluded":True},sort_keys=True)),
            )
            if target=="forecast_over":
                conn.execute(
                    """INSERT INTO calibration_runs
                       (run_id,created_at,through_date,market,source,policy_version,
                        sample_size,mode,brier_score,log_loss,calibration_error,metadata_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (run_id,utcnow(),cutoff,market,source,policy_version,len(pairs),mode,
                     metrics["brier"],metrics["log_loss"],metrics["ece"],
                     json.dumps({"method":"platt","intercept":a,"slope":b},sort_keys=True)),
                )
                for bucket in _bucket_rows(pairs,a,b):
                    conn.execute(
                        """INSERT INTO calibration_buckets
                           (run_id,market,source,policy_version,bucket_low,bucket_high,
                            sample_size,mean_predicted,actual_rate,calibrated_probability)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (run_id,market,source,policy_version,bucket["lo"],bucket["hi"],
                         bucket["n"],bucket["mean"],bucket["actual"],bucket["calibrated"]),
                    )
            summary.append({"market":market,"source":source,"target":target,
                            "n":len(pairs),"mode":mode})
    conn.commit(); conn.close()
    return {"run_id":run_id,"through_date":cutoff,"policy_version":policy_version,
            "rows":len(rows),"segments":summary}


def _latest_model(conn, market, source, as_of_date, policy_version, target):
    candidates = ((market,source),(market,"all"),("all",source),("all","all"))
    for m,s in candidates:
        args=[m,s,policy_version,target]
        date_filter=""
        if as_of_date:
            date_filter="AND through_date < ?"; args.append(as_of_date)
        row=conn.execute(
            f"""SELECT * FROM calibration_models
                WHERE market=? AND source=? AND policy_version=? AND target=?
                {date_filter}
                ORDER BY through_date DESC,created_at DESC LIMIT 1""",args).fetchone()
        if row and int(row["sample_size"])>=MIN_VALIDATION_ROWS:
            return row
    return None


def calibrated_probability(conn, raw_p, market, source, as_of_date=None,
                           policy_version=POLICY_VERSION):
    raw_p=_clip(raw_p)
    model=_latest_model(conn,market,source,as_of_date,policy_version,"forecast_over")
    if not model:
        return raw_p
    return _apply(raw_p,float(model["intercept"]),float(model["slope"]))


def validation_status(conn, market, source="sim", as_of_date=None,
                      policy_version=POLICY_VERSION):
    rows=[r for r in _eligible_rows(conn,as_of_date,policy_version,exclusive=True)
          if r["market"]==market and r["source"]==source]
    if rows:
        model_brier=sum((float(r["raw_p_over"])-(1.0 if r["outcome"]=="over" else 0.0))**2
                        for r in rows)/len(rows)
        market_brier=sum((float(r["market_p_over"])-(1.0 if r["outcome"]=="over" else 0.0))**2
                         for r in rows)/len(rows)
    else:
        model_brier=market_brier=None
    clv=[float(r["clv"]) for r in rows if r["clv_available"] and r["clv"] is not None]
    clean_slates=len({r["game_date"] for r in rows})
    return {"policy_version":policy_version,"market":market,"source":source,"sample_size":len(rows),
            "model_brier":model_brier,"market_brier":market_brier,
            "average_clv":sum(clv)/len(clv) if clv else None,
            "clean_slates":clean_slates}


def report(through_date=None, policy_version=POLICY_VERSION):
    conn=connect(); ensure_schema(conn)
    rows=list(_eligible_rows(conn,through_date,policy_version))
    pairs=_target_rows(rows,"forecast_over")
    metrics=_metrics(pairs)
    chosen=_metrics(_target_rows(rows,"chosen_side"))
    conn.close()
    return "\n".join([
        f"MLB calibration report ({policy_version}) through {through_date or 'latest'}",
        f"canonical exact priced forecasts: {len(rows)}",
        f"raw Brier: {metrics['brier']:.4f}" if rows else "raw Brier: n/a",
        f"raw log loss: {metrics['log_loss']:.4f}" if rows else "raw log loss: n/a",
        f"raw ECE: {metrics['ece']:.4f}" if rows else "raw ECE: n/a",
        f"chosen-side Brier: {chosen['brier']:.4f}" if rows else "chosen-side Brier: n/a",
    ])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--through-date")
    parser.add_argument("--report-only",action="store_true")
    parser.add_argument("--policy-version",default=POLICY_VERSION)
    args=parser.parse_args()
    if not args.report_only:
        print(json.dumps(fit(args.through_date,args.policy_version),indent=2))
    print(report(args.through_date,args.policy_version))


if __name__=="__main__":
    main()
