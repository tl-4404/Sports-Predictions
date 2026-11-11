"""PostgreSQL NBA paper testing: init-db | odds | agent | grade | report.

Examples: python -m nba.forward init-db
          python -m nba.forward odds
          python -m nba.forward agent --max-games 3
          python -m nba.forward grade
          python -m nba.forward report
"""
import argparse
from datetime import timedelta
import json
import os
from uuid import UUID

from nba.ledger import db
from nba.ledger.domain import decimal_value, full_game_market, market_start, timestamp, utcnow
from nba.ledger.polymarket import discover_events, refresh_packet, snapshot_packet
from nba.models.prediction import agent
from nba.models.prediction.research import ResearchCollector
from shared.data_fetching import polymarket


def safe_error(exc):
    # Provider/database exception strings can contain credentials; keep diagnostics
    # in the terminal/debugger, never copy arbitrary exception text into a ledger.
    return {"error_type": type(exc).__name__}


def collect_odds(conn, run_id, *, page_size=50, max_pages=10, horizon_hours=48):
    now = utcnow()
    candidates = {}
    for event in discover_events(page_size, max_pages):
        for market in event.get("markets") or []:
            if not full_game_market(market):
                continue
            try:
                start = market_start(event, market)
                if now < start <= now+timedelta(hours=horizon_hours):
                    candidates[market["slug"]] = (event, market)
            except (ValueError, TypeError) as exc:
                db.log_event(conn, run_id, "market_rejected", {"market_slug": market.get("slug"), **safe_error(exc)})
    # Refresh predictions' tracked events even if they disappear from discovery.
    tracked = {row["market_slug"]: row for row in db.tracked_markets(conn)
               if now < row["scheduled_start"] <= now+timedelta(hours=horizon_hours)}
    saved, errors = 0, 0
    for slug in sorted(set(candidates) | set(tracked)):
        try:
            if slug in candidates:
                event, market = candidates[slug]
                packet = snapshot_packet(event, market, polymarket.fetch_market_bbo(slug))
            else:
                packet = refresh_packet(tracked[slug]["event_id"], slug)
            db.save_snapshot(conn, run_id, packet)
            saved += 1
        except Exception as exc:
            errors += 1
            db.log_event(conn, run_id, "quote_failed", {"market_slug": slug, **safe_error(exc)})
    return {"snapshots": saved, "errors": errors}


def run_agent(conn, run_id, *, max_games=3, page_size=50, max_pages=10, fee_per_share=None, paper_shares=1):
    model = agent.configured_model()
    collector = ResearchCollector()
    now = utcnow()
    candidates = {}
    for event in discover_events(page_size, max_pages):
        for market in event.get("markets") or []:
            if not full_game_market(market):
                continue
            try:
                start = market_start(event, market)
                if now+timedelta(minutes=5) < start <= now+timedelta(hours=24):
                    candidates[market["slug"]] = (start, event)
            except (ValueError, TypeError):
                continue
    predictions, errors, skipped, attempted = 0, 0, 0, 0
    for slug, (start, event) in sorted(candidates.items(), key=lambda item: (item[1][0], item[0])):
        if attempted >= max_games:
            break
        attempted += 1
        try:
            # Re-read current event state instead of trusting discovery or old CSVs.
            initial = refresh_packet(str(event["id"]), slug)
            db.save_snapshot(conn, run_id, initial)
            if not initial["pregame"] or all(initial[k] is None for k in ("long_price", "short_price")):
                skipped += 1
                db.log_event(conn, run_id, "prediction_skipped", {"market_slug": slug, "reason": "no_pregame_quote"})
                continue
            inputs = collector.collect(initial)
            attempt_id = db.save_attempt(conn, run_id, slug, model, agent.PROMPT_VERSION, agent.SYSTEM_PROMPT, inputs)
            validated, raw = agent.forecast(inputs, model, trace=lambda payload: db.log_event(
                conn, run_id, "agent_response", {"attempt_id": str(attempt_id), **payload}))
            # LLM latency must not turn an old quote into a claimed entry price.
            entry = refresh_packet(initial["event_id"], slug)
            snapshot_id = db.save_snapshot(conn, run_id, entry)
            for key in ("long_team", "short_team", "scheduled_start"):
                if entry[key] != initial[key]:
                    raise ValueError("Market identity/schedule changed during research")
            prediction_id = db.save_prediction(
                conn, run_id=run_id, snapshot_id=snapshot_id, packet=entry, forecast=validated,
                inputs=inputs, raw_response=raw, source="agent", model=model,
                version=agent.PROMPT_VERSION, attempt_id=attempt_id,
                fee_per_share=fee_per_share, paper_shares=paper_shares)
            predictions += 1
            print(json.dumps({"prediction_id": str(prediction_id), "market_slug": slug,
                              "long_win_probability": validated["long_win_probability"],
                              "abstain": validated["abstain"]}))
        except Exception as exc:
            errors += 1
            db.log_event(conn, run_id, "prediction_failed", {"market_slug": slug, **safe_error(exc)})
            print(json.dumps({"market_slug": slug, **safe_error(exc)}))
    return {"predictions": predictions, "errors": errors, "skipped": skipped,
            "candidates": len(candidates), "attempted": attempted}


def grade(conn, run_id):
    saved, pending, errors = 0, 0, 0
    for row in db.tracked_markets(conn):
        if row["scheduled_start"] > utcnow():
            continue
        try:
            response = polymarket.fetch_market_settlement(row["market_slug"])
            if response is None:
                pending += 1
            else:
                saved += int(db.save_settlement(conn, run_id, row["market_slug"], response))
        except Exception as exc:
            errors += 1
            db.log_event(conn, run_id, "settlement_failed", {"market_slug": row["market_slug"], **safe_error(exc)})
    return {"new_settlements": saved, "pending": pending, "errors": errors}


def json_default(value):
    if isinstance(value, UUID):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    # Decimal stays a string in exports, preserving price precision.
    from decimal import Decimal
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(type(value).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init-db", "odds", "agent", "grade", "report"))
    parser.add_argument("--max-games", type=int, default=3, help="Bound paid LLM work per invocation")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--horizon-hours", type=int, default=48)
    parser.add_argument("--fee-per-share", help="Explicit USD paper fee assumption; absent means unknown/no paper bets")
    parser.add_argument("--paper-shares", default="1")
    parser.add_argument("--limit", type=int, default=100, help="Maximum report rows")
    args = parser.parse_args(argv)
    if not (1 <= args.max_games <= 30 and 1 <= args.page_size <= 100
            and 1 <= args.max_pages <= 100 and 1 <= args.horizon_hours <= 168 and 1 <= args.limit <= 10000):
        parser.error("Arguments exceed their documented positive bounds")
    try:
        fee = None if args.fee_per_share is None else decimal_value(args.fee_per_share, minimum=0)
        shares = decimal_value(args.paper_shares, minimum=0)
        if shares <= 0:
            raise ValueError("Positive shares required")
    except ValueError:
        parser.error("Fees must be finite/nonnegative; shares must be finite/positive")
    with db.connect() as conn:
        if args.stage == "init-db":
            db.initialize(conn)
            print("Quant PostgreSQL ledger schema v2 ready; existing SQLite app data unchanged.")
            return 0
        # Fail explicitly if setup hasn't run, rather than silently creating storage.
        if not conn.execute("SELECT 1 FROM quant_ledger.migrations WHERE version=2").fetchone():
            raise RuntimeError("Run python -m nba.forward init-db first")
        if args.stage == "report":
            print(json.dumps(db.report(conn, args.limit), default=json_default, indent=2))
            return 0
        config = {"page_size": args.page_size, "max_pages": args.max_pages,
                  "max_games": args.max_games, "horizon_hours": args.horizon_hours,
                  "fee_per_share": None if fee is None else str(fee), "paper_shares": str(shares),
                  "model": os.getenv("NBA_AGENT_MODEL"), "mode": "paper_only", "model_included": False}
        run_id = db.start_run(conn, args.stage, config)
        try:
            if args.stage == "odds":
                result = collect_odds(conn, run_id, page_size=args.page_size, max_pages=args.max_pages,
                                      horizon_hours=args.horizon_hours)
            elif args.stage == "agent":
                result = run_agent(conn, run_id, max_games=args.max_games, page_size=args.page_size,
                                   max_pages=args.max_pages, fee_per_share=fee, paper_shares=shares)
            else:
                result = grade(conn, run_id)
            db.log_event(conn, run_id, "completed_with_errors" if result["errors"] else "completed", result)
            print(json.dumps({"run_id": str(run_id), **result}))
            return int(result["errors"] > 0)
        except Exception as exc:
            db.log_event(conn, run_id, "failed", safe_error(exc))
            raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", **safe_error(exc),
                          "hint": "Check DB/schema, NBA_AGENT_MODEL, API credentials and provider access. See docs/NBA_FORWARD_TESTING.md."}))
        raise SystemExit(1)
