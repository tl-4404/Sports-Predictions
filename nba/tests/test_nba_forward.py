"""Offline NBA ledger arithmetic, market guards, agent tools and pipeline tests."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nba.ledger.domain import content_hash, decimal_value, paper_metrics, probability_scores, timestamp, validate_forecast
from nba.ledger.polymarket import discover_events, snapshot_packet
from nba.models.prediction.agent import forecast
from nba.models.prediction.research import season_for
from nba.data_fetching.providers.espn import get_team_injury_snapshot
from shared.data_fetching.polymarket import fetch_event, fetch_market_settlement
from nba import forward


NOW = datetime(2026, 10, 28, 20, tzinfo=timezone.utc)


def market_fixture():
    event = {"id": "1", "active": True, "startTime": (NOW+timedelta(hours=3)).isoformat()}
    market = {"slug": "nba-test", "active": True, "sportsMarketTypeV2": "SPORTS_MARKET_TYPE_MONEYLINE",
              "sportsMarketType": "basketball_team_full_game_winner", "marketSides": [
                  {"description": "Celtics", "long": False, "tradable": True,
                   "team": {"name": "Boston Celtics", "league": "nba"}},
                  {"description": "Knicks", "long": True, "tradable": True,
                   "team": {"name": "New York Knicks", "league": "nba"}},
              ]}
    book = {"marketSlug": "nba-test", "state": "MARKET_STATE_OPEN",
            "bestAsk": {"currency": "USD", "value": "0.44"},
            "bestBid": {"currency": "USD", "value": "0.43"}, "askShares": "10", "bidShares": "20"}
    return event, market, book


def evidence_fixture():
    return {"model_included": False, "odds_included": False, "event": {},
            "evidence": {"long_base": {"stats": {"GP": 10}},
                         "short_base": {"stats": {"GP": 10}},
                         "long_injuries": {"status": "available"},
                         "short_injuries": {"status": "available"}}}


def output_fixture():
    return {"long_win_probability": 0.6, "rationale": "Supplied evidence supports this estimate.",
            "evidence_ids": ["long_base", "short_base"], "uncertainties": [], "abstain": False}


def response(calls, stop_reason="tool_use"):
    content = [{"type": "tool_use", "id": str(i), "name": name, "input": args}
               for i, (name, args) in enumerate(calls)]
    raw = {"content": content, "stop_reason": stop_reason, "usage": {"input_tokens": 10}}
    return SimpleNamespace(stop_reason=stop_reason, model_dump=lambda **kw: raw)


class ForwardTests(unittest.TestCase):
    def test_market_direction_pregame_and_wrong_contract(self):
        event, market, book = market_fixture()
        packet = snapshot_packet(event, market, book, NOW)
        self.assertEqual(packet["long_team"], "New York Knicks")
        self.assertEqual(packet["short_price"], Decimal("0.57"))
        self.assertTrue(packet["pregame"])
        event["live"] = True
        self.assertFalse(snapshot_packet(event, market, book, NOW)["pregame"])
        event["live"] = None
        self.assertFalse(snapshot_packet(event, market, book, NOW+timedelta(hours=4))["pregame"])
        market["sportsMarketType"] = "basketball_team_first_half_winner"
        with self.assertRaises(ValueError):
            snapshot_packet(event, market, book, NOW)

    def test_schedule_disagreement_and_identity_fail_closed(self):
        event, market, book = market_fixture()
        market["gameStartTime"] = (NOW+timedelta(hours=4)).isoformat()
        with self.assertRaisesRegex(ValueError, "disagree"):
            snapshot_packet(event, market, book, NOW)
        del market["gameStartTime"]
        market["marketSides"][0]["team"]["league"] = "mlb"
        with self.assertRaises(ValueError):
            snapshot_packet(event, market, book, NOW)

    def test_prices_expected_value_scores_and_unknown_fees(self):
        metrics = paper_metrics("0.6", "0.4", payout=1)
        self.assertEqual(metrics["gross_ev_per_share"], Decimal("0.2"))
        self.assertEqual(metrics["gross_pnl_per_share"], Decimal("0.6"))
        self.assertIsNone(metrics["net_pnl_per_share"])
        self.assertEqual(paper_metrics("0.6", "0.4", 0, "0.01")["net_pnl_per_share"], Decimal("-0.41"))
        self.assertAlmostEqual(probability_scores(0.6, 1)["brier"], 0.16)
        for value in (True, "NaN", "Infinity", None):
            with self.assertRaises(ValueError):
                decimal_value(value)
        with self.assertRaises(ValueError):
            timestamp(datetime(2026, 1, 1))

    def test_schema_unknown_citations_and_probability_rejected(self):
        value = output_fixture()
        for mutation in ({"long_win_probability": 1.1}, {"long_win_probability": True},
                         {"evidence_ids": ["invented"]}, {"abstain": "no"}, {"extra": 1}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_forecast({**value, **mutation}, {"long_base", "short_base"})
        self.assertEqual(content_hash({"a": 1, "b": 2}), content_hash({"b": 2, "a": 1}))

    def test_agent_reads_evidence_then_submits_and_policy_can_abstain(self):
        for incomplete in (False, True):
            packet = evidence_fixture()
            if incomplete:
                packet["evidence"]["long_injuries"]["status"] = "unavailable"
            client = MagicMock()
            client.messages.create.side_effect = [
                response([("read_evidence", {"evidence_id": key}) for key in packet["evidence"]]),
                response([("submit_forecast", output_fixture())]),
            ]
            traces = []
            result, raw = forecast(packet, "test-model", client=client, trace=traces.append)
            self.assertEqual(result["abstain"], incomplete)
            self.assertEqual(len(traces), 2)
            self.assertEqual(len(raw["responses"]), 2)

    def test_agent_no_evidence_bypass_no_unsafe_tools_no_model_and_bounded(self):
        packet = evidence_fixture()
        for calls in ([('submit_forecast', output_fixture())], [('execute_python', {})]):
            client = MagicMock()
            client.messages.create.return_value = response(calls)
            with self.assertRaises(ValueError):
                forecast(packet, "test", client=client)
        client = MagicMock()
        client.messages.create.return_value = response([("read_evidence", {"evidence_id": "long_base"})])
        with self.assertRaisesRegex(ValueError, "turn limit"):
            forecast(packet, "test", client=client, max_turns=1)
        with self.assertRaises(ValueError):
            forecast({**packet, "model_included": True}, "test", client=client)

    @patch("nba.ledger.polymarket.polymarket.fetch_league_events")
    def test_paginated_discovery_and_truncation(self, fetch):
        fetch.side_effect = [{"events": [{"id": "1"}, {"id": "2"}]}, {"events": [{"id": "2"}]}]
        self.assertEqual(len(discover_events(2, 2)), 2)
        fetch.side_effect = None
        fetch.return_value = {"events": [{"id": "1"}]}
        with self.assertRaisesRegex(RuntimeError, "max-pages"):
            discover_events(1, 2)

    @patch("shared.data_fetching.polymarket.requests.get")
    def test_settlement_endpoint_not_display_price_and_zero_is_valid(self, get):
        get.return_value.status_code = 404
        self.assertIsNone(fetch_market_settlement("nba-test"))
        get.return_value.status_code = 200
        get.return_value.json.return_value = {"slug": "nba-test", "settlement": 0}
        self.assertEqual(fetch_market_settlement("nba-test")["settlement"], 0)
        get.return_value.json.return_value = {"slug": "wrong", "settlement": 1}
        with self.assertRaises(ValueError):
            fetch_market_settlement("nba-test")
        get.return_value.json.return_value = {"event": {"id": "wrong"}}
        with self.assertRaises(ValueError):
            fetch_event(123)

    @patch("nba.data_fetching.providers.espn._fetch")
    def test_injury_failure_not_reported_as_healthy(self, fetch):
        fetch.side_effect = RuntimeError("offline")
        self.assertEqual(get_team_injury_snapshot("Boston Celtics")["status"], "unavailable")
        fetch.side_effect = None
        fetch.return_value = {"items": [], "count": 0}
        self.assertEqual(get_team_injury_snapshot("Boston Celtics")["status"], "available")

    def test_season_rollover(self):
        self.assertEqual(season_for(NOW.date()), "2026-27")
        self.assertEqual(season_for(datetime(2026, 4, 1).date()), "2025-26")

    def test_prediction_pipeline_refreshes_entry_and_keeps_model_out(self):
        event, market, book = market_fixture()
        packet = snapshot_packet(event, market, book, NOW)
        event["markets"] = [market]
        with patch.object(forward, "utcnow", return_value=NOW), patch.object(
            forward, "discover_events", return_value=[event]
        ), patch.object(forward, "refresh_packet", return_value=packet) as refresh, patch.object(
            forward.agent, "configured_model", return_value="test-model"
        ), patch.object(forward, "ResearchCollector") as collector, patch.object(
            forward.agent, "forecast", return_value=(output_fixture(), {})
        ), patch.object(forward.db, "save_snapshot", return_value="snapshot"), patch.object(
            forward.db, "save_attempt", return_value="attempt"
        ), patch.object(forward.db, "save_prediction", return_value="prediction") as save:
            collector.return_value.collect.return_value = evidence_fixture()
            result = forward.run_agent(MagicMock(), "run")
            self.assertEqual(result["predictions"], 1)
            self.assertEqual(refresh.call_count, 2)
            self.assertEqual(save.call_args.kwargs["source"], "agent")
            self.assertFalse(save.call_args.kwargs["inputs"]["model_included"])


if __name__ == "__main__":
    unittest.main()
