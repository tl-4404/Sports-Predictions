"""Verify US moneyline direction and odds without network requests."""
import unittest

from providers.polymarket import moneyline_quotes, price_to_odds


class MoneylineTests(unittest.TestCase):
    def setUp(self):
        self.event = {"id": "1", "title": "Away vs Home"}
        # Short side first deliberately: list order must not determine direction.
        self.market = {
            "slug": "example", "active": True,
            "marketSides": [
                {"description": "Home", "long": False, "tradable": True},
                {"description": "Away", "long": True, "tradable": True},
            ],
        }
        self.book = {
            "marketSlug": "example", "state": "MARKET_STATE_OPEN",
            "bestBid": {"value": "0.43", "currency": "USD"},
            "bestAsk": {"value": "0.44", "currency": "USD"},
            "bidShares": "150", "askShares": "200",
        }

    def test_short_uses_complement_of_bid_long_uses_ask(self):
        short, long = moneyline_quotes(self.event, self.market, self.book)
        self.assertEqual(short["odds"]["contract_price"], 0.57)
        self.assertEqual(long["odds"]["contract_price"], 0.44)
        self.assertEqual(short["quoted_shares"], "150")
        self.assertEqual(long["quoted_shares"], "200")
        self.assertGreater(
            short["odds"]["contract_price"] + long["odds"]["contract_price"], 1
        )

    def test_missing_ask_does_not_fabricate_long_quote(self):
        del self.book["bestAsk"]
        short, long = moneyline_quotes(self.event, self.market, self.book)
        self.assertIsNotNone(short["odds"])
        self.assertIsNone(long["odds"])

    def test_closed_book_and_crossed_book_are_unavailable(self):
        self.book["state"] = "MARKET_STATE_SUSPENDED"
        self.assertTrue(all(row["odds"] is None for row in moneyline_quotes(
            self.event, self.market, self.book
        )))
        self.book["state"] = "MARKET_STATE_OPEN"
        self.book["bestBid"]["value"] = "0.45"
        self.assertTrue(all(row["odds"] is None for row in moneyline_quotes(
            self.event, self.market, self.book
        )))

    def test_rejects_wrong_market_and_ambiguous_sides(self):
        self.book["marketSlug"] = "wrong"
        with self.assertRaises(ValueError):
            moneyline_quotes(self.event, self.market, self.book)
        self.book["marketSlug"] = "example"
        self.market["marketSides"][0]["long"] = True
        with self.assertRaises(ValueError):
            moneyline_quotes(self.event, self.market, self.book)

    def test_standard_odds_conversions_and_invalid_prices(self):
        self.assertEqual(price_to_odds("0.25")["american_odds"], 300)
        self.assertEqual(price_to_odds("0.75")["american_odds"], -300)
        self.assertEqual(price_to_odds("0.5")["decimal_odds"], 2)
        for price in ("0", "1", "-0.1", "NaN", "Infinity"):
            with self.assertRaises(ValueError):
                price_to_odds(price)


if __name__ == "__main__":
    unittest.main()
