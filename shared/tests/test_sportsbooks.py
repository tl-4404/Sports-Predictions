"""Shared sportsbook transport and league adapter integration without network."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data_fetching.sportsbooks import OddsAPIClient


class SportsbookTests(unittest.TestCase):
    @patch('shared.data_fetching.sportsbooks.requests.get')
    def test_authenticated_request_and_response(self, get):
        response = get.return_value
        response.status_code = 200
        response.json.return_value = [{'id': 'event'}]
        client = OddsAPIClient('offline-key', base_url='https://example.test/v4/')
        self.assertEqual(client.get('/sports/example/odds', markets='h2h'),
                         [{'id': 'event'}])
        get.assert_called_once_with(
            'https://example.test/v4/sports/example/odds',
            params={'apiKey': 'offline-key', 'markets': 'h2h'}, timeout=20,
        )
        response.raise_for_status.assert_called_once()

    @patch('shared.data_fetching.sportsbooks.requests.get')
    def test_rejected_credentials_raise(self, get):
        get.return_value.status_code = 401
        with self.assertRaisesRegex(RuntimeError, 'rejected the key'):
            OddsAPIClient('offline-key').get('/sports')

    def test_missing_credentials_raise_before_request(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'ODDS_API_KEY'):
                OddsAPIClient()

    @patch('shared.data_fetching.sportsbooks.OddsAPIClient.get', return_value=[])
    def test_both_leagues_use_shared_transport(self, get):
        from nba.data_fetching.providers import odds as nba
        from mlb.data_fetching.providers import odds as mlb
        with patch.object(nba, 'API_KEY', 'offline-key'):
            self.assertEqual(nba.fetch_nba_odds(), [])
        get.assert_called_with(
            '/sports/basketball_nba/odds', regions='us', markets='h2h',
            oddsFormat='american', dateFormat='iso',
        )
        self.assertEqual(mlb.TheOddsAPIPropsFetcher('offline-key')._get(
            '/sports/baseball_mlb/events', dateFormat='iso'), [])
        get.assert_called_with('/sports/baseball_mlb/events', dateFormat='iso')


if __name__ == '__main__':
    unittest.main()
