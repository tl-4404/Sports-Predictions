"""Shared The Odds API transport for league-specific sportsbook adapters."""
import os
import requests

BASE_URL = "https://api.the-odds-api.com/v4"


class OddsAPIClient:
    def __init__(self, api_key=None, base_url=BASE_URL, timeout=20):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set ODDS_API_KEY in your environment/.env")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path, **params):
        params["apiKey"] = self.api_key
        response = requests.get(
            f"{self.base_url}/{path.lstrip('/')}", params=params,
            timeout=self.timeout,
        )
        if response.status_code == 401:
            raise RuntimeError(
                "The Odds API rejected the key. Note: player props require a "
                "paid plan — the free tier only serves featured markets."
            )
        response.raise_for_status()
        return response.json()
