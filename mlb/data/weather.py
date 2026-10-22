import os
import json
import pandas as pd
import requests
from datetime import date

from mlb.paths import CACHE_DIR

WEATHER_CACHE_PATH = CACHE_DIR / "weather_cache.json"

# load cache once at import time; populated lazily as new dates are fetched
_weather_cache: dict = {}
if os.path.exists(WEATHER_CACHE_PATH):
    with open(WEATHER_CACHE_PATH) as _f:
        _weather_cache = json.load(_f)

# Stadium coordinates for all 30 MLB teams (home_team abbreviation → lat, lon)
STADIUMS = {
    "ARI": (33.4453, -112.0667),  # Chase Field, Phoenix AZ
    "ATL": (33.8908,  -84.4681),  # Truist Park, Cumberland GA
    "BAL": (39.2838,  -76.6218),  # Oriole Park at Camden Yards
    "BOS": (42.3467,  -71.0972),  # Fenway Park
    "CHC": (41.9484,  -87.6553),  # Wrigley Field
    "CHW": (41.8299,  -87.6338),  # Guaranteed Rate Field
    "CIN": (39.0979,  -84.5082),  # Great American Ball Park
    "CLE": (41.4962,  -81.6852),  # Progressive Field
    "COL": (39.7559, -104.9942),  # Coors Field, Denver CO
    "DET": (42.3390,  -83.0485),  # Comerica Park
    "HOU": (29.7573,  -95.3555),  # Minute Maid Park (retractable roof)
    "KC":  (39.0517,  -94.4803),  # Kauffman Stadium
    "LAA": (33.8003, -117.8827),  # Angel Stadium
    "LAD": (34.0739, -118.2400),  # Dodger Stadium
    "MIA": (25.7781,  -80.2197),  # loanDepot park (retractable roof)
    "MIL": (43.0280,  -87.9712),  # American Family Field (retractable roof)
    "MIN": (44.9817,  -93.2778),  # Target Field
    "NYM": (40.7571,  -73.8458),  # Citi Field
    "NYY": (40.8296,  -73.9262),  # Yankee Stadium
    "OAK": (37.7516, -122.2005),  # Oakland Coliseum (A's in Sacramento 2025+)
    "PHI": (39.9061,  -75.1665),  # Citizens Bank Park
    "PIT": (40.4469,  -80.0058),  # PNC Park
    "SD":  (32.7076, -117.1570),  # Petco Park
    "SEA": (47.5914, -122.3325),  # T-Mobile Park (retractable roof)
    "SF":  (37.7786, -122.3893),  # Oracle Park
    "STL": (38.6226,  -90.1928),  # Busch Stadium
    "TB":  (27.7683,  -82.6534),  # Tropicana Field (dome)
    "TEX": (32.7512,  -97.0832),  # Globe Life Field (retractable roof)
    "TOR": (43.6414,  -79.3894),  # Rogers Centre (retractable roof)
    "WSH": (38.8730,  -77.0074),  # Nationals Park
}

# approximate first-pitch hour in local time (most games 7–8pm)
FIRST_PITCH_HOUR = 19

# get weather for loca & time
# return weather features at first pitch 
def _save_weather_cache():
    os.makedirs(os.path.dirname(WEATHER_CACHE_PATH), exist_ok=True)
    with open(WEATHER_CACHE_PATH, "w") as f:
        json.dump(_weather_cache, f)


def get_weather(home_team, game_date):
    coords = STADIUMS.get(home_team)
    if coords is None:
        return {}

    date_str  = pd.Timestamp(game_date).strftime("%Y-%m-%d")
    cache_key = f"{home_team}_{date_str}"

    # return cached result if available
    if cache_key in _weather_cache:
        return _weather_cache[cache_key]

    lat, lon = coords
    game_dt  = pd.Timestamp(game_date).date()

    # historical vs forecast endpoint
    if game_dt < date.today():
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude":         lat,
        "longitude":        lon,
        "start_date":       date_str,
        "end_date":         date_str,
        "hourly":           "temperature_2m,windspeed_10m,winddirection_10m,relativehumidity_2m,precipitation",
        "temperature_unit": "fahrenheit",
        "windspeed_unit":   "mph",
        "timezone":         "auto",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return {}

    hourly = data.get("hourly", {})
    hours  = hourly.get("time", [])
    if not hours:
        return {}

    # find the hourly index closest to first pitch
    target = f"{date_str}T{FIRST_PITCH_HOUR:02d}:00"
    if target in hours:
        idx = hours.index(target)
    else:
        idx = min(range(len(hours)), key=lambda i: abs(
            pd.Timestamp(hours[i]).hour - FIRST_PITCH_HOUR
        ))

    result = {
        "temp_f":         hourly["temperature_2m"][idx],
        "wind_speed_mph": hourly["windspeed_10m"][idx],
        "wind_dir_deg":   hourly["winddirection_10m"][idx],  # 0=N, 90=E, 180=S, 270=W
        "humidity_pct":   hourly["relativehumidity_2m"][idx],
        "precip_mm":      hourly["precipitation"][idx],
    }

    # cache and persist so subsequent calls skip the HTTP request
    _weather_cache[cache_key] = result
    _save_weather_cache()
    return result


if __name__ == "__main__":
    # test: Dodger Stadium, 2024-09-01
    feats = get_weather("LAD", pd.Timestamp("2024-09-01"))
    for k, v in feats.items():
        print(f"{k}: {v}")
