"""Build idempotent postgame artifacts used by the daily simulator."""
from collections import defaultdict

from mlb.providers import mlb_api


SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_bunt",
    "foul_tip", "hit_into_play", "hit_into_play_no_out",
    "hit_into_play_score", "missed_bunt",
}
WHIFF_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
}
SWING_CODES = {"S", "F", "T", "W", "M", "X", "D", "E"}
WHIFF_CODES = {"S", "W", "M"}
DOUBLE_PLAY_EVENTS = {
    "grounded_into_double_play", "strikeout_double_play",
    "sac_fly_double_play", "double_play", "triple_play",
}
ERROR_EVENTS = {"field_error", "reached_on_error"}


def fetch_game_artifacts(game_pk, game_date, home_team, away_team):
    feed = mlb_api._get(f"/game/{game_pk}/playByPlay")
    return parse_game_artifacts(feed, game_pk, game_date, home_team, away_team)


def parse_game_artifacts(feed, game_pk, game_date, home_team, away_team):
    arsenal = defaultdict(lambda: {"pitches": 0, "swings": 0,
                                   "whiffs": 0, "balls_in_play": 0})
    batters = defaultdict(lambda: {"pitches": 0, "swings": 0, "whiffs": 0})
    batted = defaultdict(lambda: {"ground_balls": 0, "balls_in_play": 0})
    fielding = {
        home_team: {"balls_in_play": 0, "dp_opportunities": 0,
                    "double_plays": 0, "errors": 0},
        away_team: {"balls_in_play": 0, "dp_opportunities": 0,
                    "double_plays": 0, "errors": 0},
    }

    previous_half = None
    outs_before = 0
    for play in feed.get("allPlays", []):
        about = play.get("about", {})
        half_key = (about.get("inning"), about.get("halfInning"))
        if half_key != previous_half:
            outs_before = 0
            previous_half = half_key

        matchup = play.get("matchup", {})
        pitcher_id = (matchup.get("pitcher") or {}).get("id")
        batter_id = (matchup.get("batter") or {}).get("id")
        half = about.get("halfInning")
        defense = home_team if half == "top" else away_team
        result = play.get("result", {})
        event_type = result.get("eventType", "")
        trajectory = str((play.get("hitData") or {}).get("trajectory", "")).lower()
        runner_on_first = any(
            (runner.get("movement") or {}).get("start") == "1B"
            for runner in play.get("runners", [])
        )

        play_in_play = False
        for event in play.get("playEvents", []):
            if not event.get("isPitch") or not pitcher_id:
                continue
            details = event.get("details", {})
            pitch_type = ((details.get("type") or {}).get("code") or "UN").upper()
            pitch_code = str(details.get("code", "")).upper()
            description = str(details.get("description", "")).lower().replace(" ", "_")
            row = arsenal[(pitcher_id, pitch_type)]
            row["pitches"] += 1
            is_swing = pitch_code in SWING_CODES or description in SWING_DESCRIPTIONS
            is_whiff = pitch_code in WHIFF_CODES or description in WHIFF_DESCRIPTIONS
            row["swings"] += int(is_swing)
            row["whiffs"] += int(is_whiff)
            in_play = bool(details.get("isInPlay"))
            event_trajectory = str(
                (event.get("hitData") or {}).get("trajectory", "")
            ).lower()
            if event_trajectory:
                trajectory = event_trajectory
            row["balls_in_play"] += int(in_play)
            if batter_id:
                batter_row = batters[(batter_id, pitch_type)]
                batter_row["pitches"] += 1
                batter_row["swings"] += int(is_swing)
                batter_row["whiffs"] += int(is_whiff)
            play_in_play = play_in_play or in_play

        is_ground = "ground" in trajectory
        if pitcher_id and play_in_play:
            batted[pitcher_id]["balls_in_play"] += 1
            batted[pitcher_id]["ground_balls"] += int(is_ground)
            fielding[defense]["balls_in_play"] += 1
            if is_ground and runner_on_first and outs_before < 2:
                fielding[defense]["dp_opportunities"] += 1
            if event_type in ERROR_EVENTS:
                fielding[defense]["errors"] += 1
            if is_ground and event_type in DOUBLE_PLAY_EVENTS:
                fielding[defense]["double_plays"] += 1

        outs_before = int((play.get("count") or {}).get("outs", outs_before))

    arsenal_rows = [
        {"game_pk": game_pk, "game_date": game_date, "pitcher_id": pid,
         "pitch_type": pitch_type, **values}
        for (pid, pitch_type), values in arsenal.items()
    ]
    batted_rows = [
        {"game_pk": game_pk, "game_date": game_date, "pitcher_id": pid, **values}
        for pid, values in batted.items()
    ]
    batter_rows = [
        {"game_pk": game_pk, "game_date": game_date, "batter_id": bid,
         "pitch_type": pitch_type, **values}
        for (bid, pitch_type), values in batters.items()
    ]
    fielding_rows = [
        {"game_pk": game_pk, "game_date": game_date, "team_name": team, **values}
        for team, values in fielding.items()
    ]
    return arsenal_rows, batter_rows, batted_rows, fielding_rows
