"""NBA Stats transport for NBA daily features and historical builders.

SDK imports are lazy so cached/offline data builders do not need nba_api installed.
Feature engineering and caching belong to the callers under nba/.
"""


def fetch_scoreboard(game_date):
    from nba_api.stats.endpoints import ScoreboardV2

    return ScoreboardV2(
        game_date=game_date, day_offset=0, league_id="00"
    ).get_data_frames()[0]


def fetch_team_stats(season, through_date, measure_type,
                     season_type="Regular Season", per_mode="PerGame"):
    from nba_api.stats.endpoints import LeagueDashTeamStats

    return LeagueDashTeamStats(
        season=season, season_type_all_star=season_type,
        measure_type_detailed_defense=measure_type,
        per_mode_detailed=per_mode, date_to_nullable=through_date, timeout=10,
    ).get_data_frames()[0]


def fetch_game_logs(season, season_type="Regular Season", team_only=False):
    from nba_api.stats.endpoints import LeagueGameFinder

    params = {"season_nullable": season, "season_type_nullable": season_type}
    if team_only:
        params.update(league_id_nullable="00", player_or_team_abbreviation="T", timeout=30)
    return LeagueGameFinder(**params).get_data_frames()[0]


def fetch_player_stats(season):
    from nba_api.stats.endpoints import LeagueDashPlayerStats

    return LeagueDashPlayerStats(
        season=season, season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]


def fetch_team_game_measures(season, measure_type):
    """Season of per-game team advanced/misc/four-factor/opponent records."""
    from nba_api.stats.endpoints import TeamGameLogs

    return TeamGameLogs(
        season_nullable=season, season_type_nullable="Regular Season",
        league_id_nullable="00", measure_type_player_game_logs_nullable=measure_type,
        per_mode_simple_nullable="Totals", timeout=30,
    ).get_data_frames()[0]
