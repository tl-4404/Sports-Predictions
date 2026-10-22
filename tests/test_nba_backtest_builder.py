"""Offline checks for NBA historical reconstruction and leakage boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nba.data.build_backtest_data import build_dataset, main


def logs():
    rows = []
    for i, (day, home_pts, away_pts) in enumerate([
        ("2025-10-21", 100, 90), ("2025-10-22", 95, 105),
        ("2025-10-25", 120, 110),
    ]):
        for team, points, matchup, won in [
            (1, home_pts, "AAA vs. BBB", home_pts > away_pts),
            (2, away_pts, "BBB @ AAA", away_pts > home_pts),
        ]:
            rows.append(dict(GAME_ID=f"002250000{i}", GAME_DATE=day, TEAM_ID=team,
                             TEAM_ABBREVIATION="AAA" if team == 1 else "BBB",
                             MATCHUP=matchup, WL="W" if won else "L", PTS=points))
    return pd.DataFrame(rows)


class BacktestBuilderTests(unittest.TestCase):
    def test_prior_games_only_and_rest(self):
        result = build_dataset(logs(), "2025-26", window=1, min_games=1)
        self.assertEqual(len(result), 2)
        first, last = result.iloc[0], result.iloc[1]
        self.assertEqual(first.home_season_PTS, 100)
        self.assertEqual(first.home_back_to_back, 1)
        self.assertEqual(first.feature_cutoff, "2025-10-21")
        self.assertEqual(last.home_season_PTS, 97.5)
        self.assertEqual(last.home_last1_PTS, 95)
        self.assertEqual(last.home_rest_days, 2)
        self.assertEqual(last.diff_season_PTS, 0)
        self.assertEqual(first.GAME_ID, "0022500001")

    def test_current_and_future_results_cannot_change_features(self):
        original = logs()
        changed = original.copy()
        changed.loc[changed.GAME_DATE >= "2025-10-22", "PTS"] += 50
        expected = build_dataset(original, "2025-26", min_games=1).iloc[0]
        actual = build_dataset(changed, "2025-26", min_games=1).iloc[0]
        features = [c for c in expected.index if c.startswith("diff_")]
        pd.testing.assert_series_equal(expected[features], actual[features])

    def test_same_day_results_are_excluded(self):
        source = logs()
        source.loc[source.GAME_DATE == "2025-10-22", "GAME_DATE"] = "2025-10-21"
        result = build_dataset(source, "2025-26", min_games=0)
        self.assertTrue(result.iloc[:2].home_prior_games.eq(0).all())

    def test_duplicates_fail_and_incomplete_games_are_excluded(self):
        source = logs()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_dataset(pd.concat([source, source.iloc[[0]]]), "2025-26")
        source.loc[source.GAME_DATE == "2025-10-22", "WL"] = None
        result = build_dataset(source, "2025-26", min_games=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].home_prior_games, 1)

    def test_offline_cli_and_feature_allowlist(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            raw, output = path / "raw.csv", path / "out.csv"
            logs().to_csv(raw, index=False)
            main(["--seasons", "2025-26", "--raw-csv", str(raw),
                  "--min-games", "1", "--output", str(output)])
            meta = json.loads(output.with_suffix(".meta.json").read_text())
            self.assertTrue(meta["feature_columns"])
            self.assertTrue(all(c.startswith("diff_") for c in meta["feature_columns"]))
            self.assertNotIn("target_home_points", meta["feature_columns"])
            self.assertFalse(meta["includes_odds"])
            self.assertEqual(len(pd.read_csv(output)), 2)


if __name__ == "__main__":
    unittest.main()
