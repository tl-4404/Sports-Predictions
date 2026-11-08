"""Offline coverage and temporal checks for enriched moneyline data."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nba.data_fetching.enrich_moneylines import ALL_STATS, MEASURES, legacy_coverage, main, merge_measures
from nba.data_fetching.build_backtest_data import build_dataset
from nba.tests.test_nba_backtest_builder import logs


def fixtures():
    base = logs()
    for field in ALL_STATS:
        if field not in base:
            base[field] = np.arange(len(base), dtype=float) + 1
    sources = {
        name: base[["GAME_ID", "TEAM_ID", "GAME_DATE", *fields]].copy()
        for name, fields in MEASURES.items()
    }
    # The base endpoint does not supply the extra selected measures.
    base = base.drop(columns=[f for fields in MEASURES.values() for f in fields])
    return base, sources


class EnrichmentTests(unittest.TestCase):
    def test_complete_join_and_feature_coverage(self):
        base, sources = fixtures()
        frame = build_dataset(merge_measures(base, sources), "2025-26", min_games=1,
                              stat_columns=ALL_STATS)
        features = [c for c in frame if c.startswith("diff_")]
        self.assertEqual(len(features), 116)
        self.assertTrue(np.isfinite(frame[features].to_numpy()).all())
        self.assertEqual(frame.iloc[0].home_season_OFF_RATING, 1)
        self.assertEqual(frame.iloc[1].home_season_OFF_RATING, 2)
        coverage = legacy_coverage(features)
        self.assertEqual(coverage["original_count"], 50)
        self.assertEqual(coverage["missing"], [])

    def test_bad_coverage_duplicate_date_and_missing_values_rejected(self):
        base, sources = fixtures()
        for problem in ("missing", "duplicate", "date", "nan", "field"):
            altered = {name: value.copy() for name, value in sources.items()}
            table = altered["Advanced"]
            if problem == "missing":
                altered["Advanced"] = table.iloc[1:]
            elif problem == "duplicate":
                altered["Advanced"] = pd.concat([table, table.iloc[[0]]])
            elif problem == "date":
                table.loc[0, "GAME_DATE"] = "2020-01-01"
            elif problem == "nan":
                table.loc[0, "OFF_RATING"] = np.nan
            else:
                altered["Advanced"] = table.drop(columns="OFF_RATING")
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                merge_measures(base, altered)

    def test_current_future_and_same_day_advanced_values_excluded(self):
        base, sources = fixtures()
        original = merge_measures(base, sources)
        changed = original.copy()
        changed.loc[changed.GAME_DATE >= pd.Timestamp("2025-10-22"), "OFF_RATING"] += 100
        before = build_dataset(original, "2025-26", min_games=1, stat_columns=ALL_STATS)
        after = build_dataset(changed, "2025-26", min_games=1, stat_columns=ALL_STATS)
        features = [c for c in before if c.startswith("diff_")]
        pd.testing.assert_series_equal(before.iloc[0][features], after.iloc[0][features])
        original.loc[original.GAME_DATE == pd.Timestamp("2025-10-22"), "GAME_DATE"] = pd.Timestamp("2025-10-21")
        same_day = build_dataset(original, "2025-26", min_games=0, stat_columns=ALL_STATS)
        self.assertTrue(same_day.iloc[:2].home_season_OFF_RATING.isna().all())

    def test_undefined_zero_attempt_percentage_is_not_imputed(self):
        base, sources = fixtures()
        base["FTA"] = 10
        base.loc[0, ["FTA", "FT_PCT"]] = [0, np.nan]
        merged = merge_measures(base, sources)
        self.assertTrue(pd.isna(merged.loc[0, "FT_PCT"]))
        base.loc[0, "FTA"] = 1
        with self.assertRaisesRegex(ValueError, "FT_PCT"):
            merge_measures(base, sources)

    def test_offline_cli_metadata(self):
        base, sources = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "enriched.csv"
            with patch("nba.data_fetching.enrich_moneylines.fetch_game_logs", return_value=base), patch(
                "nba.data_fetching.enrich_moneylines.fetch_measure_logs",
                side_effect=lambda season, measure, *args: sources[measure],
            ):
                main(["--seasons", "2025-26", "--min-games", "1", "--output", str(output)])
            meta = json.loads(output.with_suffix(".meta.json").read_text())
            self.assertEqual(meta["schema_version"], "nba_moneyline_enriched_v2")
            self.assertEqual(len(meta["feature_columns"]), 116)
            self.assertNotIn("target_home_points", meta["feature_columns"])
            self.assertFalse(meta["includes_odds"])
            self.assertEqual(meta["audit"]["games"], 2)


if __name__ == "__main__":
    unittest.main()
