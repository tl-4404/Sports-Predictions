"""Ownership and runtime path regressions for the repository layout."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class LayoutTests(unittest.TestCase):
    def test_shared_clients_do_not_depend_on_leagues_or_training(self):
        code = """
import builtins, importlib
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'nba', 'mlb', 'torch', 'pybaseball'}:
        raise AssertionError('Shared client imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
for name in ('http', 'polymarket', 'prizepicks', 'sportsbooks'):
    importlib.import_module('shared.data_fetching.' + name)
importlib.import_module('shared.betting.math')
"""
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)

    def test_paths_and_launch_wrapper_share_one_database(self):
        import app as launcher
        from shared.web import app as web
        from shared.paths import DATABASE_PATH
        from nba import paths as nba_paths
        from mlb import paths as mlb_paths
        from nba.data_fetching.providers import odds
        from mlb import db

        self.assertIs(launcher, web)
        expected = ROOT / 'shared/data/sportsaid.db'
        for actual in (DATABASE_PATH, web.DB_PATH, nba_paths.DATABASE_PATH,
                       mlb_paths.DATABASE_PATH, odds.DB_PATH, db.DB_PATH):
            self.assertEqual(actual, expected)
        self.assertEqual(web.NBA_PROBS_PATH,
                         ROOT / 'nba/data/outputs/current_games_with_probs.csv')
        self.assertEqual(Path(web.app.root_path).resolve(), ROOT / 'shared/web')
        self.assertTrue((Path(web.app.root_path) / web.app.template_folder / 'register.html').exists())

    def test_artifact_paths_are_independent_of_current_directory(self):
        from nba import paths as nba
        from mlb import paths as mlb
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as folder:
                os.chdir(folder)
                for league, paths in (('nba', nba), ('mlb', mlb)):
                    self.assertEqual(paths.CACHE_DIR, ROOT / league / 'data/cache')
                    self.assertEqual(paths.MODEL_DIR, ROOT / league / 'saved_models')
                    self.assertEqual(paths.OUTPUT_DIR, ROOT / league / 'data/outputs')
        finally:
            os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
