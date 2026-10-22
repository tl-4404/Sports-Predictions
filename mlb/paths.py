"""Canonical paths shared across MLB lifecycle stages."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CACHE_DIR = PACKAGE_ROOT / "cache"
FIXTURES_DIR = PACKAGE_ROOT / "fixtures"
MODEL_DIR = PACKAGE_ROOT / "models"
DATABASE_PATH = PROJECT_ROOT / "sportsaid.db"

