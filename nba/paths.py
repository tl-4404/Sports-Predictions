"""Canonical NBA data and artifact paths, independent of the working directory."""
from pathlib import Path
from shared.paths import DATABASE_PATH, PROJECT_ROOT

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
FIXTURES_DIR = DATA_DIR / "fixtures"
OUTPUT_DIR = DATA_DIR / "outputs"
SAVED_MODELS_DIR = PACKAGE_ROOT / "saved_models"
# Retain the established constant for callers; this always means saved artifacts.
MODEL_DIR = SAVED_MODELS_DIR
