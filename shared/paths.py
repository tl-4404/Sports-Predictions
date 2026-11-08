"""Repository paths shared by all leagues and the web application."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = PROJECT_ROOT / "shared"
DATA_DIR = SHARED_ROOT / "data"
DATABASE_PATH = DATA_DIR / "sportsaid.db"
