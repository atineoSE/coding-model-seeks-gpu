"""Environment variable loading and configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def get_env(key: str, default: str | None = None) -> str:
    """Get an environment variable or raise if missing and no default."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


# Paths
EXPORT_DIR = _PROJECT_ROOT / "web" / "public" / "data"
SUBMODULE_PATH = _PROJECT_ROOT / "external" / "openhands-index-results"
SNAPSHOTS_DIR = EXPORT_DIR / "snapshots"

# Email notifications (all optional — notifications disabled when any is missing)
SMTP_USER = os.getenv("SMTP_USER")  # Gmail address (also used as From)
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")  # App Password
NOTIFY_TO = os.getenv("NOTIFY_TO")  # Recipient email
