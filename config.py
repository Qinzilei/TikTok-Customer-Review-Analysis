"""Configuration for TikTok Google Play review analysis pipeline."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

# App target
APP_ID = "com.zhiliaoapp.musically"
APP_URL = (
    "https://play.google.com/store/apps/details"
    "?id=com.zhiliaoapp.musically&hl=en_us"
)
LANG = "en"
COUNTRY = "us"
APP_NAME = "TikTok"

# Date window
LOOKBACK_DAYS = 30

# Scraping
REVIEWS_PER_PAGE = 200
SLEEP_SECONDS = 0.5
MAX_RETRIES = 3

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_REPORTS = PROJECT_ROOT / "data" / "reports"
NOTEBOOKS = PROJECT_ROOT / "notebooks"

RAW_JSON = DATA_RAW / "reviews_raw.json"
PROCESSED_PARQUET = DATA_PROCESSED / "reviews.parquet"
PROCESSED_CSV = DATA_PROCESSED / "reviews.csv"
METRICS_JSON = DATA_REPORTS / "metrics.json"
REPORT_HTML = DATA_REPORTS / "review_report.html"
DOCS_HTML = PROJECT_ROOT / "docs" / "index.html"

# Chart colour palette (Plotly)
COLORS = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "red": "#C44E52",
    "green": "#55A868",
    "yellow": "#CCB974",
    "stars": ["#C44E52", "#DD8452", "#CCB974", "#55A868", "#4C72B0"],
}

# Pain-point theme keywords for topic tagging
PAIN_THEMES = {
    "AccountBan": [
        "ban", "banned", "suspended", "deleted account",
        "account deleted", "shadowban",
    ],
    "Ads": ["ad", "ads", "advertisement", "too many ads", "advertisements"],
    "Crashes": [
        "crash", "crashes", "freeze", "frozen", "lag",
        "bug", "glitch", "not working",
    ],
    "Algorithm": [
        "algorithm", "fyp", "for you page", "shadowban", "reach", "views",
    ],
    "Privacy": ["privacy", "data", "spy", "track", "tracking", "steal"],
    "ContentModeration": [
        "inappropriate", "children", "minor", "nsfw", "kid", "child",
    ],
}

# VADER sentiment thresholds
SENTIMENT_POSITIVE_THRESHOLD = 0.05
SENTIMENT_NEGATIVE_THRESHOLD = -0.05


def get_cutoff_date() -> datetime:
    """Return UTC cutoff datetime for the lookback window."""
    return datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)


def ensure_dirs() -> None:
    """Create project data directories if they do not exist."""
    for path in (DATA_RAW, DATA_PROCESSED, DATA_REPORTS, NOTEBOOKS, DOCS_HTML.parent):
        path.mkdir(parents=True, exist_ok=True)
