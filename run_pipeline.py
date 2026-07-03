"""Run the full TikTok Google Play review analysis pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path so config and src are importable.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import REPORT_HTML, ensure_dirs  # noqa: E402
from src.analyze_reviews import analyze_reviews  # noqa: E402
from src.clean_reviews import clean_reviews  # noqa: E402
from src.scrape_reviews import scrape_reviews  # noqa: E402
from src.visualize import generate_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pipeline control."""
    parser = argparse.ArgumentParser(
        description="TikTok Google Play review analysis pipeline",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping and use existing raw data",
    )
    return parser.parse_args()


def main() -> None:
    """Execute scrape → clean → analyze → visualize pipeline."""
    args = parse_args()
    ensure_dirs()

    print("=" * 60)
    print("TikTok Google Play Review Analysis Pipeline")
    print("=" * 60)

    if args.skip_scrape:
        print("\n[1/4] Scraping reviews... SKIPPED (--skip-scrape)")
    else:
        print("\n[1/4] Scraping reviews...")
        meta = scrape_reviews()
        print(
            f"  → Fetched {meta['total_fetched']:,} reviews "
            f"in {meta['pages_fetched']} pages"
        )

    print("\n[2/4] Cleaning data...")
    df = clean_reviews()
    print(f"  → {len(df):,} reviews after cleaning")

    print("\n[3/4] Analyzing (sentiment + topics)...")
    metrics = analyze_reviews()
    print(
        f"  → Mean rating: {metrics['mean_rating']:.2f}, "
        f"negative: {metrics['pct_negative']:.1f}%"
    )

    print("\n[4/4] Generating interactive HTML report...")
    report = generate_report(metrics)
    print(f"  → {report}")

    print(f"\nDone. Open {REPORT_HTML} in your browser to view the report.")


if __name__ == "__main__":
    main()
