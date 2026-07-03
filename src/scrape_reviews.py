"""Scrape TikTok Google Play reviews from the last 30 days."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from google_play_scraper import Sort, reviews

from config import (
    APP_ID,
    APP_NAME,
    COUNTRY,
    LANG,
    MAX_RETRIES,
    RAW_JSON,
    REVIEWS_PER_PAGE,
    SLEEP_SECONDS,
    ensure_dirs,
    get_cutoff_date,
)
from src.utils import serialize_review


def _fetch_batch(
    token: object | None,
    count: int,
) -> tuple[list[dict], object | None]:
    """
    Fetch one page of reviews with exponential backoff retries.

    Args:
        token: Pagination continuation token from the previous batch.
        count: Number of reviews to request (max 200 per Google Play limit).

    Returns:
        Tuple of (review list, next continuation token).
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return reviews(
                APP_ID,
                lang=LANG,
                country=COUNTRY,
                sort=Sort.NEWEST,
                count=count,
                continuation_token=token,
            )
        except Exception as exc:  # noqa: BLE001 — retry on transient network errors
            last_error = exc
            wait = SLEEP_SECONDS * (2 ** (attempt - 1))
            print(
                f"  Retry {attempt}/{MAX_RETRIES} after error: {exc} "
                f"(waiting {wait:.1f}s)"
            )
            time.sleep(wait)
    raise RuntimeError(
        f"Failed to fetch reviews after {MAX_RETRIES} attempts"
    ) from last_error


def scrape_reviews() -> dict:
    """
    Paginate newest reviews until the 30-day cutoff is reached.

    Reviews are saved to ``data/raw/reviews_raw.json`` with scrape metadata.

    Returns:
        Metadata dict with scrape statistics.
    """
    ensure_dirs()
    cutoff = get_cutoff_date()
    print(f"Scraping {APP_NAME} ({APP_ID}) reviews since {cutoff.date()} UTC...")

    all_reviews: list[dict] = []
    token: object | None = None
    page = 0

    while True:
        page += 1
        batch, token = _fetch_batch(token, REVIEWS_PER_PAGE)
        if not batch:
            print("  No more reviews returned.")
            break

        all_reviews.extend(batch)
        oldest = batch[-1]["at"]
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)

        print(
            f"  Page {page}: +{len(batch)} reviews "
            f"(total {len(all_reviews)}, oldest in batch {oldest.date()})"
        )

        if oldest < cutoff:
            print("  Reached 30-day cutoff.")
            break

        if token is None:
            print("  No continuation token — end of available reviews.")
            break

        time.sleep(SLEEP_SECONDS)

    payload = {
        "metadata": {
            "app_id": APP_ID,
            "app_name": APP_NAME,
            "lang": LANG,
            "country": COUNTRY,
            "cutoff_date": cutoff.isoformat(),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "total_fetched": len(all_reviews),
            "pages_fetched": page,
        },
        "reviews": [serialize_review(review) for review in all_reviews],
    }

    with RAW_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_reviews)} raw reviews to {RAW_JSON}")
    return payload["metadata"]
