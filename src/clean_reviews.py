"""Clean, deduplicate, and enrich scraped TikTok Google Play reviews."""

from __future__ import annotations

import json

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import (
    PROCESSED_CSV,
    PROCESSED_PARQUET,
    RAW_JSON,
    ensure_dirs,
    get_cutoff_date,
)
from src.utils import parse_datetime, sentiment_label

# Re-use a single VADER instance across all reviews for efficiency.
_VADER = SentimentIntensityAnalyzer()


def _score_sentiment(text: str) -> tuple[float, str]:
    """Return VADER compound score and categorical label for review text."""
    if not text:
        return 0.0, "neutral"
    compound = _VADER.polarity_scores(text)["compound"]
    return compound, sentiment_label(compound)


def clean_reviews() -> pd.DataFrame:
    """
    Load raw JSON, clean, deduplicate, and export to Parquet and CSV.

    Cleaning steps:
        - Filter to the 30-day UTC window
        - Deduplicate on reviewId
        - Normalize text and types
        - Add sentiment scores and derived feature columns

    Returns:
        Cleaned DataFrame sorted by review date (newest first).
    """
    ensure_dirs()
    cutoff = get_cutoff_date()

    with RAW_JSON.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_reviews = payload.get("reviews", [])
    if not raw_reviews:
        raise ValueError(
            f"No reviews found in {RAW_JSON}. Run scrape_reviews first."
        )

    df = pd.DataFrame(raw_reviews)

    # Parse timestamps to UTC
    df["at"] = df["at"].apply(parse_datetime)
    if "repliedAt" in df.columns:
        df["repliedAt"] = df["repliedAt"].apply(parse_datetime)

    # Keep only reviews within the lookback window
    df = df[df["at"] >= cutoff].copy()

    # Deduplicate on unique review identifier
    df = df.drop_duplicates(subset="reviewId", keep="first")

    # Normalize text fields
    df["content"] = df["content"].fillna("").astype(str).str.strip()
    reply_col = df.get("replyContent", pd.Series(dtype=str))
    df["replyContent"] = reply_col.fillna("").astype(str).str.strip()

    # Drop rows with no content and no score
    df = df[~((df["content"] == "") & (df["score"].isna()))].copy()

    # Coerce numeric types
    df["score"] = pd.to_numeric(df["score"], errors="coerce").astype("Int64")
    df["thumbsUpCount"] = (
        pd.to_numeric(df.get("thumbsUpCount", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Derived feature columns
    df["review_date"] = df["at"].dt.date
    df["is_negative"] = df["score"] <= 2
    df["has_dev_reply"] = df["replyContent"].str.len() > 0
    df["text_length"] = df["content"].str.len()

    # Sentiment analysis via VADER
    sentiment = df["content"].apply(_score_sentiment)
    df["vader_compound"] = sentiment.apply(lambda x: x[0])
    df["sentiment_label"] = sentiment.apply(lambda x: x[1])

    # Sort newest first
    df = df.sort_values("at", ascending=False).reset_index(drop=True)

    # Export to both Parquet (primary) and CSV (human-readable)
    df.to_parquet(PROCESSED_PARQUET, index=False)
    df.to_csv(PROCESSED_CSV, index=False)

    print(f"Cleaned {len(df):,} reviews → {PROCESSED_PARQUET}")
    return df
