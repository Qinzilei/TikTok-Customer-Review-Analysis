"""Shared utilities for the TikTok review analysis pipeline."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    SENTIMENT_NEGATIVE_THRESHOLD,
    SENTIMENT_POSITIVE_THRESHOLD,
    PAIN_THEMES,
)


def serialize_review(review: dict) -> dict:
    """Convert a review dict to a JSON-serializable form."""
    out: dict[str, Any] = {}
    for key, value in review.items():
        if isinstance(value, datetime):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


def parse_datetime(value: Any) -> datetime | None:
    """Parse ISO or datetime values into timezone-aware UTC datetimes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def sentiment_label(compound: float) -> str:
    """
    Map VADER compound score to a categorical sentiment label.

    Args:
        compound: VADER compound score in [-1, 1].

    Returns:
        One of 'positive', 'neutral', or 'negative'.
    """
    if compound >= SENTIMENT_POSITIVE_THRESHOLD:
        return "positive"
    if compound <= SENTIMENT_NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def tag_themes(text: str) -> list[str]:
    """
    Return pain-point theme labels matched in review text.

    Uses whole-word keyword matching against predefined theme dictionaries.
    """
    if not text or len(text) < 10:
        return []
    lower = text.lower()
    matched: list[str] = []
    for theme, keywords in PAIN_THEMES.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", lower):
                matched.append(theme)
                break
    return matched


def top_tfidf_terms(
    texts: list[str],
    top_n: int = 15,
) -> list[tuple[str, float]]:
    """
    Extract top TF-IDF terms from a corpus of review texts.

    Args:
        texts: List of review strings.
        top_n: Number of top terms to return.

    Returns:
        List of (term, score) tuples sorted by score descending.
    """
    if len(texts) < 5:
        return []
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    scores = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def representative_quotes(
    neg_df: pd.DataFrame,
    themes: list[str],
    max_chars: int = 300,
) -> dict[str, dict]:
    """
    Select the most helpful review quote per complaint theme.

    Picks the review with the highest thumbsUpCount for each theme.
    """
    quotes: dict[str, dict] = {}
    for theme in themes:
        mask = neg_df["themes"].apply(lambda tags: theme in tags)
        subset = neg_df[mask].sort_values("thumbsUpCount", ascending=False)
        if subset.empty:
            continue
        row = subset.iloc[0]
        quotes[theme] = {
            "content": str(row["content"])[:max_chars],
            "score": int(row["score"]),
            "thumbsUpCount": int(row["thumbsUpCount"]),
            "at": (
                row["at"].isoformat()
                if hasattr(row["at"], "isoformat")
                else str(row["at"])
            ),
        }
    return quotes


def save_json(path, payload: dict) -> None:
    """Write a dict to JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def escape_html(value: Any) -> str:
    """Escape a value for safe HTML embedding."""
    return html.escape(str(value))
