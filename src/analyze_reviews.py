"""Analyze cleaned TikTok reviews: metrics, sentiment, and topic extraction."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import APP_NAME, METRICS_JSON, PAIN_THEMES, PROCESSED_PARQUET, ensure_dirs
from src.utils import (
    representative_quotes,
    save_json,
    tag_themes,
    top_tfidf_terms,
)


def _daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily review count and average rating."""
    daily = (
        df.groupby("review_date")
        .agg(review_count=("reviewId", "count"), avg_rating=("score", "mean"))
        .reset_index()
        .sort_values("review_date")
    )
    daily["rolling_7d_rating"] = daily["avg_rating"].rolling(7, min_periods=1).mean()
    return daily


def _star_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage breakdown by star rating."""
    star_daily = (
        df.groupby(["review_date", "score"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    for star in range(1, 6):
        if star not in star_daily.columns:
            star_daily[star] = 0
    star_daily = star_daily[[1, 2, 3, 4, 5]]
    return star_daily.div(star_daily.sum(axis=1), axis=0) * 100


def _theme_counts(neg_df: pd.DataFrame) -> dict[str, int]:
    """Count pain-point theme occurrences in negative reviews."""
    counts: dict[str, int] = {theme: 0 for theme in PAIN_THEMES}
    for themes in neg_df["themes"]:
        for theme in themes:
            counts[theme] += 1
    return {
        theme: count
        for theme, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if count > 0
    }


def _build_insights(
    total: int,
    mean_rating: float,
    pct_negative: float,
    daily: pd.DataFrame,
    theme_counts: dict[str, int],
    theme_share: dict[str, float],
    wow_change: float | None,
) -> list[str]:
    """Generate human-readable insight bullets for the report."""
    peak = daily.loc[daily["review_count"].idxmax()]
    lowest = daily.loc[daily["avg_rating"].idxmin()]

    insights = [
        (
            f"{total:,} reviews collected over the last 30 days with an "
            f"average rating of {mean_rating:.2f} stars."
        ),
        f"{pct_negative:.1f}% of reviews are negative (1–2 stars).",
        (
            f"Peak review volume on {peak['review_date']} "
            f"({int(peak['review_count']):,} reviews)."
        ),
        (
            f"Lowest average rating on {lowest['review_date']} "
            f"({lowest['avg_rating']:.2f} stars)."
        ),
    ]

    top_themes = list(theme_counts.keys())[:3]
    if top_themes:
        theme_str = ", ".join(
            f"{theme} ({theme_share[theme]}% of negative)" for theme in top_themes
        )
        insights.append(
            f"Top complaint themes among negative reviews: {theme_str}."
        )

    if wow_change is not None:
        direction = "improved" if wow_change > 0 else "declined"
        insights.append(
            f"Average rating {direction} by {abs(wow_change):.2f} stars week-over-week."
        )

    return insights


def analyze_reviews() -> dict:
    """
    Run sentiment and topic analysis; write metrics JSON for visualization.

    Computes:
        - Volume and rating trend metrics
        - Sentiment distribution (VADER labels)
        - Pain-point theme counts and TF-IDF keywords
        - Representative negative-review quotes

    Returns:
        Metrics dictionary saved to ``data/reports/metrics.json``.
    """
    ensure_dirs()

    df = pd.read_parquet(PROCESSED_PARQUET)
    df["at"] = pd.to_datetime(df["at"], utc=True)
    df["review_date"] = pd.to_datetime(df["review_date"])

    neg_df = df[df["is_negative"]].copy()
    neg_df["themes"] = neg_df["content"].apply(tag_themes)

    daily = _daily_aggregates(df)
    star_pct = _star_mix(df)
    theme_counts = _theme_counts(neg_df)

    # TF-IDF keyword extraction on negative reviews with sufficient text
    neg_texts = neg_df.loc[neg_df["text_length"] >= 10, "content"].tolist()
    tfidf_terms = top_tfidf_terms(neg_texts)

    # Top keywords across all reviews for broader topic context
    all_texts = df.loc[df["text_length"] >= 10, "content"].tolist()
    top_keywords = top_tfidf_terms(all_texts, top_n=20)

    version_col = (
        "appVersion" if "appVersion" in neg_df.columns else "reviewCreatedVersion"
    )
    top_versions = (
        neg_df[version_col].fillna("Unknown").value_counts().head(5).to_dict()
    )

    total = len(df)
    mean_rating = float(df["score"].mean())
    pct_negative = float(df["is_negative"].mean() * 100)
    pct_dev_reply = float(df["has_dev_reply"].mean() * 100)

    peak_day_row = daily.loc[daily["review_count"].idxmax()]
    lowest_day_row = daily.loc[daily["avg_rating"].idxmin()]

    wow_change: float | None = None
    if len(daily) >= 14:
        recent = daily.tail(7)["avg_rating"].mean()
        prior = daily.iloc[-14:-7]["avg_rating"].mean()
        wow_change = round(recent - prior, 3)

    top_themes = list(theme_counts.keys())[:3]
    theme_share = {
        theme: round(theme_counts[theme] / max(len(neg_df), 1) * 100, 1)
        for theme in top_themes
    }

    quotes = representative_quotes(neg_df, list(theme_counts.keys())[:5])

    # Sentiment distribution from VADER labels
    sentiment_dist = (
        df["sentiment_label"].value_counts(normalize=True).mul(100).round(1).to_dict()
    )

    insights = _build_insights(
        total, mean_rating, pct_negative, daily,
        theme_counts, theme_share, wow_change,
    )

    metrics = {
        "app_name": APP_NAME,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {
            "start": str(df["review_date"].min().date()),
            "end": str(df["review_date"].max().date()),
        },
        "total_reviews": total,
        "mean_rating": round(mean_rating, 3),
        "pct_negative": round(pct_negative, 1),
        "pct_dev_reply": round(pct_dev_reply, 1),
        "peak_day": {
            "date": str(peak_day_row["review_date"].date()),
            "count": int(peak_day_row["review_count"]),
        },
        "lowest_rating_day": {
            "date": str(lowest_day_row["review_date"].date()),
            "avg_rating": round(float(lowest_day_row["avg_rating"]), 2),
        },
        "wow_rating_change": wow_change,
        "star_distribution": {
            str(k): int(v)
            for k, v in df["score"].value_counts().sort_index().items()
        },
        "sentiment_distribution": sentiment_dist,
        "theme_counts": theme_counts,
        "theme_share_of_negative_pct": theme_share,
        "top_negative_versions": top_versions,
        "tfidf_terms": [
            {"term": term, "score": round(float(score), 4)}
            for term, score in tfidf_terms
        ],
        "top_keywords": [
            {"term": term, "score": round(float(score), 4)}
            for term, score in top_keywords
        ],
        "representative_quotes": quotes,
        "insights": insights,
        # Serialized chart data for Plotly (strings only — JSON-safe)
        "_daily": [
            {
                "review_date": str(row["review_date"].date()),
                "review_count": int(row["review_count"]),
                "avg_rating": float(row["avg_rating"]),
                "rolling_7d_rating": float(row["rolling_7d_rating"]),
            }
            for _, row in daily.iterrows()
        ],
        "_star_pct_columns": [int(c) for c in star_pct.columns],
        "_star_pct_index": [str(d.date()) for d in star_pct.index],
        "_star_pct_values": star_pct.values.tolist(),
    }

    save_json(METRICS_JSON, metrics)
    print(f"Analysis complete → {METRICS_JSON}")
    return metrics
