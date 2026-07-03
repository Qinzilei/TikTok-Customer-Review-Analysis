"""Analyze cleaned TikTok reviews: metrics, sentiment, and topic extraction."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import APP_NAME, METRICS_JSON, PAIN_THEMES, PROCESSED_PARQUET, ensure_dirs
from src.utils import (
    detect_top_languages,
    representative_quotes,
    save_json,
    tag_themes,
    top_tfidf_terms,
    word_frequencies,
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
    daily["rolling_7d_volume"] = (
        daily["review_count"].rolling(7, min_periods=1).mean()
    )
    return daily


def _weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly review volume and average rating."""
    tmp = df.copy()
    tmp["week"] = tmp["review_date"].dt.to_period("W").dt.start_time
    return (
        tmp.groupby("week")
        .agg(review_count=("reviewId", "count"), avg_rating=("score", "mean"))
        .reset_index()
        .sort_values("week")
    )


def _monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly review volume, rating, and sentiment."""
    tmp = df.copy()
    tmp["month"] = tmp["review_date"].dt.to_period("M").dt.start_time
    monthly = (
        tmp.groupby("month")
        .agg(
            review_count=("reviewId", "count"),
            avg_rating=("score", "mean"),
            median_rating=("score", "median"),
            pct_positive=("score", lambda s: (s >= 4).mean() * 100),
            pct_negative=("score", lambda s: (s <= 2).mean() * 100),
            avg_sentiment=("vader_compound", "mean"),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly["growth_pct"] = monthly["review_count"].pct_change() * 100
    return monthly


def _monthly_star_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Star-rating counts grouped by month."""
    tmp = df.copy()
    tmp["month"] = tmp["review_date"].dt.to_period("M").dt.start_time
    dist = (
        tmp.groupby(["month", "score"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    for star in range(1, 6):
        if star not in dist.columns:
            dist[star] = 0
    return dist[[1, 2, 3, 4, 5]]


def _monthly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Sentiment label proportions by month."""
    tmp = df.copy()
    tmp["month"] = tmp["review_date"].dt.to_period("M").dt.start_time
    return (
        tmp.groupby(["month", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .div(tmp.groupby("month").size(), axis=0)
        .mul(100)
        .reset_index()
        .sort_values("month")
    )


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
            f"Peak review volume on {peak['review_date'].date()} "
            f"({int(peak['review_count']):,} reviews)."
        ),
        (
            f"Lowest average rating on {lowest['review_date'].date()} "
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
    weekly = _weekly_aggregates(df)
    monthly = _monthly_aggregates(df)
    monthly_stars = _monthly_star_distribution(df)
    monthly_sentiment = _monthly_sentiment(df)
    theme_counts = _theme_counts(neg_df)

    neg_texts = neg_df.loc[neg_df["text_length"] >= 10, "content"].tolist()
    all_texts = df.loc[df["text_length"] >= 10, "content"].tolist()
    tfidf_terms = top_tfidf_terms(neg_texts)
    top_keywords = top_tfidf_terms(all_texts, top_n=20)
    word_freq = word_frequencies(all_texts, top_n=80)
    top_languages = detect_top_languages(all_texts)

    version_col = (
        "appVersion" if "appVersion" in neg_df.columns else "reviewCreatedVersion"
    )
    top_versions = (
        neg_df[version_col].fillna("Unknown").value_counts().head(5).to_dict()
    )

    total = len(df)
    num_days = max((df["review_date"].max() - df["review_date"].min()).days + 1, 1)
    mean_rating = float(df["score"].mean())
    median_rating = float(df["score"].median())
    pct_positive = float((df["score"] >= 4).mean() * 100)
    pct_negative = float(df["is_negative"].mean() * 100)
    pct_dev_reply = float(df["has_dev_reply"].mean() * 100)
    reviews_per_day = round(total / num_days, 1)

    peak_month_row = monthly.loc[monthly["review_count"].idxmax()]
    lowest_month_row = monthly.loc[monthly["avg_rating"].idxmin()]

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
        "median_rating": round(median_rating, 2),
        "pct_positive": round(pct_positive, 1),
        "pct_negative": round(pct_negative, 1),
        "reviews_per_day": reviews_per_day,
        "pct_dev_reply": round(pct_dev_reply, 1),
        "peak_month": {
            "label": peak_month_row["month"].strftime("%b %Y"),
            "count": int(peak_month_row["review_count"]),
        },
        "lowest_rating_month": {
            "label": lowest_month_row["month"].strftime("%b %Y"),
            "avg_rating": round(float(lowest_month_row["avg_rating"]), 2),
        },
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
        "word_frequencies": [{"word": w, "count": c} for w, c in word_freq],
        "top_languages": [{"language": lang, "count": c} for lang, c in top_languages],
        "representative_quotes": quotes,
        "insights": insights,
        "_daily": [
            {
                "review_date": str(row["review_date"].date()),
                "review_count": int(row["review_count"]),
                "avg_rating": float(row["avg_rating"]),
                "rolling_7d_rating": float(row["rolling_7d_rating"]),
                "rolling_7d_volume": float(row["rolling_7d_volume"]),
            }
            for _, row in daily.iterrows()
        ],
        "_weekly": [
            {
                "week": str(row["week"].date()),
                "review_count": int(row["review_count"]),
                "avg_rating": float(row["avg_rating"]),
            }
            for _, row in weekly.iterrows()
        ],
        "_monthly": [
            {
                "month": row["month"].strftime("%b %Y"),
                "month_iso": str(row["month"].date()),
                "review_count": int(row["review_count"]),
                "avg_rating": float(row["avg_rating"]),
                "growth_pct": (
                    None if pd.isna(row["growth_pct"]) else round(float(row["growth_pct"]), 1)
                ),
            }
            for _, row in monthly.iterrows()
        ],
        "_monthly_stars": {
            "months": [d.strftime("%b %Y") for d in monthly_stars.index],
            "stars": {
                str(star): monthly_stars[star].tolist()
                for star in monthly_stars.columns
            },
        },
        "_monthly_sentiment": [
            {
                "month": row["month"].strftime("%b %Y"),
                "positive": round(float(row.get("positive", 0)), 1),
                "neutral": round(float(row.get("neutral", 0)), 1),
                "negative": round(float(row.get("negative", 0)), 1),
            }
            for _, row in monthly_sentiment.iterrows()
        ],
    }

    save_json(METRICS_JSON, metrics)
    print(f"Analysis complete → {METRICS_JSON}")
    return metrics
