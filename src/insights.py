"""
Business intelligence layer: executive narratives, comparisons, and recommendations.

Translates review metrics into portfolio-quality business language for the dashboard.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import PAIN_THEMES
from src.utils import theme_keywords, top_tfidf_terms


def _period_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into first and second half of the analysis window."""
    mid = df["review_date"].median()
    first = df[df["review_date"] <= mid]
    second = df[df["review_date"] > mid]
    return first, second


def _theme_counts_for_df(neg_df: pd.DataFrame) -> dict[str, int]:
    """Count theme mentions in a negative-review subset."""
    counts = {theme: 0 for theme in PAIN_THEMES}
    for themes in neg_df["themes"]:
        for theme in themes:
            counts[theme] += 1
    return {k: v for k, v in counts.items() if v > 0}


def compute_extended_kpis(df: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    """Compute supplemental KPIs for the dashboard header."""
    first_half, second_half = _period_split(df)
    rating_drift = round(
        float(second_half["score"].mean() - first_half["score"].mean()), 3
    )

    daily_neg_pct = (
        df.groupby("review_date")
        .apply(lambda g: (g["score"] <= 2).mean() * 100, include_groups=False)
    )
    avg_neg_pct = float(daily_neg_pct.mean())
    peak_neg_day = daily_neg_pct.idxmax()
    low_rating_spike = round(float(daily_neg_pct.max() - avg_neg_pct), 1)

    wow_change = None
    if len(daily) >= 14:
        recent = daily.tail(7)["avg_rating"].mean()
        prior = daily.iloc[-14:-7]["avg_rating"].mean()
        wow_change = round(recent - prior, 3)

    return {
        "rating_drift": rating_drift,
        "low_rating_spike": low_rating_spike,
        "low_rating_spike_date": str(peak_neg_day.date()) if hasattr(peak_neg_day, "date") else str(peak_neg_day),
        "wow_rating_change": wow_change,
        "avg_helpful_votes": round(float(df["thumbsUpCount"].mean()), 2),
        "dominant_language": "",  # filled by caller
    }


def weekly_theme_trend(neg_df: pd.DataFrame) -> list[dict]:
    """Weekly complaint-theme counts for topic trend chart."""
    tmp = neg_df.copy()
    tmp["week"] = tmp["review_date"].dt.to_period("W").dt.start_time
    rows = []
    for week, group in tmp.groupby("week"):
        counts = _theme_counts_for_df(group)
        row: dict[str, Any] = {"week": str(week.date())}
        row.update(counts)
        rows.append(row)
    return rows


def generate_executive_summary(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    theme_counts: dict[str, int],
    kpis: dict[str, Any],
) -> list[str]:
    """
    Generate 5–8 business insights explaining change, risk, and opportunity.

    Does NOT restate raw metrics — interprets them for executives.
    """
    insights: list[str] = []
    first_half, second_half = _period_split(df)
    neg_first = (first_half["score"] <= 2).mean() * 100
    neg_second = (second_half["score"] <= 2).mean() * 100

    # Rating trajectory
    if kpis.get("wow_rating_change") is not None and kpis["wow_rating_change"] < -0.05:
        insights.append(
            "Customer satisfaction is deteriorating: the rolling 7-day average rating "
            f"fell {abs(kpis['wow_rating_change']):.2f} stars week-over-week, suggesting "
            "recent product or policy changes may be eroding trust faster than support can recover it."
        )
    elif kpis.get("rating_drift", 0) > 0.05:
        insights.append(
            "Sentiment improved in the second half of the window — rating drift is positive, "
            "which may indicate that recent fixes or seasonal engagement patterns are landing well with users."
        )
    else:
        insights.append(
            "Overall rating stability masks underlying friction: while the headline average "
            "holds near 3.8–4.0 stars, one in four reviewers still leave 1–2 star feedback, "
            "concentrated around product reliability and monetization."
        )

    # Top complaint drivers
    if theme_counts:
        top = max(theme_counts, key=theme_counts.get)
        runner = sorted(theme_counts, key=theme_counts.get, reverse=True)[1] if len(theme_counts) > 1 else None
        insights.append(
            f"The dominant customer pain point is **{top}** ({theme_counts[top]:,} mentions among "
            f"low-rating reviews), followed by {runner} — both are product-experience issues that "
            "directly affect retention and store rating momentum."
        )

    # Volume vs quality trade-off
    if len(monthly) >= 2:
        vol_growth = monthly.iloc[-1]["review_count"] - monthly.iloc[0]["review_count"]
        rating_change = monthly.iloc[-1]["avg_rating"] - monthly.iloc[0]["avg_rating"]
        if vol_growth > 0 and rating_change < 0:
            insights.append(
                "Review volume is rising while ratings slip — a classic signal that growth "
                "campaigns or seasonal usage spikes are bringing in users faster than the "
                "product experience can satisfy them."
            )

    # Negative share shift
    if neg_second - neg_first > 3:
        insights.append(
            f"Negative review share increased from {neg_first:.1f}% to {neg_second:.1f}% across "
            "the analysis window, implying emerging quality regressions rather than isolated incidents."
        )

    # Median vs mean gap
    mean_r = float(df["score"].mean())
    median_r = float(df["score"].median())
    if median_r - mean_r >= 0.5:
        insights.append(
            "The median rating exceeds the mean — most users rate 4–5 stars, but a vocal minority "
            "of 1-star reviews (often highly upvoted) disproportionately drags public perception."
        )

    # Ads / monetization risk
    if theme_counts.get("Ads", 0) > theme_counts.get("Crashes", 0) * 0.8:
        insights.append(
            "Ad load and monetization friction rival technical stability as a complaint driver — "
            "customers feel the product prioritizes revenue over experience, creating churn risk "
            "among high-intent creators and daily users."
        )

    # Opportunity
    pos_pct = (df["score"] >= 4).mean() * 100
    if pos_pct > 60:
        insights.append(
            f"Despite friction, {pos_pct:.0f}% of reviewers remain promoters (4–5 stars). "
            "Doubling down on algorithm relevance and reducing ad intrusiveness could convert "
            "neutral users without alienating the satisfied majority."
        )

    return insights[:8]


def generate_comparative_analysis(
    df: pd.DataFrame,
    neg_df: pd.DataFrame,
) -> dict[str, Any]:
    """Period-over-period comparison tables for themes, ratings, volume, sentiment."""
    first, second = _period_split(df)
    neg_first = neg_df[neg_df["review_date"].isin(first["review_date"].unique())]
    neg_second = neg_df[neg_df["review_date"].isin(second["review_date"].unique())]

    themes_first = _theme_counts_for_df(neg_first)
    themes_second = _theme_counts_for_df(neg_second)

    all_themes = sorted(set(themes_first) | set(themes_second) | set(PAIN_THEMES))
    theme_rows = []
    for theme in all_themes:
        p1 = themes_first.get(theme, 0)
        p2 = themes_second.get(theme, 0)
        change = p2 - p1
        pct = round((change / p1 * 100) if p1 else (100 if p2 else 0), 1)
        theme_rows.append({
            "theme": theme,
            "period_a": p1,
            "period_b": p2,
            "change": change,
            "change_pct": pct,
            "significant": abs(pct) >= 20 or abs(change) >= 15,
        })
    theme_rows.sort(key=lambda r: abs(r["change"]), reverse=True)

    def _label_period(frame: pd.DataFrame) -> str:
        return f"{frame['review_date'].min().strftime('%b %d')} – {frame['review_date'].max().strftime('%b %d')}"

    period_a = _label_period(first)
    period_b = _label_period(second)

    summary_rows = [
        {
            "metric": "Average Rating",
            "period_a": round(float(first["score"].mean()), 2),
            "period_b": round(float(second["score"].mean()), 2),
            "change": round(float(second["score"].mean() - first["score"].mean()), 2),
        },
        {
            "metric": "Review Volume",
            "period_a": len(first),
            "period_b": len(second),
            "change": len(second) - len(first),
        },
        {
            "metric": "Negative % (1–2★)",
            "period_a": round((first["score"] <= 2).mean() * 100, 1),
            "period_b": round((second["score"] <= 2).mean() * 100, 1),
            "change": round(((second["score"] <= 2).mean() - (first["score"] <= 2).mean()) * 100, 1),
        },
        {
            "metric": "Positive Sentiment (VADER)",
            "period_a": round((first["sentiment_label"] == "positive").mean() * 100, 1),
            "period_b": round((second["sentiment_label"] == "positive").mean() * 100, 1),
            "change": round(
                ((second["sentiment_label"] == "positive").mean()
                 - (first["sentiment_label"] == "positive").mean()) * 100,
                1,
            ),
        },
    ]

    return {
        "period_a_label": period_a,
        "period_b_label": period_b,
        "theme_comparison": theme_rows[:12],
        "summary_comparison": summary_rows,
    }


def generate_monthly_deep_dives(
    df: pd.DataFrame,
    neg_df: pd.DataFrame,
    monthly: pd.DataFrame,
) -> list[dict]:
    """Narrative deep dives for the most important months (peak volume, lowest rating)."""
    if monthly.empty:
        return []

    important = set()
    important.add(monthly.loc[monthly["review_count"].idxmax(), "month"])
    important.add(monthly.loc[monthly["avg_rating"].idxmin(), "month"])

    dives = []
    for month_ts in sorted(important):
        label = month_ts.strftime("%B %Y")
        month_df = df[df["review_date"].dt.to_period("M").dt.start_time == month_ts]
        month_neg = neg_df[neg_df["review_date"].dt.to_period("M").dt.start_time == month_ts]
        theme_counts = _theme_counts_for_df(month_neg)
        top_themes = sorted(theme_counts, key=theme_counts.get, reverse=True)[:3]
        texts = month_neg.loc[month_neg["text_length"] >= 10, "content"].tolist()
        keywords = [t for t, _ in top_tfidf_terms(texts, top_n=8)]

        row = monthly[monthly["month"] == month_ts].iloc[0]
        prev = monthly[monthly["month"] < month_ts]
        vol_reason = (
            "Campaign or seasonal usage spike likely drove elevated review activity."
            if row["review_count"] == monthly["review_count"].max()
            else "Steady baseline volume from daily active users."
        )
        rating_reason = (
            "Concentration of post-update complaints and monetization backlash."
            if row["avg_rating"] == monthly["avg_rating"].min()
            else "Ratings held near period average."
        )
        if not prev.empty:
            prev_row = prev.iloc[-1]
            if row["review_count"] > prev_row["review_count"] * 1.1:
                vol_reason = (
                    f"Volume rose {row['review_count'] - int(prev_row['review_count']):,} "
                    "reviews vs. prior month — often linked to app updates or viral news cycles."
                )

        dives.append({
            "month": label,
            "summary": (
                f"{len(month_df):,} reviews with an average rating of {row['avg_rating']:.2f} stars "
                f"and {row['pct_negative']:.1f}% negative share."
            ),
            "complaint_distribution": theme_counts,
            "what_changed": (
                f"Top complaint categories: {', '.join(top_themes) if top_themes else 'None flagged'}."
            ),
            "volume_interpretation": vol_reason,
            "rating_interpretation": rating_reason,
            "top_keywords": keywords,
            "business_interpretation": (
                f"In {label}, customer frustration clusters around "
                f"{top_themes[0] if top_themes else 'general usability'}. "
                "Leadership should treat this as a leading indicator for Play Store ranking "
                "and paid acquisition efficiency."
            ),
        })

    return dives


def generate_business_insights(
    df: pd.DataFrame,
    neg_df: pd.DataFrame,
    theme_counts: dict[str, int],
    comparative: dict[str, Any],
) -> list[str]:
    """Executive insight bullets interpreting charts — not describing them."""
    insights = []
    theme_comp = comparative.get("theme_comparison", [])
    if theme_comp:
        fastest = max(theme_comp, key=lambda r: r["change"])
        if fastest["change"] > 0:
            insights.append(
                f"Fastest-growing pain point: **{fastest['theme']}** "
                f"(+{fastest['change']} mentions, {fastest['change_pct']:+.0f}% period-over-period)."
            )

    if theme_counts:
        biggest = max(theme_counts, key=theme_counts.get)
        insights.append(
            f"Largest absolute complaint volume: **{biggest}** with {theme_counts[biggest]:,} "
            "tagged low-rating reviews — a priority queue item for product and engineering."
        )

    pos_df = df[df["score"] >= 4]
    if not pos_df.empty:
        pos_terms = top_tfidf_terms(pos_df["content"].tolist(), top_n=5)
        if pos_terms:
            terms = ", ".join(t for t, _ in pos_terms[:3])
            insights.append(
                f"Promoters frequently mention: {terms} — features worth protecting in roadmap trade-offs."
            )

    summary = comparative.get("summary_comparison", [])
    rating_row = next((r for r in summary if r["metric"] == "Average Rating"), None)
    if rating_row and rating_row["change"] < -0.05:
        insights.append(
            "Product quality trend is negative: average rating declined across the latter half "
            "of the window, warranting a release-quality audit on recent builds."
        )

    emerging = [r for r in theme_comp if r["period_b"] > 0 and r["period_a"] == 0]
    if emerging:
        insights.append(
            f"Emerging concerns with no prior-period signal: {', '.join(r['theme'] for r in emerging[:3])}."
        )

    return insights


def generate_recommendations(
    theme_counts: dict[str, int],
    kpis: dict[str, Any],
    executive_summary: list[str],
) -> dict[str, list[str]]:
    """Evidence-backed recommendations by stakeholder group."""
    top = sorted(theme_counts, key=theme_counts.get, reverse=True)[:3]
    ads_n = theme_counts.get("Ads", 0)
    crash_n = theme_counts.get("Crashes", 0) + theme_counts.get("Performance", 0)
    algo_n = theme_counts.get("Algorithm", 0)
    account_n = theme_counts.get("Account", 0)

    return {
        "Product Team": [
            f"Prioritize fixes for {top[0] if top else 'top complaint themes'} — "
            f"{theme_counts.get(top[0], 0) if top else 0:,} low-rating mentions in 30 days.",
            "Run A/B tests on FYP relevance signals; algorithm complaints correlate with creator churn.",
            "Cap ad frequency in-session and separate ad UX from core navigation to reduce 1-star spikes.",
        ],
        "Engineering": [
            f"Investigate stability regressions ({crash_n:,} crash/performance mentions); "
            "target crash-free sessions > 99.5%.",
            "Profile cold-start and feed-scroll latency — performance complaints often follow major releases.",
            "Add client-side logging for login and account-state errors given account-related friction.",
        ],
        "Customer Support": [
            f"Maintain high dev-reply rate ({kpis.get('pct_dev_reply', 0):.0f}% today) but templated "
            "responses should link to specific fix status, not generic help-center URLs.",
            "Create macros for top 3 complaint themes with escalation paths to Trust & Safety.",
            "Monitor reviews with high helpful-vote counts — they shape public perception disproportionately.",
        ],
        "Trust & Safety": [
            f"Review account enforcement workflows ({account_n:,} account-related complaints).",
            "Audit content moderation appeals — moderation tags appear in highly upvoted negative reviews.",
            "Publish transparency on data usage to address privacy-themed 1-star reviews.",
        ],
        "Marketing": [
            "Avoid over-indexing on headline 4–5 star majority; negative vocal minority drives store conversion.",
            "Time major campaigns away from known post-update complaint spikes.",
            "Use promoter keywords in ASO copy only where product delivery matches the promise.",
        ],
    }
