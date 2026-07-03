"""Generate executive analytics dashboard with Plotly charts and dark-theme HTML."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from config import APP_NAME, APP_URL, DOCS_HTML, METRICS_JSON, REPORT_HTML, ensure_dirs
from src.dashboard import (
    DASHBOARD_CSS,
    render_business_insights,
    render_charts_section,
    render_comparative,
    render_customer_voice,
    render_executive_summary,
    render_kpis,
    render_monthly_deep_dives,
    render_nav,
    render_recommendations,
    render_word_clouds,
)
from src.utils import escape_html

# Dashboard colour palette (Datadog / Bloomberg-inspired dark theme)
ACCENT = {
    "cyan": "#22d3ee",
    "indigo": "#6366f1",
    "violet": "#a78bfa",
    "emerald": "#34d399",
    "amber": "#fbbf24",
    "rose": "#fb7185",
    "slate": "#94a3b8",
}
CHART_COLORS = ["#6366f1", "#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#fb7185"]
STAR_COLORS = ["#fb7185", "#fbbf24", "#a78bfa", "#34d399", "#6366f1"]

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _legend() -> dict:
    """Shared legend styling for dashboard charts."""
    return dict(
        bgcolor="rgba(15,23,42,0.6)",
        bordercolor="rgba(148,163,184,0.15)",
        font=dict(color="#cbd5e1"),
    )


def _dark_layout(title: str, height: int = 360, **kwargs: Any) -> dict:
    """Return consistent dark-theme Plotly layout overrides."""
    layout = dict(
        title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter, system-ui, sans-serif", size=12),
        height=height,
        margin=dict(l=48, r=24, t=48, b=48),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.12)",
            zerolinecolor="rgba(148,163,184,0.12)",
            tickfont=dict(color="#64748b"),
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.12)",
            zerolinecolor="rgba(148,163,184,0.12)",
            tickfont=dict(color="#64748b"),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            font=dict(color="#f1f5f9"),
        ),
    )
    layout.update(kwargs)
    return layout


def _chart_div(fig: go.Figure, div_id: str) -> str:
    """Render a Plotly figure as an embeddable HTML div."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config=PLOTLY_CONFIG,
    )


def _fig_weekly_volume(weekly: pd.DataFrame) -> go.Figure:
    """Weekly review volume bar chart."""
    fig = go.Figure(
        go.Bar(
            x=weekly["week"],
            y=weekly["review_count"],
            marker=dict(
                color=weekly["review_count"],
                colorscale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#22d3ee"]],
                line=dict(width=0),
            ),
            hovertemplate="Week of %{x}<br>Reviews: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(**_dark_layout("Weekly Review Volume", height=340))
    fig.update_xaxes(title="Week")
    fig.update_yaxes(title="Reviews")
    return fig


def _fig_daily_trend(daily: pd.DataFrame) -> go.Figure:
    """Daily review count with 7-day moving average."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["review_date"],
            y=daily["review_count"],
            mode="lines",
            name="Daily",
            line=dict(color="rgba(99,102,241,0.35)", width=1.5),
            hovertemplate="%{x}<br>%{y:,} reviews<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["review_date"],
            y=daily["rolling_7d_volume"],
            mode="lines",
            name="7-day MA",
            line=dict(color=ACCENT["cyan"], width=2.5),
            hovertemplate="%{x}<br>MA: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_dark_layout("Daily Review Trend + Moving Average", height=340),
        legend={**_legend(), "orientation": "h", "y": 1.12, "x": 0},
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Reviews")
    return fig


def _fig_monthly_rating_dist(monthly_stars: dict) -> go.Figure:
    """Grouped bar chart of star distribution by month."""
    months = monthly_stars["months"]
    fig = go.Figure()
    for idx, star in enumerate(["1", "2", "3", "4", "5"]):
        fig.add_trace(
            go.Bar(
                name=f"{star} star",
                x=months,
                y=monthly_stars["stars"][star],
                marker_color=STAR_COLORS[idx],
                hovertemplate=f"{star} star: %{{y:,}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_dark_layout("Monthly Rating Distribution", height=360),
        barmode="group",
        legend={**_legend(), "orientation": "h", "y": 1.14, "x": 0},
    )
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Review Count")
    return fig


def _fig_monthly_growth(monthly: pd.DataFrame) -> go.Figure:
    """Monthly review count and growth rate trajectory."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["review_count"],
            name="Volume",
            marker=dict(color=ACCENT["indigo"], opacity=0.85),
            yaxis="y",
            hovertemplate="%{x}<br>%{y:,} reviews<extra></extra>",
        )
    )
    growth = monthly["growth_pct"].fillna(0)
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=growth,
            name="Growth %",
            mode="lines+markers",
            line=dict(color=ACCENT["amber"], width=2.5),
            marker=dict(size=7),
            yaxis="y2",
            hovertemplate="%{x}<br>Growth: %{y:.1f}%<extra></extra>",
        )
    )
    layout = _dark_layout("Monthly Growth Trajectory", height=360)
    layout.pop("yaxis", None)
    fig.update_layout(
        **layout,
        yaxis=dict(title="Reviews", gridcolor="rgba(148,163,184,0.12)"),
        yaxis2=dict(
            title="MoM Growth %",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color=ACCENT["amber"]),
        ),
        legend={**_legend(), "orientation": "h", "y": 1.14, "x": 0},
    )
    return fig


def _fig_top_languages(languages: list[dict]) -> go.Figure:
    """Horizontal bar chart of detected review languages."""
    if not languages:
        return go.Figure().update_layout(**_dark_layout("Top Languages", height=320))
    langs = [item["language"] for item in languages][::-1]
    counts = [item["count"] for item in languages][::-1]
    fig = go.Figure(
        go.Bar(
            x=counts,
            y=langs,
            orientation="h",
            marker=dict(
                color=counts,
                colorscale=[[0, "#312e81"], [1, "#22d3ee"]],
                line=dict(width=0),
            ),
            hovertemplate="%{y}: %{x:,} reviews<extra></extra>",
        )
    )
    fig.update_layout(**_dark_layout("Top Languages", height=max(300, len(langs) * 36)))
    fig.update_xaxes(title="Reviews (sampled)")
    return fig


def _fig_word_cloud(word_freq: list[dict], title: str = "Word Cloud") -> go.Figure:
    """Interactive word cloud using spiral-placed Plotly text scatter."""
    if not word_freq:
        return go.Figure().update_layout(**_dark_layout(title, height=360))

    words = [w["word"] for w in word_freq[:60]]
    counts = [w["count"] for w in word_freq[:60]]
    max_count = max(counts) if counts else 1

    xs, ys, sizes, texts, colors = [], [], [], [], []
    for i, (word, count) in enumerate(zip(words, counts)):
        angle = i * 0.72
        radius = 0.35 * (i ** 0.52)
        xs.append(radius * math.cos(angle))
        ys.append(radius * math.sin(angle))
        sizes.append(11 + (count / max_count) * 28)
        texts.append(word)
        colors.append(CHART_COLORS[i % len(CHART_COLORS)])

    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="text",
            text=texts,
            textfont=dict(size=sizes, color=colors),
            hovertext=[f"{w}: {c:,}" for w, c in zip(words, counts)],
            hoverinfo="text",
        )
    )
    layout = _dark_layout(title, height=360)
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(
        **layout,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def _fig_topic_trend(topic_trend: list[dict]) -> go.Figure:
    """Multi-line chart of weekly complaint-theme trends."""
    if not topic_trend:
        return go.Figure().update_layout(**_dark_layout("Topic Trend", height=360))

    df = pd.DataFrame(topic_trend)
    df["week"] = pd.to_datetime(df["week"])
    theme_cols = [c for c in df.columns if c != "week"]

    fig = go.Figure()
    for idx, theme in enumerate(theme_cols):
        if df[theme].sum() == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["week"],
                y=df[theme],
                mode="lines+markers",
                name=theme,
                line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2),
                marker=dict(size=5),
                hovertemplate=f"{theme}: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_dark_layout("Complaint Topic Trend (Weekly)", height=360),
        legend={**_legend(), "orientation": "h", "y": 1.14, "x": 0},
    )
    fig.update_xaxes(title="Week")
    fig.update_yaxes(title="Mentions")
    return fig


def _fig_tfidf(tfidf_terms: list[dict]) -> go.Figure:
    """TF-IDF keyword bar chart for negative reviews."""
    if not tfidf_terms:
        return go.Figure().update_layout(**_dark_layout("TF-IDF Keywords", height=360))
    terms = [t["term"] for t in tfidf_terms][::-1]
    scores = [t["score"] for t in tfidf_terms][::-1]
    fig = go.Figure(
        go.Bar(
            x=scores,
            y=terms,
            orientation="h",
            marker=dict(color=ACCENT["violet"], opacity=0.9),
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(**_dark_layout("TF-IDF Keywords (Low-Rating Reviews)", height=420))
    fig.update_xaxes(title="TF-IDF Score")
    return fig


def _fig_pain_points(theme_counts: dict[str, int]) -> go.Figure:
    """Pain-point theme analysis for low-rating reviews."""
    if not theme_counts:
        return go.Figure().update_layout(**_dark_layout("Pain-Point Analysis", height=340))
    themes = list(theme_counts.keys())[::-1]
    counts = [theme_counts[t] for t in themes][::-1]
    fig = go.Figure(
        go.Bar(
            x=counts,
            y=themes,
            orientation="h",
            marker=dict(
                color=counts,
                colorscale=[[0, "#7f1d1d"], [1, "#fb7185"]],
                line=dict(width=0),
            ),
            hovertemplate="%{y}: %{x:,} reviews<extra></extra>",
        )
    )
    fig.update_layout(**_dark_layout("Pain-Point Analysis (1–2 Star Reviews)", height=360))
    fig.update_xaxes(title="Mentions")
    return fig


def _fig_monthly_sentiment(sentiment_rows: list[dict]) -> go.Figure:
    """Stacked area chart of monthly sentiment proportions."""
    if not sentiment_rows:
        return go.Figure().update_layout(**_dark_layout("Monthly Sentiment Trend", height=360))
    months = [r["month"] for r in sentiment_rows]
    fig = go.Figure()
    for label, color in [
        ("positive", ACCENT["emerald"]),
        ("neutral", ACCENT["slate"]),
        ("negative", ACCENT["rose"]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=months,
                y=[r[label] for r in sentiment_rows],
                mode="lines",
                stackgroup="one",
                name=label.capitalize(),
                line=dict(width=0.8, color=color),
                fillcolor=color,
                hovertemplate=f"{label}: %{{y:.1f}}%<extra></extra>",
            )
        )
    layout = _dark_layout("Monthly Sentiment Trend", height=360)
    layout.pop("yaxis", None)
    fig.update_layout(
        **layout,
        yaxis=dict(range=[0, 100], title="% of Reviews"),
        legend={**_legend(), "orientation": "h", "y": 1.14, "x": 0},
    )
    return fig


    return fig


def generate_report(metrics: dict | None = None) -> Path:
    """
    Build the portfolio-quality executive analytics dashboard.

    Outputs to ``data/reports/review_report.html`` and ``docs/index.html``.
    """
    ensure_dirs()

    if metrics is None:
        with METRICS_JSON.open(encoding="utf-8") as handle:
            metrics = json.load(handle)

    daily = pd.DataFrame(metrics["_daily"])
    daily["review_date"] = pd.to_datetime(daily["review_date"])
    weekly = pd.DataFrame(metrics["_weekly"])
    weekly["week"] = pd.to_datetime(weekly["week"])
    monthly = pd.DataFrame(metrics["_monthly"])
    monthly["growth_pct"] = monthly["growth_pct"].apply(
        lambda x: 0.0 if x is None else x
    )

    figures = {
        "weekly_volume": _fig_weekly_volume(weekly),
        "daily_trend": _fig_daily_trend(daily),
        "monthly_rating": _fig_monthly_rating_dist(metrics["_monthly_stars"]),
        "monthly_growth": _fig_monthly_growth(monthly),
        "top_languages": _fig_top_languages(metrics.get("top_languages", [])),
        "tfidf": _fig_tfidf(metrics.get("tfidf_terms", [])),
        "pain_points": _fig_pain_points(metrics.get("theme_counts", {})),
        "monthly_sentiment": _fig_monthly_sentiment(metrics.get("_monthly_sentiment", [])),
        "topic_trend": _fig_topic_trend(metrics.get("_topic_trend", [])),
    }

    chart_layout = [
        ("weekly_volume", "full"),
        ("daily_trend", "half"),
        ("monthly_growth", "half"),
        ("monthly_rating", "full"),
        ("monthly_sentiment", "half"),
        ("top_languages", "half"),
        ("topic_trend", "full"),
        ("tfidf", "half"),
        ("pain_points", "half"),
    ]

    charts_parts = []
    for chart_id, span in chart_layout:
        cls = "chart-card full" if span == "full" else "chart-card"
        charts_parts.append(
            f'<div class="{cls}">\n  {_chart_div(figures[chart_id], chart_id)}\n</div>'
        )
    charts_block = "\n".join(charts_parts)

    cloud_html = {
        "overall": _chart_div(
            _fig_word_cloud(metrics.get("word_frequencies", []), "Overall"),
            "cloud_overall",
        ),
        "negative": _chart_div(
            _fig_word_cloud(metrics.get("word_frequencies_negative", []), "Negative"),
            "cloud_negative",
        ),
        "positive": _chart_div(
            _fig_word_cloud(metrics.get("word_frequencies_positive", []), "Positive"),
            "cloud_positive",
        ),
    }

    date_range = f"{metrics['date_range']['start']} → {metrics['date_range']['end']}"

    comparative = metrics.get("comparative_analysis") or {}
    recommendations = metrics.get("recommendations") or {}
    exec_summary = metrics.get("executive_summary") or metrics.get("insights") or []
    business = metrics.get("business_insights") or []

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape_html(APP_NAME)} · Product Analytics Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{DASHBOARD_CSS}</style>
</head>
<body>
{render_nav()}
<div class="bg-gradient"></div>
<div class="dashboard">

  <header class="page-header">
    <span class="badge">Product Analytics · Google Play</span>
    <h1>{escape_html(APP_NAME)} Customer Review Intelligence</h1>
    <p class="subtitle">
      {escape_html(date_range)} · en-US ·
      <a href="{escape_html(APP_URL)}" target="_blank" rel="noopener">View on Google Play</a>
      · Generated {escape_html(metrics.get('analyzed_at', '')[:10])}
    </p>
  </header>

  {render_executive_summary(exec_summary)}
  {render_kpis(metrics)}
  {render_charts_section(charts_block)}
  {render_customer_voice(metrics.get('customer_voice', []))}
  {render_monthly_deep_dives(metrics.get('monthly_deep_dives', []))}
  {render_comparative(comparative)}
  {render_business_insights(business)}
  {render_recommendations(recommendations)}
  {render_word_clouds(cloud_html)}

  <footer class="footer">
    Pipeline: scrape → clean → analyze → visualize ·
    Data: <code>data/processed/reviews.parquet</code> ·
    Metrics: <code>data/reports/metrics.json</code> ·
    Methods: VADER · TF-IDF · langdetect · Plotly
  </footer>

</div>
</body>
</html>"""

    REPORT_HTML.write_text(report_html, encoding="utf-8")
    DOCS_HTML.write_text(report_html, encoding="utf-8")
    print(f"Executive dashboard saved → {REPORT_HTML}")
    print(f"GitHub Pages copy      → {DOCS_HTML}")
    return REPORT_HTML


def visualize(metrics: dict | None = None) -> Path:
    """Alias for generate_report."""
    return generate_report(metrics)
