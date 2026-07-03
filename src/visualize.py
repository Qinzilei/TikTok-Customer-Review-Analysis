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


def _fig_word_cloud(word_freq: list[dict]) -> go.Figure:
    """Interactive word cloud using spiral-placed Plotly text scatter."""
    if not word_freq:
        return go.Figure().update_layout(**_dark_layout("Word Cloud", height=400))

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
    layout = _dark_layout("Interactive Word Cloud", height=420)
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(
        **layout,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
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


def _kpi_card(label: str, value: str, sub: str = "", accent: str = "cyan") -> str:
    """Render a single KPI glass card."""
    return f"""<div class="kpi-card accent-{accent}">
  <div class="kpi-label">{escape_html(label)}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{escape_html(sub)}</div>
</div>"""


def _quote_cards(quotes: dict[str, dict]) -> str:
    """Render representative quote cards for pain-point themes."""
    if not quotes:
        return '<p class="muted">No representative quotes available.</p>'
    cards = []
    for theme, q in quotes.items():
        cards.append(
            f"""<div class="quote-card">
  <div class="quote-header">
    <span class="quote-theme">{escape_html(theme)}</span>
    <span class="quote-meta">{escape_html(q['score'])}★ · {escape_html(q['thumbsUpCount'])} helpful</span>
  </div>
  <p class="quote-text">"{escape_html(q['content'])}"</p>
</div>"""
        )
    return "\n".join(cards)


DASHBOARD_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg-0: #06080f;
  --bg-1: #0b0f19;
  --bg-2: #111827;
  --glass: rgba(255,255,255,0.04);
  --glass-border: rgba(255,255,255,0.08);
  --text: #e2e8f0;
  --muted: #64748b;
  --cyan: #22d3ee;
  --indigo: #6366f1;
  --violet: #a78bfa;
  --emerald: #34d399;
  --amber: #fbbf24;
  --rose: #fb7185;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  background: var(--bg-0);
  color: var(--text);
  min-height: 100vh;
  line-height: 1.5;
}

.bg-gradient {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,102,241,0.18), transparent),
    radial-gradient(ellipse 60% 40% at 80% 0%, rgba(34,211,238,0.12), transparent),
    radial-gradient(ellipse 50% 30% at 50% 100%, rgba(167,139,250,0.08), transparent);
}

.dashboard {
  position: relative; z-index: 1;
  max-width: 1440px; margin: 0 auto;
  padding: 32px 24px 64px;
}

/* Header */
.header {
  display: flex; flex-wrap: wrap; align-items: flex-end;
  justify-content: space-between; gap: 16px;
  margin-bottom: 32px;
}
.header h1 {
  font-size: clamp(1.5rem, 3vw, 2rem);
  font-weight: 700; letter-spacing: -0.03em;
  background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-meta { font-size: 0.85rem; color: var(--muted); }
.header-meta a { color: var(--cyan); text-decoration: none; }
.header-meta a:hover { text-decoration: underline; }
.badge {
  display: inline-block; padding: 4px 10px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
  text-transform: uppercase; background: rgba(99,102,241,0.15);
  color: var(--violet); border: 1px solid rgba(99,102,241,0.25);
}

/* KPI grid */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 16px; margin-bottom: 32px;
}
.kpi-card {
  background: var(--glass);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  border-radius: 16px; padding: 20px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.3);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  position: relative; overflow: hidden;
}
.kpi-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--cyan), var(--indigo));
  opacity: 0.7;
}
.kpi-card.accent-emerald::before { background: linear-gradient(90deg, #34d399, #22d3ee); }
.kpi-card.accent-rose::before    { background: linear-gradient(90deg, #fb7185, #fbbf24); }
.kpi-card.accent-amber::before   { background: linear-gradient(90deg, #fbbf24, #fb7185); }
.kpi-card.accent-violet::before  { background: linear-gradient(90deg, #a78bfa, #6366f1); }
.kpi-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 32px rgba(99,102,241,0.15);
  border-color: rgba(99,102,241,0.25);
}
.kpi-label {
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted); margin-bottom: 8px;
}
.kpi-value {
  font-size: clamp(1.4rem, 2.5vw, 1.85rem);
  font-weight: 700; font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
}
.kpi-sub { font-size: 0.78rem; color: var(--muted); margin-top: 4px; }

/* Chart cards */
.section-title {
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--muted);
  margin: 32px 0 16px;
}
.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}
.chart-card {
  background: var(--glass);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  border-radius: 16px; padding: 20px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.25);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
  overflow: hidden;
}
.chart-card:hover {
  border-color: rgba(34,211,238,0.15);
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.chart-card.full { grid-column: 1 / -1; }
.chart-card .plotly-graph-div { width: 100% !important; }

/* Quotes */
.quotes-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px; margin-top: 16px;
}
.quote-card {
  background: rgba(99,102,241,0.06);
  border: 1px solid rgba(99,102,241,0.15);
  border-radius: 12px; padding: 16px;
  transition: border-color 0.2s;
}
.quote-card:hover { border-color: rgba(99,102,241,0.35); }
.quote-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 8px; flex-wrap: wrap; gap: 4px;
}
.quote-theme { font-weight: 600; color: var(--violet); font-size: 0.85rem; }
.quote-meta { font-size: 0.75rem; color: var(--muted); }
.quote-text { font-size: 0.85rem; color: #cbd5e1; font-style: italic; line-height: 1.6; }

.muted { color: var(--muted); font-size: 0.85rem; }
.footer {
  margin-top: 48px; padding-top: 24px;
  border-top: 1px solid var(--glass-border);
  font-size: 0.78rem; color: var(--muted);
}
.footer code {
  background: rgba(255,255,255,0.06); padding: 2px 6px;
  border-radius: 4px; font-size: 0.75rem;
}

@media (max-width: 900px) {
  .charts-grid { grid-template-columns: 1fr; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 480px) {
  .kpi-grid { grid-template-columns: 1fr; }
  .dashboard { padding: 20px 16px 48px; }
}
"""


def generate_report(metrics: dict | None = None) -> Path:
    """
    Build the executive analytics dashboard HTML report.

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
        "word_cloud": _fig_word_cloud(metrics.get("word_frequencies", [])),
        "tfidf": _fig_tfidf(metrics.get("tfidf_terms", [])),
        "pain_points": _fig_pain_points(metrics.get("theme_counts", {})),
        "monthly_sentiment": _fig_monthly_sentiment(metrics.get("_monthly_sentiment", [])),
    }

    chart_layout = [
        ("weekly_volume", "full"),
        ("daily_trend", "half"),
        ("monthly_growth", "half"),
        ("monthly_rating", "full"),
        ("monthly_sentiment", "half"),
        ("top_languages", "half"),
        ("word_cloud", "full"),
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

    date_range = f"{metrics['date_range']['start']} → {metrics['date_range']['end']}"
    wow = metrics.get("wow_rating_change")
    wow_note = ""
    if wow is not None:
        arrow = "↑" if wow > 0 else "↓"
        wow_note = f" · WoW rating {arrow} {abs(wow):.2f}"

    kpi_html = "".join([
        _kpi_card("Total Reviews", f"{metrics['total_reviews']:,}", date_range, "cyan"),
        _kpi_card("Average Rating", f"{metrics['mean_rating']:.2f}", "out of 5 stars", "indigo"),
        _kpi_card("Median Rating", f"{metrics['median_rating']:.1f}", "50th percentile", "violet"),
        _kpi_card("Positive %", f"{metrics['pct_positive']:.1f}%", "4–5 star reviews", "emerald"),
        _kpi_card("Negative %", f"{metrics['pct_negative']:.1f}%", "1–2 star reviews", "rose"),
        _kpi_card("Reviews / Day", f"{metrics['reviews_per_day']:,.1f}", "daily average", "amber"),
        _kpi_card(
            "Peak Review Month",
            metrics["peak_month"]["label"],
            f"{metrics['peak_month']['count']:,} reviews",
            "cyan",
        ),
        _kpi_card(
            "Lowest Rating Month",
            metrics["lowest_rating_month"]["label"],
            f"{metrics['lowest_rating_month']['avg_rating']:.2f} avg",
            "rose",
        ),
    ])

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape_html(APP_NAME)} · Executive Review Analytics</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{DASHBOARD_CSS}</style>
</head>
<body>
<div class="bg-gradient"></div>
<div class="dashboard">

  <header class="header">
    <div>
      <span class="badge">Executive Dashboard</span>
      <h1>{escape_html(APP_NAME)} Google Play Review Analytics</h1>
      <p class="header-meta">
        {escape_html(date_range)} · en-US ·
        <a href="{escape_html(APP_URL)}" target="_blank" rel="noopener">Google Play</a>
        {escape_html(wow_note)}
      </p>
    </div>
    <p class="header-meta">Generated {escape_html(metrics.get('analyzed_at', '')[:10])}</p>
  </header>

  <section class="kpi-grid">{kpi_html}</section>

  <p class="section-title">Volume &amp; Trends</p>
  <div class="charts-grid">
{charts_block}
  </div>

  <p class="section-title">Voice of Customer — Representative Quotes</p>
  <div class="quotes-grid">{_quote_cards(metrics.get('representative_quotes', {}))}</div>

  <footer class="footer">
    Data: <code>data/processed/reviews.parquet</code> ·
    Metrics: <code>data/reports/metrics.json</code> ·
    Methodology: google-play-scraper · VADER sentiment · TF-IDF · langdetect · Plotly
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
