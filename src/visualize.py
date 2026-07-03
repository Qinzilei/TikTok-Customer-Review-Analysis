"""Generate interactive Plotly charts and HTML report for TikTok review analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from config import APP_NAME, APP_URL, COLORS, DOCS_HTML, METRICS_JSON, REPORT_HTML, ensure_dirs
from src.utils import escape_html


def _fig_daily_volume(daily: pd.DataFrame) -> go.Figure:
    """Bar chart of daily review volume."""
    fig = go.Figure(
        go.Bar(
            x=daily["review_date"],
            y=daily["review_count"],
            marker_color=COLORS["blue"],
            hovertemplate="%{x}<br>Reviews: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Daily Review Volume (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="Review Count",
        template="plotly_white",
        height=400,
    )
    return fig


def _fig_rating_trend(daily: pd.DataFrame) -> go.Figure:
    """Line chart of daily and rolling average rating."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["review_date"],
            y=daily["avg_rating"],
            mode="lines+markers",
            name="Daily avg",
            line=dict(color=COLORS["blue"]),
            marker=dict(size=4),
            hovertemplate="%{x}<br>Rating: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["review_date"],
            y=daily["rolling_7d_rating"],
            mode="lines",
            name="7-day rolling avg",
            line=dict(color=COLORS["orange"], width=2.5),
            hovertemplate="%{x}<br>7d avg: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Average Rating Trend",
        xaxis_title="Date",
        yaxis_title="Average Rating",
        yaxis=dict(range=[1, 5]),
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _fig_star_distribution(star_dist: dict[str, int]) -> go.Figure:
    """Bar chart of overall star rating distribution."""
    stars = sorted(star_dist.keys(), key=int)
    counts = [star_dist[s] for s in stars]
    fig = go.Figure(
        go.Bar(
            x=[f"{s} star" for s in stars],
            y=counts,
            marker_color=COLORS["stars"][: len(stars)],
            hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Star Rating Distribution",
        xaxis_title="Stars",
        yaxis_title="Count",
        template="plotly_white",
        height=380,
    )
    return fig


def _fig_star_mix(star_pct: pd.DataFrame) -> go.Figure:
    """Stacked area chart of daily star-rating proportions."""
    fig = go.Figure()
    for idx, col in enumerate(star_pct.columns):
        fig.add_trace(
            go.Scatter(
                x=star_pct.index,
                y=star_pct[col],
                mode="lines",
                stackgroup="one",
                name=f"{col} star",
                line=dict(width=0.5),
                fillcolor=COLORS["stars"][idx],
                hovertemplate=f"{col} star: %{{y:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        title="Daily Star Mix (% of Reviews)",
        xaxis_title="Date",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _fig_sentiment(sentiment_dist: dict[str, float]) -> go.Figure:
    """Pie chart of VADER sentiment label distribution."""
    labels = list(sentiment_dist.keys())
    values = list(sentiment_dist.values())
    color_map = {
        "positive": COLORS["green"],
        "neutral": COLORS["yellow"],
        "negative": COLORS["red"],
    }
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=[color_map.get(label, COLORS["blue"]) for label in labels]),
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        title="Sentiment Distribution (VADER)",
        template="plotly_white",
        height=380,
    )
    return fig


def _fig_pain_themes(theme_counts: dict[str, int]) -> go.Figure:
    """Horizontal bar chart of pain-point theme counts."""
    if not theme_counts:
        return go.Figure().update_layout(title="Pain-Point Themes (no data)")
    themes = list(theme_counts.keys())
    counts = [theme_counts[t] for t in themes]
    fig = go.Figure(
        go.Bar(
            x=counts,
            y=themes,
            orientation="h",
            marker_color=COLORS["red"],
            hovertemplate="%{y}: %{x:,} reviews<extra></extra>",
        )
    )
    fig.update_layout(
        title="Pain-Point Themes (1–2 Star Reviews)",
        xaxis_title="Review Count",
        template="plotly_white",
        height=max(320, len(themes) * 45),
    )
    return fig


def _fig_tfidf(tfidf_terms: list[dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart of top TF-IDF terms in negative reviews."""
    if not tfidf_terms:
        return go.Figure().update_layout(title="Top TF-IDF Terms (no data)")
    terms = [t["term"] for t in tfidf_terms]
    scores = [t["score"] for t in tfidf_terms]
    fig = go.Figure(
        go.Bar(
            x=scores,
            y=terms,
            orientation="h",
            marker_color=COLORS["orange"],
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top TF-IDF Terms (Negative Reviews)",
        xaxis_title="TF-IDF Score",
        template="plotly_white",
        height=max(320, len(terms) * 28),
    )
    return fig


def _chart_div(fig: go.Figure, div_id: str) -> str:
    """Render a Plotly figure as an embeddable HTML div (CDN plotly.js)."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": True, "responsive": True},
    )


def _quote_boxes(quotes: dict[str, dict]) -> str:
    """Build HTML callout blocks for representative review quotes."""
    if not quotes:
        return "<p class='small'>No representative quotes available.</p>"
    blocks = []
    for theme, quote in quotes.items():
        blocks.append(
            f"""<div class="callout">
  <b>{escape_html(theme)}</b> — {escape_html(quote['score'])}★ ·
  {escape_html(quote['thumbsUpCount'])} helpful<br>
  <em>"{escape_html(quote['content'])}"</em>
</div>"""
        )
    return "\n".join(blocks)


def _li(items: list[str]) -> str:
    """Build an HTML unordered list from insight strings."""
    return "".join(f"<li>{escape_html(item)}</li>" for item in items)


def generate_report(metrics: dict | None = None) -> Path:
    """
    Build an interactive HTML report with embedded Plotly charts.

    Args:
        metrics: Pre-computed metrics dict. Loads from JSON if not provided.

    Returns:
        Path to the generated ``data/reports/review_report.html`` file.
    """
    ensure_dirs()

    if metrics is None:
        with METRICS_JSON.open(encoding="utf-8") as handle:
            metrics = json.load(handle)

    # Reconstruct DataFrames from serialized metrics
    daily = pd.DataFrame(metrics["_daily"])
    daily["review_date"] = pd.to_datetime(daily["review_date"])

    star_pct = pd.DataFrame(
        metrics["_star_pct_values"],
        index=pd.to_datetime(metrics["_star_pct_index"]),
        columns=metrics["_star_pct_columns"],
    )

    # Build Plotly figures
    figures = {
        "daily_volume": _fig_daily_volume(daily),
        "rating_trend": _fig_rating_trend(daily),
        "star_distribution": _fig_star_distribution(metrics["star_distribution"]),
        "star_mix": _fig_star_mix(star_pct),
        "sentiment": _fig_sentiment(metrics["sentiment_distribution"]),
        "pain_themes": _fig_pain_themes(metrics.get("theme_counts", {})),
        "tfidf": _fig_tfidf(metrics.get("tfidf_terms", [])),
    }

    chart_sections = []
    chart_meta = [
        ("daily_volume", "Daily review volume over the analysis window."),
        ("rating_trend", "Daily average rating and 7-day rolling mean."),
        ("star_mix", "Daily proportion of 1–5 star reviews."),
        ("star_distribution", "Overall star rating histogram."),
        ("sentiment", "VADER sentiment label distribution across all reviews."),
        ("pain_themes", "Complaint themes detected in negative reviews."),
        ("tfidf", "Top TF-IDF terms in negative review text."),
    ]
    for div_id, caption in chart_meta:
        chart_sections.append(
            f"""<figure>
  {_chart_div(figures[div_id], div_id)}
  <figcaption>{escape_html(caption)}</figcaption>
</figure>"""
        )

    # Theme table rows
    theme_rows = ""
    for theme, count in metrics.get("theme_counts", {}).items():
        share = metrics.get("theme_share_of_negative_pct", {}).get(theme, "")
        share_txt = f" ({share}% of negative)" if share else ""
        theme_rows += (
            f"<tr><td>{escape_html(theme)}</td>"
            f"<td class='n'>{count:,}</td>"
            f"<td>{escape_html(share_txt)}</td></tr>"
        )

    version_rows = ""
    for version, count in metrics.get("top_negative_versions", {}).items():
        version_rows += (
            f"<tr><td>{escape_html(version)}</td>"
            f"<td class='n'>{count:,}</td></tr>"
        )

    tfidf_items = ", ".join(
        t["term"] for t in metrics.get("tfidf_terms", [])[:10]
    )

    date_range = (
        f"{metrics['date_range']['start']} → {metrics['date_range']['end']}"
    )
    wow = metrics.get("wow_rating_change")
    wow_text = ""
    if wow is not None:
        direction = "up" if wow > 0 else "down"
        wow_text = (
            f" &nbsp;·&nbsp; WoW rating <b>{direction} {abs(wow):.2f}</b> "
            f"<span>stars</span>"
        )

    # Strip internal chart-serialization keys before rendering appendix
    public_metrics_keys = [
        k for k in metrics if not k.startswith("_")
    ]

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape_html(APP_NAME)} Google Play Review Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{ --tx:#1a1a1a; --muted:#666; --link:#1772d0; --rule:#e6e6e6; }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; color:var(--tx); background:#fff;
    font-family:"Lato","Helvetica Neue",Helvetica,Arial,-apple-system,sans-serif;
    font-size:16px; line-height:1.65;
  }}
  .container {{ max-width:920px; margin:0 auto; padding:46px 22px 80px; }}
  a {{ color:var(--link); text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  h1 {{ font-size:34px; font-weight:700; letter-spacing:-.01em; margin:0 0 4px; }}
  .meta {{ color:var(--muted); font-size:15px; margin-bottom:16px; }}
  .kpis {{
    font-size:15px; font-weight:700; padding:14px 0;
    border-top:1px solid var(--rule); border-bottom:1px solid var(--rule);
  }}
  .kpis span {{ color:var(--muted); font-weight:400; }}
  .kpis b {{ color:var(--tx); }}
  h2 {{ font-size:21px; font-weight:700; margin:40px 0 10px; }}
  h3 {{ font-size:17px; font-weight:700; margin:28px 0 8px; }}
  p {{ margin:0 0 14px; }}
  p.small {{ font-size:14.5px; color:var(--muted); }}
  figure {{ margin:24px 0; }}
  figcaption {{ font-size:14px; color:var(--muted); margin-top:9px; }}
  .callout {{
    border-left:3px solid var(--link); background:#eff6ff;
    padding:12px 16px; border-radius:0 8px 8px 0; margin:16px 0; font-size:14.5px;
  }}
  ul {{ margin:6px 0 14px; }}
  li {{ margin-bottom:4px; }}
  table {{ width:100%; border-collapse:collapse; margin:8px 0; font-size:14.5px; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--rule); }}
  th {{
    font-size:12px; text-transform:uppercase;
    letter-spacing:.04em; color:var(--muted);
  }}
  td.n {{ font-variant-numeric:tabular-nums; font-weight:700; }}
  .appendix {{
    margin-top:40px; padding-top:16px;
    border-top:1px solid var(--rule); font-size:14.5px; color:var(--muted);
  }}
  .plotly-graph-div {{ width:100% !important; }}
</style>
</head>
<body>
<div class="container">

  <h1>{escape_html(APP_NAME)} Google Play Review Analysis</h1>
  <div class="meta">Customer Review Analytics · Google Play · en-US · Last 30 days</div>
  <div class="kpis">
    <b>{metrics['total_reviews']:,}</b> <span>reviews</span> &nbsp;·&nbsp;
    avg rating <b>{metrics['mean_rating']:.2f}</b> <span>stars</span> &nbsp;·&nbsp;
    <b>{metrics['pct_negative']:.1f}%</b> <span>negative (1–2★)</span> &nbsp;·&nbsp;
    <b>{metrics['pct_dev_reply']:.1f}%</b> <span>dev replies</span>{wow_text}
  </div>
  <p class="small">
    Date range: {escape_html(date_range)} ·
    Source: <a href="{escape_html(APP_URL)}">Google Play</a>
  </p>

  <h2>Executive summary</h2>
  <ul>{_li(metrics.get('insights', []))}</ul>

  <h2>Review volume</h2>
  <p>Daily review counts show how user feedback volume changed over the 30-day window.
    Peak day: <b>{escape_html(metrics['peak_day']['date'])}</b>
    with <b>{metrics['peak_day']['count']:,}</b> reviews.</p>
  {chart_sections[0]}

  <h2>Rating trends</h2>
  <p>Average daily rating with a 7-day rolling average to smooth day-to-day noise.
    Lowest-rated day: <b>{escape_html(metrics['lowest_rating_day']['date'])}</b>
    ({metrics['lowest_rating_day']['avg_rating']:.2f} stars).</p>
  {chart_sections[1]}
  {chart_sections[2]}
  {chart_sections[3]}

  <h2>Sentiment analysis</h2>
  <p>VADER sentiment scores classify each review as positive, neutral, or negative
    based on review text (independent of star rating).</p>
  {chart_sections[4]}

  <h2>Pain points &amp; topic extraction</h2>
  <p>Analysis of 1–2 star reviews using keyword theme tagging and TF-IDF
    term extraction.</p>
  {chart_sections[5]}
  <table>
    <thead><tr><th>Theme</th><th>Count</th><th>Share</th></tr></thead>
    <tbody>{theme_rows or '<tr><td colspan="3">No themes detected</td></tr>'}</tbody>
  </table>
  {chart_sections[6]}
  <p class="small">Top terms: {escape_html(tfidf_items or 'N/A')}</p>

  <h3>Representative quotes</h3>
  {_quote_boxes(metrics.get('representative_quotes', {}))}

  <h2>Version breakdown (negative reviews)</h2>
  <table>
    <thead><tr><th>App Version</th><th>Negative Reviews</th></tr></thead>
    <tbody>{version_rows or '<tr><td colspan="2">No version data</td></tr>'}</tbody>
  </table>

  <h2>Key insights</h2>
  <ul>{_li(metrics.get('insights', []))}</ul>

  <div class="appendix">
    <b>Data appendix</b><br>
    Raw data: <code>data/raw/reviews_raw.json</code><br>
    Processed: <code>data/processed/reviews.parquet</code>,
    <code>data/processed/reviews.csv</code><br>
    Metrics: <code>data/reports/metrics.json</code><br>
    Generated: {escape_html(metrics.get('analyzed_at', ''))}<br>
    Metrics keys: {escape_html(', '.join(public_metrics_keys))}<br>
    Methodology: Paginated scrape (Sort.NEWEST) via google-play-scraper;
    30-day UTC cutoff; VADER sentiment; keyword theme tagging + TF-IDF
    on negative reviews; interactive charts via Plotly.
  </div>

</div>
</body>
</html>"""

    REPORT_HTML.write_text(report_html, encoding="utf-8")
    DOCS_HTML.write_text(report_html, encoding="utf-8")
    print(f"Interactive report saved → {REPORT_HTML}")
    print(f"GitHub Pages copy   → {DOCS_HTML}")
    return REPORT_HTML


def visualize(metrics: dict | None = None) -> Path:
    """Alias for generate_report — builds charts and HTML report."""
    return generate_report(metrics)
