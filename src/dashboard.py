"""HTML/CSS templates and section builders for the executive analytics dashboard."""

from __future__ import annotations

import re
from typing import Any

from src.utils import escape_html

# ---------------------------------------------------------------------------
# Stylesheet — Bloomberg / Datadog-inspired dark executive theme
# ---------------------------------------------------------------------------

DASHBOARD_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg-0: #050810;
  --bg-1: #0a0e17;
  --glass: rgba(255,255,255,0.035);
  --glass-hover: rgba(255,255,255,0.055);
  --glass-border: rgba(255,255,255,0.07);
  --text: #e8edf5;
  --muted: #64748b;
  --cyan: #22d3ee;
  --indigo: #6366f1;
  --violet: #a78bfa;
  --emerald: #34d399;
  --amber: #fbbf24;
  --rose: #fb7185;
  --nav-h: 56px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; scroll-padding-top: calc(var(--nav-h) + 20px); }

body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg-0);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
}

.bg-gradient {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 70% 45% at 15% -5%, rgba(99,102,241,0.14), transparent),
    radial-gradient(ellipse 55% 35% at 85% 5%, rgba(34,211,238,0.09), transparent),
    radial-gradient(ellipse 40% 25% at 50% 100%, rgba(167,139,250,0.06), transparent);
}

/* Sticky navigation */
.topnav {
  position: sticky; top: 0; z-index: 100;
  height: var(--nav-h);
  background: rgba(5,8,16,0.82);
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--glass-border);
  display: flex; align-items: center;
  padding: 0 24px; gap: 8px;
  overflow-x: auto; white-space: nowrap;
}
.topnav-brand {
  font-weight: 700; font-size: 0.85rem; color: var(--text);
  margin-right: 16px; flex-shrink: 0;
  background: linear-gradient(135deg, var(--cyan), var(--violet));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.topnav a {
  color: var(--muted); text-decoration: none; font-size: 0.78rem;
  font-weight: 500; padding: 6px 12px; border-radius: 8px;
  transition: color 0.2s, background 0.2s;
}
.topnav a:hover { color: var(--text); background: var(--glass); }

.dashboard {
  position: relative; z-index: 1;
  max-width: 1440px; margin: 0 auto;
  padding: 28px 24px 80px;
}

/* Header */
.page-header { margin-bottom: 28px; }
.page-header h1 {
  font-size: clamp(1.6rem, 3vw, 2.1rem);
  font-weight: 700; letter-spacing: -0.03em; margin: 8px 0 6px;
}
.page-header .subtitle { color: var(--muted); font-size: 0.88rem; }
.page-header .subtitle a { color: var(--cyan); text-decoration: none; }
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 999px; font-size: 0.68rem;
  font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  background: rgba(99,102,241,0.12); color: var(--violet);
  border: 1px solid rgba(99,102,241,0.22);
}

/* Section structure */
section { margin-bottom: 40px; animation: fadeUp 0.5s ease both; }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.section-head {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 18px; padding-bottom: 12px;
  border-bottom: 1px solid var(--glass-border);
}
.section-head h2 {
  font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em;
}
.section-icon { font-size: 1.1rem; opacity: 0.85; }

/* Glass panels */
.panel {
  background: var(--glass);
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
  border: 1px solid var(--glass-border);
  border-radius: 16px; padding: 22px;
  box-shadow: 0 4px 30px rgba(0,0,0,0.35);
  transition: border-color 0.25s, box-shadow 0.25s;
}
.panel:hover { border-color: rgba(99,102,241,0.18); }

/* Executive summary */
.insight-list { list-style: none; display: flex; flex-direction: column; gap: 12px; }
.insight-list li {
  padding: 14px 16px 14px 20px;
  border-left: 3px solid var(--indigo);
  background: rgba(99,102,241,0.05);
  border-radius: 0 10px 10px 0;
  font-size: 0.9rem; color: #cbd5e1; line-height: 1.65;
  transition: background 0.2s;
}
.insight-list li:hover { background: rgba(99,102,241,0.09); }
.insight-list li strong { color: var(--text); font-weight: 600; }

/* KPI grid */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
  gap: 14px;
}
.kpi-card {
  background: var(--glass);
  backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  border-radius: 14px; padding: 16px 18px;
  position: relative; overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
  animation: fadeUp 0.4s ease both;
}
.kpi-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--cyan), var(--indigo));
  opacity: 0.75;
}
.kpi-card.alert::before { background: linear-gradient(90deg, var(--rose), var(--amber)); }
.kpi-card.positive::before { background: linear-gradient(90deg, var(--emerald), var(--cyan)); }
.kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 36px rgba(99,102,241,0.12);
  border-color: rgba(99,102,241,0.22);
}
.kpi-label {
  font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--muted); margin-bottom: 6px;
}
.kpi-value {
  font-size: clamp(1.25rem, 2.2vw, 1.65rem);
  font-weight: 700; font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
}
.kpi-value.up { color: var(--emerald); }
.kpi-value.down { color: var(--rose); }
.kpi-sub { font-size: 0.72rem; color: var(--muted); margin-top: 3px; }
.kpi-indicator {
  display: inline-block; font-size: 0.65rem; font-weight: 600;
  padding: 2px 7px; border-radius: 6px; margin-top: 6px;
}
.kpi-indicator.warn { background: rgba(251,191,36,0.15); color: var(--amber); }
.kpi-indicator.bad  { background: rgba(251,113,133,0.15); color: var(--rose); }
.kpi-indicator.good { background: rgba(52,211,153,0.15); color: var(--emerald); }

/* Charts */
.charts-grid {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px;
}
.chart-card {
  background: var(--glass);
  backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  border-radius: 16px; padding: 18px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.28);
  transition: border-color 0.2s;
  overflow: hidden;
}
.chart-card:hover { border-color: rgba(34,211,238,0.12); }
.chart-card.full { grid-column: 1 / -1; }
.chart-card .plotly-graph-div { width: 100% !important; }

/* Word cloud row */
.cloud-row {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px;
}
.cloud-card {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 16px; padding: 16px;
  min-height: 380px;
}
.cloud-card h3 {
  font-size: 0.78rem; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em;
  margin-bottom: 8px; text-align: center;
}

/* Customer voice */
.voice-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}
.voice-card {
  background: rgba(99,102,241,0.04);
  border: 1px solid rgba(99,102,241,0.12);
  border-radius: 14px; padding: 18px;
  transition: border-color 0.2s, transform 0.2s;
}
.voice-card:hover { border-color: rgba(99,102,241,0.3); transform: translateY(-2px); }
.voice-top {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 10px; flex-wrap: wrap; gap: 6px;
}
.voice-theme {
  font-weight: 600; color: var(--violet); font-size: 0.88rem;
  display: flex; align-items: center; gap: 6px;
}
.voice-meta { font-size: 0.72rem; color: var(--muted); }
.voice-quote {
  font-size: 0.84rem; color: #cbd5e1; font-style: italic;
  line-height: 1.65; margin-bottom: 10px;
}
.voice-keywords { display: flex; flex-wrap: wrap; gap: 5px; }
.kw-tag {
  font-size: 0.65rem; padding: 3px 8px; border-radius: 6px;
  background: rgba(34,211,238,0.1); color: var(--cyan);
  border: 1px solid rgba(34,211,238,0.18);
}

/* Tables */
.data-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.data-table th {
  text-align: left; padding: 10px 12px;
  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); border-bottom: 1px solid var(--glass-border);
}
.data-table td {
  padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.04);
  font-variant-numeric: tabular-nums;
}
.data-table tr:hover td { background: var(--glass-hover); }
.data-table .sig-up { color: var(--rose); font-weight: 600; }
.data-table .sig-down { color: var(--emerald); font-weight: 600; }
.data-table .neutral { color: var(--muted); }

/* Monthly deep dive */
.dive-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }
.dive-card h3 { font-size: 1rem; font-weight: 600; margin-bottom: 10px; color: var(--cyan); }
.dive-card p { font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px; }
.dive-card .label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-top: 12px; }

/* Recommendations */
.rec-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
.rec-card h3 {
  font-size: 0.82rem; font-weight: 600; color: var(--violet);
  margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.04em;
}
.rec-card ul { list-style: none; display: flex; flex-direction: column; gap: 8px; }
.rec-card li {
  font-size: 0.82rem; color: #94a3b8; padding-left: 14px;
  position: relative; line-height: 1.55;
}
.rec-card li::before {
  content: '→'; position: absolute; left: 0; color: var(--indigo);
}

.footer {
  margin-top: 48px; padding-top: 20px;
  border-top: 1px solid var(--glass-border);
  font-size: 0.75rem; color: var(--muted);
}
.footer code {
  background: rgba(255,255,255,0.05); padding: 2px 6px;
  border-radius: 4px; font-size: 0.72rem;
}

@media (max-width: 1024px) {
  .cloud-row { grid-template-columns: 1fr; }
  .charts-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .dashboard { padding: 16px 14px 60px; }
}
"""


def _md_bold(text: str) -> str:
    """Convert **bold** markdown to HTML strong tags."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escape_html(text))


def render_nav() -> str:
    """Sticky section navigation."""
    links = [
        ("#summary", "Summary"),
        ("#kpis", "KPIs"),
        ("#charts", "Analytics"),
        ("#voice", "Customer Voice"),
        ("#deep-dive", "Deep Dive"),
        ("#compare", "Compare"),
        ("#insights", "Insights"),
        ("#recs", "Actions"),
        ("#clouds", "Word Clouds"),
    ]
    items = "".join(f'<a href="{href}">{label}</a>' for href, label in links)
    return f"""<nav class="topnav">
  <span class="topnav-brand">TikTok Analytics</span>
  {items}
</nav>"""


def render_executive_summary(items: list[str]) -> str:
    """Executive summary panel with business insights."""
    lis = "".join(f"<li>{_md_bold(item)}</li>" for item in items)
    return f"""<section id="summary">
  <div class="section-head"><span class="section-icon">📋</span><h2>Executive Summary</h2></div>
  <div class="panel"><ul class="insight-list">{lis}</ul></div>
</section>"""


def render_kpis(metrics: dict[str, Any]) -> str:
    """KPI card grid with alert styling for unusual values."""
    wow = metrics.get("wow_rating_change")
    wow_val = f"{wow:+.2f}" if wow is not None else "N/A"
    wow_cls = "down" if wow is not None and wow < 0 else "up" if wow and wow > 0 else ""
    wow_ind = ""
    if wow is not None and abs(wow) >= 0.05:
        wow_ind = f'<span class="kpi-indicator {"bad" if wow < 0 else "good"}">{"Declining" if wow < 0 else "Improving"}</span>'

    drift = metrics.get("rating_drift", 0)
    drift_cls = "alert" if drift < -0.05 else "positive" if drift > 0.05 else ""

    spike = metrics.get("low_rating_spike", 0)
    spike_ind = ""
    if spike >= 5:
        spike_ind = '<span class="kpi-indicator warn">Spike detected</span>'

    cards = [
        ("Total Reviews", f"{metrics['total_reviews']:,}", metrics["date_range"]["start"] + " – " + metrics["date_range"]["end"], ""),
        ("Average Rating", f"{metrics['mean_rating']:.2f}", "out of 5 stars", "alert" if metrics["mean_rating"] < 3.5 else ""),
        ("Median Rating", f"{metrics['median_rating']:.1f}", "50th percentile", ""),
        ("Positive %", f"{metrics['pct_positive']:.1f}%", "4–5 star reviews", "positive"),
        ("Negative %", f"{metrics['pct_negative']:.1f}%", "1–2 star reviews", "alert" if metrics["pct_negative"] > 25 else ""),
        ("Reviews / Day", f"{metrics['reviews_per_day']:,.1f}", "daily average", ""),
        ("Peak Month", metrics["peak_month"]["label"], f"{metrics['peak_month']['count']:,} reviews", ""),
        ("Lowest Rating Month", metrics["lowest_rating_month"]["label"], f"{metrics['lowest_rating_month']['avg_rating']:.2f} avg", "alert"),
        ("WoW Rating Δ", wow_val, "7-day rolling change", wow_cls),
        ("Rating Drift", f"{drift:+.2f}", "2nd half vs 1st half", drift_cls),
        ("Low-Rating Spike", f"+{spike:.1f}pp", metrics.get("low_rating_spike_date", ""), "alert" if spike >= 5 else ""),
        ("Dominant Language", metrics.get("dominant_language", "—"), "detected from text", ""),
        ("Avg Helpful Votes", f"{metrics.get('avg_helpful_votes', 0):.1f}", "per review", ""),
    ]

    html_parts = []
    for label, value, sub, cls in cards:
        extra = wow_ind if label == "WoW Rating Δ" else spike_ind if label == "Low-Rating Spike" else ""
        val_cls = ""
        if label == "WoW Rating Δ" and wow_cls:
            val_cls = wow_cls
        html_parts.append(f"""<div class="kpi-card {cls}">
  <div class="kpi-label">{escape_html(label)}</div>
  <div class="kpi-value {val_cls}">{escape_html(value)}</div>
  <div class="kpi-sub">{escape_html(sub)}</div>
  {extra}
</div>""")

    return f"""<section id="kpis">
  <div class="section-head"><span class="section-icon">📊</span><h2>Key Performance Indicators</h2></div>
  <div class="kpi-grid">{"".join(html_parts)}</div>
</section>"""


def render_customer_voice(entries: list[dict]) -> str:
    """Customer Voice cards grouped by complaint category."""
    if not entries:
        return """<section id="voice"><div class="section-head"><h2>Customer Voice</h2></div>
<p class="subtitle">No categorized reviews found.</p></section>"""

    cards = []
    for e in entries:
        kws = "".join(f'<span class="kw-tag">{escape_html(k)}</span>' for k in e.get("keywords", []))
        cards.append(f"""<div class="voice-card">
  <div class="voice-top">
    <span class="voice-theme">💬 {escape_html(e['theme'])}</span>
    <span class="voice-meta">{e['score']}★ · {e['thumbsUpCount']:,} helpful · {e['review_count']:,} reviews</span>
  </div>
  <p class="voice-quote">"{escape_html(e['content'])}"</p>
  <div class="voice-keywords">{kws}</div>
</div>""")

    return f"""<section id="voice">
  <div class="section-head"><span class="section-icon">🎙️</span><h2>Customer Voice</h2></div>
  <p class="subtitle" style="margin-bottom:16px;color:var(--muted);font-size:0.85rem;">
    Most helpful review per complaint category — selected by thumbs-up count, not random sampling.
  </p>
  <div class="voice-grid">{"".join(cards)}</div>
</section>"""


def render_comparative(comp: dict[str, Any]) -> str:
    """Period-over-period comparison tables."""
    pa = escape_html(comp.get("period_a_label", "Period A"))
    pb = escape_html(comp.get("period_b_label", "Period B"))

    theme_rows = ""
    for row in comp.get("theme_comparison", []):
        chg = row["change"]
        cls = "sig-up" if chg > 0 and row.get("significant") else "sig-down" if chg < 0 and row.get("significant") else "neutral"
        sign = "+" if chg > 0 else ""
        theme_rows += f"""<tr>
  <td>{escape_html(row['theme'])}</td>
  <td>{row['period_a']:,}</td><td>{row['period_b']:,}</td>
  <td class="{cls}">{sign}{chg:,}</td>
</tr>"""

    summary_rows = ""
    for row in comp.get("summary_comparison", []):
        chg = row["change"]
        sign = "+" if isinstance(chg, (int, float)) and chg > 0 else ""
        summary_rows += f"""<tr>
  <td>{escape_html(row['metric'])}</td>
  <td>{row['period_a']}</td><td>{row['period_b']}</td>
  <td>{sign}{chg}</td>
</tr>"""

    return f"""<section id="compare">
  <div class="section-head"><span class="section-icon">⚖️</span><h2>Comparative Analysis</h2></div>
  <div class="charts-grid" style="margin-bottom:18px;">
    <div class="panel">
      <h3 style="font-size:0.85rem;margin-bottom:12px;color:var(--muted);">Summary Metrics</h3>
      <table class="data-table">
        <thead><tr><th>Metric</th><th>{pa}</th><th>{pb}</th><th>Change</th></tr></thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </div>
    <div class="panel">
      <h3 style="font-size:0.85rem;margin-bottom:12px;color:var(--muted);">Complaint Themes</h3>
      <table class="data-table">
        <thead><tr><th>Theme</th><th>{pa}</th><th>{pb}</th><th>Δ</th></tr></thead>
        <tbody>{theme_rows}</tbody>
      </table>
    </div>
  </div>
</section>"""


def render_monthly_deep_dives(dives: list[dict]) -> str:
    """Monthly deep-dive narrative cards."""
    if not dives:
        return ""

    cards = []
    for d in dives:
        complaints = ", ".join(
            f"{k} ({v})" for k, v in sorted(
                d.get("complaint_distribution", {}).items(),
                key=lambda x: x[1], reverse=True,
            )[:5]
        )
        kws = ", ".join(d.get("top_keywords", [])[:6])
        cards.append(f"""<div class="panel dive-card">
  <h3>{escape_html(d['month'])}</h3>
  <p>{escape_html(d['summary'])}</p>
  <p class="label">Complaint Distribution</p>
  <p>{escape_html(complaints or 'None')}</p>
  <p class="label">What Changed</p>
  <p>{escape_html(d['what_changed'])}</p>
  <p class="label">Volume</p>
  <p>{escape_html(d['volume_interpretation'])}</p>
  <p class="label">Rating</p>
  <p>{escape_html(d['rating_interpretation'])}</p>
  <p class="label">Top Keywords</p>
  <p>{escape_html(kws)}</p>
  <p class="label">Business Interpretation</p>
  <p>{escape_html(d['business_interpretation'])}</p>
</div>""")

    return f"""<section id="deep-dive">
  <div class="section-head"><span class="section-icon">🔍</span><h2>Monthly Deep Dive</h2></div>
  <div class="dive-grid">{"".join(cards)}</div>
</section>"""


def render_business_insights(items: list[str]) -> str:
    """Executive insights section."""
    lis = "".join(f"<li>{_md_bold(item)}</li>" for item in items)
    return f"""<section id="insights">
  <div class="section-head"><span class="section-icon">💡</span><h2>Business Insights</h2></div>
  <div class="panel"><ul class="insight-list">{lis}</ul></div>
</section>"""


def render_recommendations(recs: dict[str, list[str]]) -> str:
    """Stakeholder-specific recommendation cards."""
    cards = []
    icons = {
        "Product Team": "🎯",
        "Engineering": "⚙️",
        "Customer Support": "🎧",
        "Trust & Safety": "🛡️",
        "Marketing": "📣",
    }
    for team, items in recs.items():
        lis = "".join(f"<li>{escape_html(item)}</li>" for item in items)
        icon = icons.get(team, "•")
        cards.append(f"""<div class="panel rec-card">
  <h3>{icon} {escape_html(team)}</h3>
  <ul>{lis}</ul>
</div>""")
    return f"""<section id="recs">
  <div class="section-head"><span class="section-icon">✅</span><h2>Business Recommendations</h2></div>
  <div class="rec-grid">{"".join(cards)}</div>
</section>"""


def render_word_clouds(cloud_html: dict[str, str]) -> str:
    """Three word-cloud chart containers."""
    return f"""<section id="clouds">
  <div class="section-head"><span class="section-icon">☁️</span><h2>Word Cloud Analysis</h2></div>
  <div class="cloud-row">
    <div class="cloud-card"><h3>Overall</h3>{cloud_html.get('overall', '')}</div>
    <div class="cloud-card"><h3>Negative Reviews</h3>{cloud_html.get('negative', '')}</div>
    <div class="cloud-card"><h3>Positive Reviews</h3>{cloud_html.get('positive', '')}</div>
  </div>
</section>"""


def render_charts_section(charts_block: str) -> str:
    """Analytics charts grid."""
    return f"""<section id="charts">
  <div class="section-head"><span class="section-icon">📈</span><h2>Analytics &amp; Visualizations</h2></div>
  <div class="charts-grid">{charts_block}</div>
</section>"""
