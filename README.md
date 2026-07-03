# TikTok Google Play Review Analysis

**Production analytics pipeline** that scrapes, analyzes, and visualizes TikTok Google Play customer reviews — built for product analytics, business intelligence, and data science portfolios.

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Executive_Analytics-6366f1?style=for-the-badge)](https://qinzilei.github.io/TikTok-Customer-Review-Analysis/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3D4F81?style=flat-square)]()

**Live demo:** [Executive Analytics Dashboard](https://qinzilei.github.io/TikTok-Customer-Review-Analysis/)

---

## Project Overview

This project turns raw Google Play review data into an **executive-level product analytics report**. It answers questions product leaders actually ask: *What are customers unhappy about? What changed? Where are the risks? What should each team do next?*

Target app: [TikTok on Google Play](https://play.google.com/store/apps/details?id=com.zhiliaoapp.musically&hl=en_us) (`com.zhiliaoapp.musically`, en-US)

---

## Business Problem

App store reviews are a high-signal, underused source of product intelligence. TikTok receives thousands of reviews per week, but manually reading them does not scale. Teams need:

- **Trend detection** — rating drift, complaint spikes, volume anomalies
- **Root-cause clustering** — algorithm, ads, crashes, account issues
- **Stakeholder-ready narratives** — not charts alone, but *what it means* for Product, Engineering, and Support

This pipeline automates that workflow end-to-end.

---

## Solution

A four-stage Python pipeline:

1. **Scrape** the last 30 days of reviews (paginated, rate-limited)
2. **Clean** and deduplicate; enrich with VADER sentiment labels
3. **Analyze** — KPIs, topic tagging, TF-IDF, period comparisons, business insights
4. **Visualize** — dark-theme executive dashboard with Plotly charts, published to GitHub Pages

---

## Architecture

```
Google Play Store
       │
       ▼
 scrape_reviews.py ──► data/raw/reviews_raw.json
       │
       ▼
 clean_reviews.py   ──► data/processed/reviews.parquet + .csv
       │
       ▼
 analyze_reviews.py ──► data/reports/metrics.json
       │                  (insights.py: narratives, comparisons, recs)
       ▼
 visualize.py      ──► data/reports/review_report.html
       │                  docs/index.html (GitHub Pages)
       ▼
 Executive Dashboard (live)
```

---

## Dashboard Preview

The live dashboard includes:

| Section | Contents |
|---------|----------|
| **Executive Summary** | 5–8 auto-generated business insights (not metric restatements) |
| **KPI Cards** | 13 metrics with alert styling for anomalies |
| **Analytics** | Weekly volume, daily trend, monthly growth, sentiment, topic trend |
| **Customer Voice** | Top helpful review per complaint category with keywords |
| **Monthly Deep Dive** | Narrative analysis of peak/lowest months |
| **Comparative Analysis** | Period-over-period theme and metric tables |
| **Business Insights** | Interpreted findings from the data |
| **Recommendations** | Evidence-backed actions for 5 stakeholder teams |
| **Word Clouds** | Overall, negative, and positive review corpora |

Open the **[live dashboard](https://qinzilei.github.io/TikTok-Customer-Review-Analysis/)** to explore interactive Plotly charts.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Scraping | `google-play-scraper` |
| Data | `pandas`, `pyarrow` (Parquet) |
| NLP | VADER sentiment, scikit-learn TF-IDF, `langdetect` |
| Visualization | Plotly (interactive charts) |
| Deployment | GitHub Pages + GitHub Actions |
| Language | Python 3.11+ |

---

## Methodology

- **Scraping:** Paginated fetch (`Sort.NEWEST`), 30-day UTC cutoff, exponential backoff retries
- **Sentiment:** VADER compound scores → positive / neutral / negative labels
- **Topics:** Rule-based complaint tagging (11 categories) + TF-IDF keyword extraction
- **Languages:** `langdetect` on a stratified sample
- **Insights:** Period-split comparative analysis (first vs second half of window)
- **Quotes:** Highest `thumbsUpCount` review per category — never random selection

---

## Project Structure

```
TikTok-Customer-Review-Analysis/
├── data/
│   ├── raw/              # Raw scraped JSON
│   ├── processed/        # Cleaned Parquet + CSV
│   └── reports/          # metrics.json + review_report.html
├── docs/index.html       # GitHub Pages deploy
├── notebooks/01_eda.ipynb
├── src/
│   ├── scrape_reviews.py
│   ├── clean_reviews.py
│   ├── analyze_reviews.py
│   ├── insights.py       # Business narratives & recommendations
│   ├── visualize.py      # Plotly chart builders
│   └── dashboard.py      # HTML/CSS section templates
├── config.py
├── run_pipeline.py
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/Qinzilei/TikTok-Customer-Review-Analysis.git
cd TikTok-Customer-Review-Analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

```bash
# Full pipeline (scrape + analyze + dashboard)
python run_pipeline.py

# Re-analyze existing raw data (faster)
python run_pipeline.py --skip-scrape

# Open local report
open data/reports/review_report.html
```

---

## Sample Business Insights

From a recent 30-day window (18,000+ reviews):

- Customer satisfaction **deteriorated week-over-week** as rolling average rating declined — recent changes may be eroding trust faster than support can recover.
- **Algorithm** and **Ads** rival **Crashes** as top complaint drivers — monetization friction is as loud as technical instability.
- The **median rating (5.0) exceeds the mean (3.85)** — a vocal 1-star minority with high helpful votes disproportionately shapes public perception.
- **Ad load** complaints correlate with highly upvoted reviews, creating churn risk among daily users and creators.

Full insights, comparisons, and team-specific recommendations are in the [live dashboard](https://qinzilei.github.io/TikTok-Customer-Review-Analysis/).

---

## Future Improvements

- [ ] App Store (iOS) review integration for cross-platform comparison
- [ ] LLM-assisted theme summarization with human-readable cluster labels
- [ ] Scheduled daily scrape via GitHub Actions cron
- [ ] Anomaly detection on review volume and rating (statistical control charts)
- [ ] Version-level regression analysis tied to release dates
- [ ] Streamlit internal tool for ad-hoc date-range filtering

---

## License

MIT — feel free to use as a portfolio reference or starting point.
