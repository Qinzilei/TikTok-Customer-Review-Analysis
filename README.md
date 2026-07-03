# TikTok Google Play Review Analysis

Production-ready Python pipeline for scraping, cleaning, analyzing, and reporting on TikTok Google Play reviews from the last 30 days.

**Live demo:** [Executive Analytics Dashboard](https://qinzilei.github.io/TikTok-Customer-Review-Analysis/)

**Target app:** [TikTok on Google Play](https://play.google.com/store/apps/details?id=com.zhiliaoapp.musically&hl=en_us) (`com.zhiliaoapp.musically`, en-US)

## Project structure

```
TikTok-Customer-Review-Analysis/
├── data/
│   ├── raw/              # Raw scraped JSON
│   ├── processed/        # Cleaned Parquet + CSV exports
│   └── reports/          # Metrics JSON + interactive HTML report
├── notebooks/            # Exploratory analysis notebooks
├── src/
│   ├── scrape_reviews.py # Google Play scraper (30-day window)
│   ├── clean_reviews.py  # Deduplication, normalization, VADER sentiment
│   ├── analyze_reviews.py# Metrics, topic extraction, TF-IDF keywords
│   ├── visualize.py      # Plotly charts + interactive HTML report
│   └── utils.py          # Shared helpers
├── config.py             # App ID, paths, theme keywords
├── docs/index.html       # GitHub Pages deploy (synced from report)
├── requirements.txt
├── run_pipeline.py       # Single entry point
└── README.md
```

## Quick start

```bash
cd TikTok-Customer-Review-Analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
open data/reports/review_report.html
```

Re-run analysis on existing raw data without re-scraping:

```bash
python run_pipeline.py --skip-scrape
```

## Pipeline steps

| Step | Module | Output |
|------|--------|--------|
| 1. Scrape | `src/scrape_reviews.py` | `data/raw/reviews_raw.json` |
| 2. Clean | `src/clean_reviews.py` | `data/processed/reviews.parquet`, `.csv` |
| 3. Analyze | `src/analyze_reviews.py` | `data/reports/metrics.json` |
| 4. Visualize | `src/visualize.py` | `data/reports/review_report.html` |

## Features

- **Scraping:** Paginated fetch (Sort.NEWEST) with 30-day UTC cutoff and retry backoff
- **Cleaning:** Deduplication, type coercion, derived features, VADER sentiment labels
- **Analysis:** Volume/trend metrics, pain-point theme tagging, TF-IDF keyword extraction
- **Visualization:** Dark-theme executive dashboard with interactive Plotly charts (KPI cards, word cloud, sentiment trends)
- **Exports:** Parquet (primary) and CSV (human-readable) datasets

## Methodology

- Reviews scraped via [`google-play-scraper`](https://pypi.org/project/google-play-scraper/) (no API key required)
- Sentiment scored with VADER (`vaderSentiment`)
- Topics extracted via keyword theme rules + scikit-learn TF-IDF
- Charts rendered with Plotly (zoom, pan, hover tooltips)
