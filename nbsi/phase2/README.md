# NBElastic Phase-2 — ETL → Panel Builders (v1.2)

This phase builds the data layer from Polygon news → cleaned articles → sentiment/relevance features → sector panel for Phase-3 fusion/backtest. It preserves Phase-0 business rules and rate limits.

## How to run

1. Preconditions
   - Git tag `nbelastic-v1.2-phase0` exists
   - Secrets at `nbsi/phase1/configs/secrets.yaml` (Polygon, Finnhub, Alpaca) — not committed
   - Phase-0 RL+QA artifacts exist under `artifacts/phase0_rl_qa/` (spy_proxy_report.csv, price_health.csv, coverage_report.csv)

2. Execute Phase-2 driver

```
python nbsi/phase2/scripts/run_phase2.py
```

Outputs in `artifacts/phase2/`:
- news_raw_YYYYMMDD.parquet (per day)
- news_clean_YYYYMMDD.parquet (per day)
- article_sentiment.parquet
- article_relevance.parquet
- sector_panel.parquet (union, 60 ET trading days)
- qa_phase2.log (QA summary)

## Schemas
- news_raw: Polygon v2 reference/news fields (id, published_utc, publisher, title, description, tickers, article_url, ...)
- news_clean: article_id, published_utc, assigned_date_et, source, title, description, tickers, url
- article_sentiment: article_id, polarity (p_pos - p_neg), confidence (max softmax)
- article_relevance: article_id, relevance_{SECTOR} in [0,1]
- sector_panel: date_et, sector, n_articles, stale_share, mean_polarity, std_polarity, pct_extreme, conf_mean, rel_weight_sum, spy_sentiment, rv20, rv60

## Notes
- Polygon news pagination uses `next_url` (absolute) and bearer header; do not append API key to the URL
- Finnhub ≤30 req/s ceiling (target ~25/s steady); 429+Retry-After honored (inherited client from Phase-1)
- Embargo: if published_utc > 15:55 ET, assign to next ET trading day; signals computed around 15:15 ET
- Alpaca OPG (time_in_force="opg") is used in Phase-4 for next-open routing (not executed here)

