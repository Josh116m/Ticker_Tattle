# NBElastic Phase‑0 Diagnostic

This folder contains a self‑contained implementation of the Phase‑0
diagnostic for the NBElastic project.  It performs the data
quality checks described in the specification by downloading recent
news and market data, computing coverage/diversity/staleness metrics,
building a SPY sentiment proxy, validating price completeness and
realised volatility, and assessing the availability of earnings data.

## Files

* `config.yaml` – Sample configuration with API keys and
  parameters.  **Replace these values or set environment variables
  securely before running in production.**
* `diagnostic_phase0.py` – Main script that orchestrates all
  diagnostics and writes the required CSV reports and summary.
* `utils.py` – Helper functions for tokenisation, n‑gram Jaccard
  similarity, deduplication and realised volatility.

## How to run

1. Install the required Python packages (requests, pandas, numpy,
   pyyaml, transformers, torch, pytz) into your environment.  A GPU is
   optional but recommended for the sentiment model.  For example:

   ```bash
   pip install requests pandas numpy pyyaml pytz transformers torch
   ```

2. Place your Polygon and Finnhub API keys into `config.yaml`
   (or export them via environment variables and adjust the
   configuration loader accordingly).

3. Run the diagnostic and specify an output directory, e.g.:

   ```bash
   python diagnostic_phase0.py --config config.yaml --output-dir ./phase0_reports
   ```

The script will download news and price data for the last two
weeks, compute the metrics, and write the following outputs into
`phase0_reports/`:

* `coverage_report.csv` – Per‑day, per‑sector counts and quality
  indicators.
* `spy_proxy_report.csv` – Daily SPY sentiment counts, fallback
  flags and average polarity.
* `price_health.csv` – Daily bar completeness and realised
  volatility for each sector ETF and SPY.
* `earnings_density.csv` – Sector earnings density for each day.
* `phase0_summary.md` – A human‑readable summary of the diagnostic
  results with pass/fail indications.

If the diagnostic fails any of the threshold checks, consult the
summary for suggested fallback actions.  Otherwise, you can proceed
to the next phases of the NBElastic build.
