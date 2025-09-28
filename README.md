![nbsi tests](https://github.com/Josh116m/Ticker_Tattle/actions/workflows/nbsi-tests.yml/badge.svg)


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


## Phase-5 — Reporting

Generates a daily equity report and chart from Phase-4 outputs, with an optional SPY overlay.

**Inputs (from Phase-4)**
- `artifacts/phase4/pnl_by_day.parquet` — must include `date_et` + one of `ret`/`pnl`/`portfolio_ret`/`daily_ret`
- `artifacts/phase4/fills.parquet`
- *(optional)* `artifacts/phase4/exec_summary.json` — used for exposure stats (`avg_gross`, `stop_days`)

**Optional inputs (for SPY overlay)**
- `artifacts/phase3/opens.parquet`, `artifacts/phase3/closes.parquet` — SPY open→close overlay (dashed). If missing, the script logs a QA note and proceeds.

**Outputs**
- `artifacts/phase5/daily_equity.csv` — columns: `date_et, ret, equity`
- `artifacts/phase5/equity_curve.png` — strategy curve (+ dashed SPY overlay when available)
- `artifacts/phase5/daily_report.md` — days, final equity, exposure stats, last-5-days fills-by-ticker table, QA notes
- `artifacts/phase5/qa_phase5.log` — “QA PASS …” + overlay status notes

**Run**
````bash
python nbsi/phase5/scripts/run_phase5.py \
  --from artifacts/phase4 \
  --out  artifacts/phase5 \
  --phase3-root artifacts/phase3   # optional, for SPY overlay
````

**Example CLI output**

```
PHASE 5 REPORT COMPLETE
```

**Notes**

- SPY overlay is best-effort; if pivots are absent or dates don’t overlap, the report is still produced and a QA note explains why the overlay was skipped.
- Exposure stats are pulled from `exec_summary.json` when present (avg gross, stop days).
- Last-5-days fills snapshot is derived from `fills.parquet`.

**Tests (no network/secrets)**

````bash
python -m unittest nbsi.phase5.tests.test_phase5_smoke -v
````


## Troubleshooting — Phase-5 scheduled report (Windows)

**Where it runs**
- Task name: `NBElastic Phase5 Report`
- Command: `cmd.exe /c "D:\Ticker Tattle\scripts\run_phase5_report.cmd"`
- Start in: `D:\Ticker Tattle`

**Common issues / fixes**
- Script runs manually but not via Scheduler:
  - Set “Start in (optional)” to the repo root.
  - Configure “Run whether user is logged on or not.”
  - Ensure the user has read/write permissions to `artifacts\phase4` and `artifacts\phase5`.
- No new outputs / stale PNG:
  - Check `artifacts\phase5\qa_phase5.log` (logs are rotated: `qa_phase5.log`, `.1` … `.7`).
  - Verify `python` on PATH matches the interpreter with `pandas/pyarrow/matplotlib`.
- Exit code ≠ 0:
  - Run: `scripts\run_phase5_report.cmd` in a terminal to see the full traceback.
  - Make sure Phase-4 outputs exist: `artifacts\phase4\pnl_by_day.parquet`, `artifacts\phase4\fills.parquet`.
  - Optional SPY overlay: ensure Phase-3 pivots exist: `artifacts\phase3\opens.parquet`, `closes.parquet`.

**Manual runs & log locations**
- Manual run: `scripts\run_phase5_report.cmd`
- Logs: `artifacts\phase5\qa_phase5.log` (plus rotated `.1`–`.7`)


## Phase-4 — Routing Runbook

**Dry preview (safe, no network):**
````bash
python nbsi/phase4/scripts/run_phase4.py --mode route --dry-run true --from artifacts/phase3 --emit-csv true
````

Outputs:

- artifacts/phase4/orders_intents.parquet (+ optional orders_intents.csv)
- artifacts/phase4/orders_preview.parquet
- QA log tail includes a run banner and `provenance: host=... user=...`.

**Paper submission (Alpaca sandbox)**
Guarded: requires env creds and explicit confirm gate.

````bash
set ALPACA_KEY_ID=...        # or persist via User env
set ALPACA_SECRET_KEY=...
python nbsi/phase4/scripts/run_phase4.py --mode route --dry-run false --confirm "I UNDERSTAND" --from artifacts/phase3
````

Outputs:

- artifacts/phase4/orders_submitted.parquet (status_code per order)
- QA log line: `[route] SUBMITTED rows=... -> ...orders_submitted.parquet`
- End message: `PHASE 4 ROUTE (SUBMITTED) COMPLETE`

**Notes**

- Business rules remain frozen (v1.2): 2-day hold, 3L/3S, sector ≤30%, gross ≤150%, 5% daily stop, OPG timing.
- `NBSI_OUT_ROOT` is honored by all scripts if set.
- For safety, the submit path refuses without `--confirm "I UNDERSTAND"` and valid env creds.
