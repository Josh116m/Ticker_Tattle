# Nightly maintenance (Windows, safe-by-default)
# - Assumes repo layout as in Ticker_Tattle and that Python 3.11.9 is on PATH
# - Runs from repo root automatically (regardless of where invoked)
# - Submits paper orders to Alpaca after DRY route (requires ALPACA_* env) + daily report
# - Verifies required artifacts and tails QA logs

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Ensure we run from the repo root (script lives in scripts/)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Resolve-Path (Join-Path $scriptDir '..')
Set-Location $repoRoot

Write-Host ("[nightly] repoRoot=" + (Get-Location))
# Resolve Python interpreter (prefer repo venv if present)
$venvPy = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (Test-Path $venvPy) { $pyExe = $venvPy } else { $pyExe = 'python' }
Write-Host ("[nightly] using python: " + $pyExe)


# 1) Check required API keys
if (-not $env:POLYGON_API_KEY) {
  Write-Error 'POLYGON_API_KEY not set; cannot refresh data.'
  exit 1
}

# 2) Refresh Phase-2 news and sentiment data (always run to get latest news)
Write-Host '[nightly] Refreshing Phase-2 news and sentiment...'
try {
  & $pyExe nbsi/phase2/scripts/run_phase2.py
  if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Phase-2 refresh failed; continuing with existing data...'
  }
} catch {
  Write-Warning "Phase-2 refresh error: $_; continuing with existing data..."
}

# 3) Refresh Phase-3 prices and pivots (always run to get latest prices)
Write-Host '[nightly] Refreshing Phase-3 prices and pivots...'
& $pyExe nbsi/phase3/scripts/fetch_prices_polygon.py
if ($LASTEXITCODE -ne 0) {
  Write-Error 'Phase-3 price fetch failed.'
  exit 1
}
& $pyExe nbsi/phase3/scripts/export_price_pivots.py --in artifacts/phase3/prices.parquet --out-root artifacts/phase3
if ($LASTEXITCODE -ne 0) {
  Write-Error 'Phase-3 pivot export failed.'
  exit 1
}

# Verify SPY is present in pivots
try {
  $py = "import pandas as pd, sys; df=pd.read_parquet('artifacts/phase3/closes.parquet'); sys.exit(0 if 'SPY' in df.columns else 1)"
  & $pyExe -c $py | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Write-Error 'SPY not found in Phase-3 pivots after refresh.'
    exit 1
  }
  Write-Host '[nightly] SPY verified in pivots.'
} catch {
  Write-Error 'Failed to verify SPY in pivots.'
  exit 1
}

# 4) Phase-4 simulate (paper sim, no orders)
Write-Host '[nightly] Phase-4 simulate...'
& $pyExe nbsi/phase4/scripts/run_phase4.py --mode simulate --dry-run false --from artifacts/phase3

# 5) Phase-4 route (DRY) and emit CSV for visibility
Write-Host '[nightly] Phase-4 route (DRY) + CSV...'
& $pyExe nbsi/phase4/scripts/run_phase4.py --mode route --dry-run true --from artifacts/phase3 --emit-csv true

# 6) Phase-4 route (SUBMIT to Alpaca paper)
Write-Host '[nightly] Phase-4 route (SUBMIT to Alpaca paper)...'
if (-not $env:ALPACA_KEY_ID -or -not $env:ALPACA_SECRET_KEY) {
  Write-Error 'ALPACA_KEY_ID/ALPACA_SECRET_KEY not set; cannot submit orders.'
  exit 3
}
& $pyExe nbsi/phase4/scripts/run_phase4.py --mode route --dry-run false --from artifacts/phase3 --confirm "I UNDERSTAND"


# 7) Phase-5 report (rotates logs first via script)
Write-Host '[nightly] Phase-5 report...'
& scripts/run_phase5_report.cmd

# 8) Verify artifacts and tail QA logs
$required = @(
  'artifacts/phase4/orders_intents.parquet',
  'artifacts/phase4/orders_intents.csv',
  'artifacts/phase4/orders_preview.parquet',
  'artifacts/phase4/orders_submitted.parquet',
  'artifacts/phase5/daily_equity.csv',
  'artifacts/phase5/equity_curve.png',
  'artifacts/phase5/daily_report.md'
)
$missing = @()
foreach ($p in $required) { if (-not (Test-Path $p)) { $missing += $p } }

Write-Host '--- QA tails (Phase-4) ---'
if (Test-Path 'artifacts/phase4/qa_phase4.log') { Get-Content 'artifacts/phase4/qa_phase4.log' -Tail 6 }
Write-Host '--- QA tails (Phase-5) ---'
if (Test-Path 'artifacts/phase5/qa_phase5.log') { Get-Content 'artifacts/phase5/qa_phase5.log' -Tail 6 }

if ($missing.Count -gt 0) {
  Write-Error ('Missing artifacts: ' + ($missing -join ', '))
  exit 2
}

# Optional: warn if QA PASS not visible
try {
  $p5_tail = Get-Content 'artifacts/phase5/qa_phase5.log' -Tail 200 -ErrorAction Stop
  $tailText = ($p5_tail -join "`n")
  if (-not ($tailText -match 'QA PASS')) {
    Write-Warning 'QA PASS not detected in Phase-5 tail; inspect logs above.'
  }
} catch {
  Write-Warning 'Phase-5 log not found for PASS check.'
}

Write-Host 'Nightly maintenance complete.'
exit 0

