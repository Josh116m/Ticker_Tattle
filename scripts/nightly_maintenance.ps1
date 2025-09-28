# Nightly maintenance (Windows, safe-by-default)
# - Assumes repo layout as in Ticker_Tattle and that Python 3.11.9 is on PATH
# - Runs from repo root automatically (regardless of where invoked)
# - No live orders: simulate + DRY route + daily report
# - Verifies required artifacts and tails QA logs

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Ensure we run from the repo root (script lives in scripts/)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Resolve-Path (Join-Path $scriptDir '..')
Set-Location $repoRoot

Write-Host ("[nightly] repoRoot=" + (Get-Location))

# 1) Ensure pivots exist and include SPY; refresh if missing/stale
$needPivots = $false
if (-not (Test-Path 'artifacts/phase3/closes.parquet')) {
  $needPivots = $true
} else {
  try {
    $py = "import pandas as pd, sys; df=pd.read_parquet('artifacts/phase3/closes.parquet'); sys.exit(0 if 'SPY' in df.columns else 1)"
    & python -c $py | Out-Null
    if ($LASTEXITCODE -ne 0) { $needPivots = $true }
  } catch {
    $needPivots = $true
  }
}

if ($needPivots) {
  if (-not $env:POLYGON_API_KEY) {
    Write-Error 'POLYGON_API_KEY not set; cannot refresh pivots.'
    exit 1
  }
  Write-Host '[nightly] Refreshing prices and pivots...'
  & python nbsi/phase3/scripts/fetch_prices_polygon.py
  & python nbsi/phase3/scripts/export_price_pivots.py --in artifacts/phase3/prices.parquet --out-root artifacts/phase3
}

# 2) Phase-4 simulate (paper sim, no orders)
Write-Host '[nightly] Phase-4 simulate...'
& python nbsi/phase4/scripts/run_phase4.py --mode simulate --dry-run false --from artifacts/phase3

# 3) Phase-4 route (DRY) and emit CSV for visibility
Write-Host '[nightly] Phase-4 route (DRY) + CSV...'
& python nbsi/phase4/scripts/run_phase4.py --mode route --dry-run true --from artifacts/phase3 --emit-csv true

# 4) Phase-5 report (rotates logs first via script)
Write-Host '[nightly] Phase-5 report...'
& scripts/run_phase5_report.cmd

# 5) Verify artifacts and tail QA logs
$required = @(
  'artifacts/phase4/orders_intents.parquet',
  'artifacts/phase4/orders_intents.csv',
  'artifacts/phase4/orders_preview.parquet',
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
  $p5_tail = Get-Content 'artifacts/phase5/qa_phase5.log' -Tail 50 -ErrorAction Stop
  if ($p5_tail -notmatch 'QA PASS') { Write-Warning 'QA PASS not detected in Phase-5 tail; inspect logs above.' }
} catch { Write-Warning 'Phase-5 log not found for PASS check.' }

Write-Host 'Nightly maintenance complete.'
exit 0

