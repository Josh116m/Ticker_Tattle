# Nightly Data Refresh Update

## Summary

The nightly maintenance script has been updated to **always refresh data daily** before running trading operations. This ensures you always have the latest news, sentiment, and price data for trading decisions.

## What Changed

### Before (Old Behavior)
- Only refreshed Phase-3 prices if SPY was missing from pivots
- Never refreshed Phase-2 news/sentiment data
- Could run with stale data from days ago

### After (New Behavior)
- **Always refreshes Phase-2 news and sentiment** (latest articles and FinBERT analysis)
- **Always refreshes Phase-3 prices and pivots** (latest market data)
- Verifies SPY is present in pivots after refresh
- Fails fast if data refresh fails (except Phase-2, which warns but continues)

## New Nightly Workflow

The updated `scripts/nightly_maintenance.ps1` now runs in this order:

1. **Check API Keys** - Verify POLYGON_API_KEY is set
2. **Refresh Phase-2 News** - Fetch latest news articles and run sentiment analysis
3. **Refresh Phase-3 Prices** - Fetch latest daily prices for SPY + 11 sector ETFs
4. **Export Price Pivots** - Create opens/closes parquet files
5. **Verify SPY** - Ensure SPY is present in the data
6. **Phase-4 Simulate** - Run backtest simulation
7. **Phase-4 Route (DRY)** - Generate order intents (dry run)
8. **Phase-4 Route (SUBMIT)** - Submit orders to Alpaca paper account
9. **Phase-5 Report** - Generate daily equity report and charts
10. **Verify & QA** - Check all artifacts exist and show QA logs

## SPY Status

✅ **SPY is present in your current data**
- Current Phase-3 data: 2025-07-07 to 2025-09-26
- SPY column exists in closes.parquet
- All 12 tickers present: SPY, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY

The script now explicitly verifies SPY after each refresh and will fail if it's missing.

## Testing the Updated Script

### Manual Test (Recommended)
Run the nightly maintenance script manually to verify it works:

```powershell
# Make sure you're in the repo root
cd "D:\Ticker Tattle"

# Run the updated script
powershell -ExecutionPolicy Bypass -File scripts\nightly_maintenance.ps1
```

**Expected behavior:**
- Phase-2 refresh will run (may take 5-10 minutes for news fetch + FinBERT)
- Phase-3 prices will refresh (1-2 minutes)
- SPY verification will pass
- Phase-4 and Phase-5 will run as before
- All QA logs should show "PASS"

### Quick Verification (Without Full Run)
Just test the data refresh steps:

```powershell
# Test Phase-3 price refresh only
python nbsi/phase3/scripts/fetch_prices_polygon.py
python nbsi/phase3/scripts/export_price_pivots.py --in artifacts/phase3/prices.parquet --out-root artifacts/phase3

# Verify SPY is present
python -c "import pandas as pd; df = pd.read_parquet('artifacts/phase3/closes.parquet'); print('SPY present:', 'SPY' in df.columns)"
```

## Scheduled Task

The scheduled task will automatically use the updated script starting with the next run:
- **Next Run:** 2025-10-01 05:45:00 AM (tomorrow morning)
- **Schedule:** Monday-Friday at 5:45 AM Central
- **No changes needed** - the task already points to the updated script

## Error Handling

### Phase-2 Refresh Failures
- If Phase-2 fails (e.g., API rate limit, network issue), the script will:
  - Log a warning
  - Continue with existing Phase-2 data
  - Still refresh Phase-3 prices
  - Complete the trading workflow

### Phase-3 Refresh Failures
- If Phase-3 fails, the script will:
  - Exit with error code 1
  - Not proceed to trading operations
  - This is intentional - we don't want to trade on stale prices

### SPY Verification Failures
- If SPY is missing after refresh, the script will:
  - Exit with error code 1
  - Not proceed to trading operations

## Performance Impact

**Estimated additional runtime:**
- Phase-2 news refresh: ~5-10 minutes (depends on news volume)
- Phase-3 price refresh: ~1-2 minutes (12 tickers)
- **Total added time:** ~6-12 minutes per run

The nightly task runs at 5:45 AM, well before market open at 9:30 AM ET, so this is not a concern.

## API Usage

**Polygon API calls per day:**
- News API: ~60-100 requests (depends on pagination)
- Prices API: 12 requests (one per ticker)
- **Total:** ~72-112 requests per day

This is well within Polygon's free tier limits (5 requests/minute, unlimited daily).

## Monitoring

Check these logs after each run:
- `artifacts/phase2/qa_phase2.log` - News fetch and sentiment QA
- `artifacts/phase3/qa_phase3.log` - Price data and model QA
- `artifacts/phase4/qa_phase4.log` - Trading simulation QA
- `artifacts/phase5/qa_phase5.log` - Report generation QA

All should show "QA PASS" or "PASS" messages.

## Rollback (If Needed)

If you need to revert to the old behavior:

```powershell
git revert HEAD
```

This will undo the changes and restore the conditional refresh logic.

## Next Steps

1. ✅ **Changes committed** - Script updated and committed to git
2. ⏳ **Test manually** - Run the script once to verify it works
3. ⏳ **Monitor tomorrow** - Check logs after tomorrow's 5:45 AM run
4. ⏳ **Verify data freshness** - Confirm Phase-2 and Phase-3 data are current

## Questions?

- **Why does Phase-2 take so long?** - FinBERT sentiment analysis runs on CPU/GPU for all articles
- **Can I skip Phase-2 refresh?** - Yes, comment out lines 28-37 in the script if needed
- **What if I hit API rate limits?** - Polygon has built-in rate limiting; the script will handle it
- **Will this use more API quota?** - Yes, but still well within free tier limits

---

**Status:** ✅ Ready for testing
**Commit:** 7a272ae5 - "feat(nightly): always refresh Phase-2 news and Phase-3 prices daily"

