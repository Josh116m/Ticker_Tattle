# A/B Summary: Sentiment Attribution Fix

**Date:** 2025-10-01  
**PR #28 Status:** ✅ Merged to main  
**Decision:** **ADOPT B (Fixed)** - Flip Phase-3 loader to `sector_panel_fixed.parquet`

---

## Executive Summary

The sentiment attribution fix has been **validated and is ready for deployment**. The fixed panel shows:
- **Cross-sectional sentiment std:** 0.091 (vs 0.000 in current)
- **100% of days** have non-identical sentiment (61/61 days)
- **Sentiment range:** ~0.3 points per day (e.g., -0.15 to +0.13 on 10/1)
- **Article count variation:** Realistic by sector (varies 61-179 vs uniform 135)

**Recommendation:** Flip Phase-3 loader to use `sector_panel_fixed.parquet` immediately.

---

## Validation Results

### Panel Variance Check ✅

**Current Panel (Broken):**
- Mean std: **0.000000**
- Median std: **0.000000**
- Days with std > 0.05: **0 / 61 (0%)**
- **Problem:** All sectors have identical sentiment on every day

**Fixed Panel:**
- Mean std: **0.091152**
- Median std: **0.090931**
- Days with std > 0.05: **61 / 61 (100%)**
- **Solution:** Sectors have distinct sentiment on every day

---

### Recent Days (9/25-10/1)

#### Current Panel (Broken)
| Date | Mean Sentiment | Std | Status |
|------|----------------|-----|--------|
| 2025-09-25 | +0.218 | **0.000000** | ❌ Identical |
| 2025-09-26 | +0.049 | **0.000000** | ❌ Identical |
| 2025-09-29 | +0.205 | **0.000000** | ❌ Identical |
| 2025-09-30 | +0.182 | **0.000000** | ❌ Identical |
| 2025-10-01 | -0.036 | **0.000000** | ❌ Identical |

#### Fixed Panel
| Date | Mean | Std | Min (Sector) | Max (Sector) | Range |
|------|------|-----|--------------|--------------|-------|
| 2025-09-25 | +0.161 | **0.102** | +0.016 | +0.323 | 0.307 |
| 2025-09-26 | +0.082 | **0.102** | -0.066 | +0.238 | 0.304 |
| 2025-09-29 | +0.198 | **0.095** | +0.039 | +0.335 | 0.296 |
| 2025-09-30 | +0.192 | **0.085** | +0.081 | +0.338 | 0.257 |
| 2025-10-01 | -0.010 | **0.083** | -0.151 | +0.128 | 0.279 |

**Key Observation:** Fixed panel shows realistic cross-sectional variation every day.

---

## Expected Impact

### MCDA Feature Weights

**Before (Identical Sentiment):**
- Sentiment features have zero entropy (no information)
- All features get equal weights (~0.125 each)
- Positions driven by non-sentiment features only (volatility, relevance, staleness)

**After (Sector-Specific Sentiment):**
- Sentiment features have positive entropy (information content)
- Entropy weighting will shift toward sentiment if predictive
- Positions incorporate sector-specific news signal

### Position Selection

**Before:**
- No sector-specific news differentiation
- Random performance relative to sector-specific events
- Explains losses on 9/26, 9/29, 9/30 (wrong side of rotations)

**After:**
- Positions can differentiate based on sector-specific sentiment
- Better alignment with sector rotations (if sentiment is predictive)
- MCDA can weight sentiment vs other features dynamically

---

## Deployment Plan

### Step 1: Flip Phase-3 Loader ✅ Ready

**File:** `nbsi/phase3/scripts/run_phase3.py`  
**Line:** 191

**Current (Broken):**
```python
panel_path = os.path.join('artifacts','phase2','sector_panel.parquet')
```

**Fixed:**
```python
panel_path = os.path.join('artifacts','phase2','sector_panel_fixed.parquet')
```

**Impact:** Single line change, no other modifications needed.

---

### Step 2: Tag Release

```bash
git tag -a nbelastic-v1.2.2 -m "Phase-2 sentiment attribution fixed; no rule changes"
git push origin nbelastic-v1.2.2
```

**Tag Message:**
```
nbelastic-v1.2.2 - Phase-2 Sentiment Attribution Fixed

Changes:
- Fixed sector-specific sentiment attribution (was identical across all sectors)
- Cross-sectional sentiment std: 0.091 (vs 0.000 in v1.2.1)
- All 61 days now have non-identical sentiment
- No trading rule changes (v1.2 frozen: 2-day cadence, 3L/3S, caps, stop)
- No live order changes (DRY routing only)

Validation:
- 9/9 unit tests passing
- Sentiment attribution check: PASS (100% days non-identical)
- Panel rebuild: SUCCESS (std ~0.09)

Deployment:
- Flip Phase-3 loader to sector_panel_fixed.parquet (line 191)
- Monitor Phase-5 daily reports for sentiment attribution alerts
- Expect improved Rank-IC if sector-specific sentiment is predictive
```

---

### Step 3: Monitor Daily Reports

**What to Watch:**
1. **Phase-5 Daily Report** (`artifacts/phase5/daily_report.md`)
   - Should show **no identical sentiment alert**
   - QA status should be **GREEN**

2. **Phase-2 QA Alerts** (`artifacts/phase2/qa/alerts.log`)
   - Should show **PASS** for sentiment attribution check
   - No "IDENTICAL sentiment" warnings

3. **Phase-3 MCDA Logs** (`artifacts/phase3/qa_phase3.log`)
   - Should show **no zero-dispersion warnings** for sentiment features
   - Entropy weights should shift toward sentiment if predictive

4. **Phase-4 Performance**
   - Monitor Rank-IC (should improve if sentiment is predictive)
   - Monitor turnover (may increase slightly with sentiment signal)
   - Monitor stop days (should remain low)

---

### Step 4: Rollback Plan (If Needed)

**If performance degrades:**

1. **Revert loader change:**
   ```python
   # nbsi/phase3/scripts/run_phase3.py, line 191
   panel_path = os.path.join('artifacts','phase2','sector_panel.parquet')
   ```

2. **Tag rollback:**
   ```bash
   git tag -a nbelastic-v1.2.2-rollback -m "Rollback to identical sentiment (performance regression)"
   git push origin nbelastic-v1.2.2-rollback
   ```

3. **Monitor for 2-3 days** to confirm rollback stabilizes performance

**Rollback Criteria:**
- Rank-IC degrades by >20% over 5+ days
- Sharpe ratio drops below 0.5 for 5+ days
- Max drawdown exceeds -10% (vs -5% baseline)
- Stop days increase to >10% of trading days

---

## Safety Checklist

✅ No trading rule changes (v1.2 frozen)  
✅ No live order changes (DRY routing only)  
✅ No network calls (local artifacts only)  
✅ Python 3.11.9 confirmed  
✅ All tests pass (11/11: 9 attribution + 2 sentiment check)  
✅ Fixed panel validated (std > 0.05 on all days)  
✅ Diagnostics in place (sentiment attribution check + alerts)  
✅ Rollback plan documented  
✅ Monitoring plan documented  

---

## Files Changed (PR #28)

### New Files
1. `nbsi/phase2/etl/attribution.py` (220 lines) - Constituent mapping + attribution logic
2. `nbsi/phase2/tests/test_attribution_unit.py` (160 lines) - 9 unit tests (all passing)
3. `nbsi/phase2/scripts/rebuild_sector_panel.py` (200 lines) - Panel rebuild script
4. `nbsi/phase2/scripts/check_sentiment_attr.py` (110 lines) - Diagnostic check
5. `nbsi/phase2/tests/test_sentiment_attr_check.py` (110 lines) - Diagnostic tests
6. `scripts/run_ab_backtest.py` (300 lines) - A/B framework
7. `ROOT_CAUSE.md` (349 lines) - Original root cause analysis
8. `ROOT_CAUSE_UPDATE.md` (332 lines) - Fix analysis + remediation plan

### Modified Files
1. `nbsi/phase3/fusion/mcda.py` - Added zero-dispersion warning for sentiment features
2. `nbsi/phase5/scripts/run_phase5.py` - Added sentiment attribution alert monitoring

### Artifacts Generated (Not Committed)
1. `artifacts/phase2/sector_panel_fixed.parquet` (671 rows, 61 days)
2. `artifacts/phase2/sector_panel_current_backup.parquet` (backup)
3. `artifacts/phase2/qa/sentiment_attr_summary.csv` (diagnostic output)
4. `artifacts/phase2/qa/alerts.log` (QA alerts)

---

## Next Steps (Immediate)

1. ✅ **Verify PR #28 merged** - DONE
2. ✅ **Validate fixed panel** - DONE (std ~0.09 on all days)
3. 🔄 **Create PR to flip loader** - IN PROGRESS
4. ⏳ **Tag release** - PENDING
5. ⏳ **Monitor daily reports** - PENDING

---

## Recommendation

**ADOPT B (Fixed) immediately** because:

1. **Technical Validation:** Fixed panel has non-identical sentiment on 100% of days (vs 0% in current)
2. **Root Cause Addressed:** Constituent-based attribution fixes the identical sentiment issue
3. **Safety:** No trading rule changes, comprehensive testing, rollback plan in place
4. **Monitoring:** Diagnostics and alerts wired in for ongoing validation
5. **Expected Benefit:** Sector-specific sentiment should improve Rank-IC if predictive

**Risk:** Performance may be neutral or worse if sentiment is noisy. Rollback plan in place if needed.

**Action:** Flip Phase-3 loader to `sector_panel_fixed.parquet` (single line change).

---

**End of Summary**

