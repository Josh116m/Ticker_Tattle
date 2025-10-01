#!/usr/bin/env python
"""Quick check of panel variance to verify the fix"""
import pandas as pd
import numpy as np

print("="*80)
print("PANEL VARIANCE CHECK")
print("="*80)

# Current panel
print("\n📊 Current Panel (artifacts/phase2/sector_panel.parquet):")
panel_current = pd.read_parquet('artifacts/phase2/sector_panel.parquet')
print(f"  Shape: {panel_current.shape}")
print(f"  Date range: {panel_current['date_et'].min()} to {panel_current['date_et'].max()}")

print("\n  Last 5 days:")
for date in sorted(panel_current['date_et'].unique())[-5:]:
    day = panel_current[panel_current['date_et'] == date]
    std = day['mean_polarity'].std()
    mean = day['mean_polarity'].mean()
    print(f"    {date}: mean={mean:+.3f}, std={std:.6f}")

# Fixed panel
print("\n📊 Fixed Panel (artifacts/phase2/sector_panel_fixed.parquet):")
panel_fixed = pd.read_parquet('artifacts/phase2/sector_panel_fixed.parquet')
print(f"  Shape: {panel_fixed.shape}")
print(f"  Date range: {panel_fixed['date_et'].min()} to {panel_fixed['date_et'].max()}")

print("\n  Last 5 days:")
for date in sorted(panel_fixed['date_et'].unique())[-5:]:
    day = panel_fixed[panel_fixed['date_et'] == date]
    std = day['mean_polarity'].std()
    mean = day['mean_polarity'].mean()
    min_sent = day['mean_polarity'].min()
    max_sent = day['mean_polarity'].max()
    print(f"    {date}: mean={mean:+.3f}, std={std:.6f}, range=[{min_sent:+.3f}, {max_sent:+.3f}]")

# Overall statistics
print("\n📈 Overall Statistics:")
current_stds = []
fixed_stds = []

for date in panel_current['date_et'].unique():
    day_current = panel_current[panel_current['date_et'] == date]
    day_fixed = panel_fixed[panel_fixed['date_et'] == date]
    current_stds.append(day_current['mean_polarity'].std())
    fixed_stds.append(day_fixed['mean_polarity'].std())

print(f"\nCurrent Panel:")
print(f"  Mean std: {np.mean(current_stds):.6f}")
print(f"  Median std: {np.median(current_stds):.6f}")
print(f"  Days with std > 0.05: {sum(s > 0.05 for s in current_stds)} / {len(current_stds)}")

print(f"\nFixed Panel:")
print(f"  Mean std: {np.mean(fixed_stds):.6f}")
print(f"  Median std: {np.median(fixed_stds):.6f}")
print(f"  Days with std > 0.05: {sum(s > 0.05 for s in fixed_stds)} / {len(fixed_stds)}")

print("\n" + "="*80)
if np.mean(fixed_stds) > 0.05:
    print("✅ PASS: Fixed panel has non-identical sentiment (std > 0.05)")
else:
    print("❌ FAIL: Fixed panel still has identical sentiment")
print("="*80)

