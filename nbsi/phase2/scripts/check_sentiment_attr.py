#!/usr/bin/env python
"""
Part D1: Sentiment Attribution Diagnostic (Phase-2)
Checks if sentiment is identical across sectors (broken attribution).
Fails (exit code 2) if std==0 across >=X% of last K days.
Outputs: artifacts/phase2/qa/sentiment_attr_summary.csv and alerts.log
"""
import sys
import argparse
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='Check sentiment attribution across sectors')
    ap.add_argument('--panel', default='artifacts/phase2/sector_panel.parquet',
                    help='Path to sector panel parquet')
    ap.add_argument('--last-k-days', type=int, default=20,
                    help='Number of recent days to check')
    ap.add_argument('--fail-threshold-pct', type=float, default=80.0,
                    help='Fail if >=X%% of days have identical sentiment')
    ap.add_argument('--out-dir', default='artifacts/phase2/qa',
                    help='Output directory for summary and alerts')
    args = ap.parse_args()
    
    # Load panel
    panel = pd.read_parquet(args.panel)
    panel['date_et'] = pd.to_datetime(panel['date_et'])
    
    # Get last K unique dates
    unique_dates = sorted(panel['date_et'].unique())
    if len(unique_dates) > args.last_k_days:
        target_dates = unique_dates[-args.last_k_days:]
    else:
        target_dates = unique_dates
    
    # Analyze each date
    results = []
    identical_count = 0
    
    for date in target_dates:
        day_data = panel[panel['date_et'] == date]
        
        if len(day_data) == 0:
            continue
        
        # Compute cross-sectional std
        sentiment_std = day_data['mean_polarity'].std()
        articles_std = day_data['n_articles'].std()
        
        # Check if identical (use small epsilon for floating point)
        is_identical = (sentiment_std < 1e-10)
        
        if is_identical:
            identical_count += 1
        
        results.append({
            'date_et': date,
            'n_sectors': len(day_data),
            'sentiment_mean': day_data['mean_polarity'].mean(),
            'sentiment_std': sentiment_std,
            'articles_mean': day_data['n_articles'].mean(),
            'articles_std': articles_std,
            'identical': is_identical
        })
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Save summary
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = out_dir / 'sentiment_attr_summary.csv'
    df.to_csv(summary_path, index=False)
    
    # Compute failure condition
    total_days = len(df)
    identical_pct = (identical_count / total_days * 100) if total_days > 0 else 0.0
    
    # Write alert
    alerts_path = out_dir / 'alerts.log'
    with open(alerts_path, 'w') as f:
        if identical_pct >= args.fail_threshold_pct:
            identical_dates = df[df['identical']]['date_et'].dt.strftime('%Y-%m-%d').tolist()
            f.write(f"[sentiment-attr] FAIL: IDENTICAL sentiment across sectors on {identical_count}/{total_days} days ({identical_pct:.1f}%)\n")
            f.write(f"[sentiment-attr] Dates: {', '.join(identical_dates[:10])}\n")
            if len(identical_dates) > 10:
                f.write(f"[sentiment-attr] ... and {len(identical_dates)-10} more\n")
            status = 'FAIL'
            exit_code = 2
        else:
            f.write(f"[sentiment-attr] PASS: Sentiment varies across sectors ({identical_count}/{total_days} identical, {identical_pct:.1f}%)\n")
            status = 'PASS'
            exit_code = 0
    
    # Print summary
    print(f"Sentiment Attribution Check: {status}")
    print(f"  Analyzed: {total_days} days (last {args.last_k_days})")
    print(f"  Identical: {identical_count} days ({identical_pct:.1f}%)")
    print(f"  Threshold: {args.fail_threshold_pct}%")
    print(f"  Summary: {summary_path}")
    print(f"  Alerts: {alerts_path}")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

