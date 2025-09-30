#!/usr/bin/env python
"""
A/B Backtest: Current vs Fixed Sentiment Attribution
Runs Phase-3 → Phase-4 → Phase-5 for both variants and compares results.
"""
import os
import sys
import shutil
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_phase3(variant: str, panel_path: str, output_dir: str):
    """Run Phase-3 with specified panel"""
    print(f"\n{'='*80}")
    print(f"Running Phase-3 ({variant})")
    print(f"{'='*80}")
    
    # Temporarily swap the panel
    original_panel = Path('artifacts/phase2/sector_panel.parquet')
    backup_panel = Path('artifacts/phase2/sector_panel_backup.parquet')
    
    # Backup original
    if original_panel.exists() and not backup_panel.exists():
        shutil.copy(original_panel, backup_panel)
    
    # Copy the variant panel to the expected location
    shutil.copy(panel_path, original_panel)
    
    try:
        # Run Phase-3
        cmd = [
            'python', 'nbsi/phase3/scripts/run_phase3.py',
            '--config', 'nbsi/phase3/configs/config.yaml',
            '--output-dir', output_dir
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    finally:
        # Restore original panel
        if backup_panel.exists():
            shutil.copy(backup_panel, original_panel)
    
    print(f"✅ Phase-3 ({variant}) complete")


def run_phase4(variant: str, phase3_dir: str, output_dir: str):
    """Run Phase-4 simulate and route (DRY)"""
    print(f"\n{'='*80}")
    print(f"Running Phase-4 ({variant})")
    print(f"{'='*80}")
    
    # Run simulate
    cmd = [
        'python', 'nbsi/phase4/scripts/run_phase4.py',
        '--mode', 'simulate',
        '--from', phase3_dir,
        '--out', output_dir
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    
    # Run route (DRY)
    cmd = [
        'python', 'nbsi/phase4/scripts/run_phase4.py',
        '--mode', 'route',
        '--dry-run', 'true',
        '--from', phase3_dir,
        '--out', output_dir,
        '--emit-csv', 'true'
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    
    print(f"✅ Phase-4 ({variant}) complete")


def run_phase5(variant: str, phase4_dir: str, output_dir: str):
    """Run Phase-5 reporting"""
    print(f"\n{'='*80}")
    print(f"Running Phase-5 ({variant})")
    print(f"{'='*80}")
    
    cmd = [
        'python', 'nbsi/phase5/scripts/run_phase5.py',
        '--from', phase4_dir,
        '--out', output_dir,
        '--phase3-root', 'artifacts/phase3'
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    
    print(f"✅ Phase-5 ({variant}) complete")


def compare_results(a_dir: str, b_dir: str, output_dir: str):
    """Compare A vs B results and generate summary"""
    print(f"\n{'='*80}")
    print("Comparing A vs B Results")
    print(f"{'='*80}")
    
    results = {}
    
    # Load Phase-3 metrics
    for variant, base_dir in [('A', a_dir), ('B', b_dir)]:
        # Try to find metrics in rank/entropy subdirectory
        metrics_path = Path(base_dir) / 'rank' / 'entropy' / 'metrics.json'
        if not metrics_path.exists():
            # Try root directory
            metrics_path = Path(base_dir) / 'metrics.json'
        
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            results[variant] = {'phase3': metrics}
        else:
            print(f"⚠️  Warning: No metrics found for {variant} at {metrics_path}")
            results[variant] = {'phase3': {}}
    
    # Load Phase-4 exec_summary
    for variant, base_dir in [('A', a_dir.replace('phase3', 'phase4')), 
                               ('B', b_dir.replace('phase3', 'phase4'))]:
        exec_path = Path(base_dir) / 'exec_summary.json'
        if exec_path.exists():
            with open(exec_path) as f:
                exec_sum = json.load(f)
            results[variant.replace('phase3', 'phase4').split('/')[0]]['phase4'] = exec_sum
        else:
            print(f"⚠️  Warning: No exec_summary found for {variant}")
    
    # Load Phase-5 equity curves
    equity_a = pd.read_csv(Path(a_dir.replace('phase3', 'phase5')) / 'daily_equity.csv')
    equity_b = pd.read_csv(Path(b_dir.replace('phase3', 'phase5')) / 'daily_equity.csv')
    
    # Create comparison table
    comparison = {
        'Metric': [],
        'A (Current)': [],
        'B (Fixed)': [],
        'Winner': []
    }
    
    # Phase-3 metrics
    if 'A' in results and 'B' in results:
        # Rank-IC
        rank_ic_a = results['A']['phase3'].get('rank_ic_mean', 0)
        rank_ic_b = results['B']['phase3'].get('rank_ic_mean', 0)
        comparison['Metric'].append('Rank-IC (mean)')
        comparison['A (Current)'].append(f"{rank_ic_a:.4f}")
        comparison['B (Fixed)'].append(f"{rank_ic_b:.4f}")
        comparison['Winner'].append('B' if rank_ic_b > rank_ic_a else 'A')
        
        # Turnover
        turnover_a = results['A']['phase3'].get('turnover_mean', 0)
        turnover_b = results['B']['phase3'].get('turnover_mean', 0)
        comparison['Metric'].append('Turnover (mean)')
        comparison['A (Current)'].append(f"{turnover_a:.4f}")
        comparison['B (Fixed)'].append(f"{turnover_b:.4f}")
        comparison['Winner'].append('A' if turnover_a < turnover_b else 'B')  # Lower is better
    
    # Phase-4 metrics
    if 'A' in results and 'phase4' in results['A'] and 'B' in results and 'phase4' in results['B']:
        avg_gross_a = results['A']['phase4'].get('avg_gross', 0)
        avg_gross_b = results['B']['phase4'].get('avg_gross', 0)
        comparison['Metric'].append('Avg Gross Exposure')
        comparison['A (Current)'].append(f"{avg_gross_a:.4f}")
        comparison['B (Fixed)'].append(f"{avg_gross_b:.4f}")
        comparison['Winner'].append('-')
        
        stop_days_a = results['A']['phase4'].get('stop_days', 0)
        stop_days_b = results['B']['phase4'].get('stop_days', 0)
        comparison['Metric'].append('Stop Days')
        comparison['A (Current)'].append(f"{stop_days_a}")
        comparison['B (Fixed)'].append(f"{stop_days_b}")
        comparison['Winner'].append('A' if stop_days_a < stop_days_b else 'B')  # Lower is better
    
    # Phase-5 equity
    final_equity_a = equity_a['equity'].iloc[-1]
    final_equity_b = equity_b['equity'].iloc[-1]
    comparison['Metric'].append('Final Equity')
    comparison['A (Current)'].append(f"{final_equity_a:.4f}")
    comparison['B (Fixed)'].append(f"{final_equity_b:.4f}")
    comparison['Winner'].append('B' if final_equity_b > final_equity_a else 'A')
    
    # Compute returns
    ret_a = equity_a['ret'].values
    ret_b = equity_b['ret'].values
    
    # Sharpe ratio
    sharpe_a = np.mean(ret_a) / np.std(ret_a) * np.sqrt(252) if np.std(ret_a) > 0 else 0
    sharpe_b = np.mean(ret_b) / np.std(ret_b) * np.sqrt(252) if np.std(ret_b) > 0 else 0
    comparison['Metric'].append('Sharpe Ratio')
    comparison['A (Current)'].append(f"{sharpe_a:.4f}")
    comparison['B (Fixed)'].append(f"{sharpe_b:.4f}")
    comparison['Winner'].append('B' if sharpe_b > sharpe_a else 'A')
    
    # Create DataFrame
    df_comparison = pd.DataFrame(comparison)
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / 'ab_summary.csv'
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n✅ Saved comparison to {csv_path}")
    
    # Print table
    print("\n" + "="*80)
    print("A/B COMPARISON SUMMARY")
    print("="*80)
    print(df_comparison.to_string(index=False))
    
    # Create equity overlay plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pd.to_datetime(equity_a['date_et']), equity_a['equity'], label='A (Current)', linewidth=2)
    ax.plot(pd.to_datetime(equity_b['date_et']), equity_b['equity'], label='B (Fixed)', linewidth=2, linestyle='--')
    ax.set_title('A/B Equity Curve Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date (ET)')
    ax.set_ylabel('Equity (start=1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    
    png_path = output_path / 'ab_equity_overlay.png'
    fig.savefig(png_path, dpi=144)
    plt.close(fig)
    print(f"✅ Saved equity overlay to {png_path}")
    
    # Determine winner
    b_wins = sum(1 for w in comparison['Winner'] if w == 'B')
    a_wins = sum(1 for w in comparison['Winner'] if w == 'A')
    
    print(f"\n{'='*80}")
    print(f"WINNER: {'B (Fixed)' if b_wins > a_wins else 'A (Current)' if a_wins > b_wins else 'TIE'}")
    print(f"  B wins: {b_wins} metrics")
    print(f"  A wins: {a_wins} metrics")
    print(f"{'='*80}")
    
    return df_comparison


def main():
    # Define paths
    panel_a = 'artifacts/phase2/sector_panel.parquet'  # Current (broken)
    panel_b = 'artifacts/phase2/sector_panel_fixed.parquet'  # Fixed
    
    phase3_a = 'artifacts/phase3/A'
    phase3_b = 'artifacts/phase3/B'
    
    phase4_a = 'artifacts/phase4/A'
    phase4_b = 'artifacts/phase4/B'
    
    phase5_a = 'artifacts/phase5/A'
    phase5_b = 'artifacts/phase5/B'
    
    # Run A (Current)
    run_phase3('A', panel_a, phase3_a)
    run_phase4('A', phase3_a, phase4_a)
    run_phase5('A', phase4_a, phase5_a)
    
    # Run B (Fixed)
    run_phase3('B', panel_b, phase3_b)
    run_phase4('B', phase3_b, phase4_b)
    run_phase5('B', phase4_b, phase5_b)
    
    # Compare results
    compare_results(phase3_a, phase3_b, 'artifacts/phase5/diag')
    
    print("\n" + "="*80)
    print("A/B BACKTEST COMPLETE")
    print("="*80)
    print("Results:")
    print(f"  Phase-3 A: {phase3_a}")
    print(f"  Phase-3 B: {phase3_b}")
    print(f"  Phase-4 A: {phase4_a}")
    print(f"  Phase-4 B: {phase4_b}")
    print(f"  Phase-5 A: {phase5_a}")
    print(f"  Phase-5 B: {phase5_b}")
    print(f"  Summary: artifacts/phase5/diag/ab_summary.csv")
    print(f"  Overlay: artifacts/phase5/diag/ab_equity_overlay.png")


if __name__ == '__main__':
    main()

