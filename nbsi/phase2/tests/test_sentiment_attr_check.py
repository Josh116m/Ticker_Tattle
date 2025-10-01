"""
Unit test for sentiment attribution checker
"""
import os
import sys
import tempfile
import pandas as pd
import pytest
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nbsi.phase2.scripts.check_sentiment_attr import main as check_main


def test_sentiment_attr_identical():
    """Test that checker fails when all sectors have identical sentiment"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake panel with identical sentiment
        dates = pd.date_range('2025-09-01', '2025-09-10', freq='D')
        sectors = ['XLB', 'XLC', 'XLE']
        
        records = []
        for date in dates:
            for sector in sectors:
                records.append({
                    'date_et': date,
                    'sector': sector,
                    'mean_polarity': 0.5,  # IDENTICAL
                    'n_articles': 100  # IDENTICAL
                })
        
        panel = pd.DataFrame(records)
        panel_path = Path(tmpdir) / 'panel.parquet'
        panel.to_parquet(panel_path, index=False)
        
        # Run checker (should fail with exit code 2)
        out_dir = Path(tmpdir) / 'qa'
        
        sys.argv = [
            'check_sentiment_attr.py',
            '--panel', str(panel_path),
            '--last-k-days', '10',
            '--fail-threshold-pct', '80',
            '--out-dir', str(out_dir)
        ]
        
        with pytest.raises(SystemExit) as exc_info:
            check_main()
        
        assert exc_info.value.code == 2  # Should fail
        
        # Check outputs exist
        assert (out_dir / 'sentiment_attr_summary.csv').exists()
        assert (out_dir / 'alerts.log').exists()
        
        # Check alert content
        with open(out_dir / 'alerts.log') as f:
            content = f.read()
            assert 'FAIL' in content
            assert 'IDENTICAL' in content


def test_sentiment_attr_varied():
    """Test that checker passes when sectors have different sentiment"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake panel with varied sentiment
        dates = pd.date_range('2025-09-01', '2025-09-10', freq='D')
        sectors = ['XLB', 'XLC', 'XLE']
        
        records = []
        for date in dates:
            for i, sector in enumerate(sectors):
                records.append({
                    'date_et': date,
                    'sector': sector,
                    'mean_polarity': 0.1 * i,  # DIFFERENT
                    'n_articles': 100 + i * 10  # DIFFERENT
                })
        
        panel = pd.DataFrame(records)
        panel_path = Path(tmpdir) / 'panel.parquet'
        panel.to_parquet(panel_path, index=False)
        
        # Run checker (should pass with exit code 0)
        out_dir = Path(tmpdir) / 'qa'
        
        sys.argv = [
            'check_sentiment_attr.py',
            '--panel', str(panel_path),
            '--last-k-days', '10',
            '--fail-threshold-pct', '80',
            '--out-dir', str(out_dir)
        ]
        
        with pytest.raises(SystemExit) as exc_info:
            check_main()
        
        assert exc_info.value.code == 0  # Should pass
        
        # Check alert content
        with open(out_dir / 'alerts.log') as f:
            content = f.read()
            assert 'PASS' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

