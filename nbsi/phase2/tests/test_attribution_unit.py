"""
Unit tests for Phase-2 sentiment attribution module.
Tests sector-specific mapping via constituent tickers.
"""
import unittest
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nbsi.phase2.etl.attribution import (
    assign_sector,
    extract_tickers,
    aggregate_sector_sentiment,
    get_sector_for_ticker,
    SECTOR_CONSTITUENTS,
    TICKER_TO_SECTOR,
)


class TestAttribution(unittest.TestCase):
    
    def test_extract_tickers(self):
        """Test ticker extraction from text"""
        text = "Apple (AAPL) and Microsoft (MSFT) rally on earnings"
        tickers = extract_tickers(text)
        self.assertIn('AAPL', tickers)
        self.assertIn('MSFT', tickers)
    
    def test_assign_sector_via_ticker(self):
        """Test sector assignment via ticker mention"""
        # AAPL -> XLK (Technology)
        article = {
            'headline': 'Apple announces new iPhone',
            'summary': 'AAPL stock rises on product launch'
        }
        sector = assign_sector(article)
        self.assertEqual(sector, 'XLK')
        
        # XOM -> XLE (Energy)
        article = {
            'headline': 'Exxon Mobil reports strong earnings',
            'summary': 'XOM beats expectations'
        }
        sector = assign_sector(article)
        self.assertEqual(sector, 'XLE')
    
    def test_assign_sector_no_match(self):
        """Test that generic articles return None"""
        article = {
            'headline': 'Market overview for today',
            'summary': 'Stocks mixed in early trading'
        }
        sector = assign_sector(article)
        self.assertIsNone(sector)
    
    def test_assign_sector_explicit_tag(self):
        """Test sector assignment via explicit tag"""
        article = {
            'headline': 'Healthcare sector update',
            'summary': 'No specific tickers mentioned',
            'sector_tag': 'XLV'
        }
        sector = assign_sector(article)
        self.assertEqual(sector, 'XLV')
    
    def test_aggregate_distinct_sectors(self):
        """Test that aggregation produces non-identical sentiment across sectors"""
        # Create articles for 3 distinct sectors with different sentiment
        articles = [
            # Technology (positive)
            {'headline': 'AAPL soars', 'polarity': 0.8, 'confidence': 0.9},
            {'headline': 'MSFT rallies', 'polarity': 0.7, 'confidence': 0.85},
            {'headline': 'NVDA hits record', 'polarity': 0.9, 'confidence': 0.95},
            
            # Energy (negative)
            {'headline': 'XOM declines', 'polarity': -0.6, 'confidence': 0.8},
            {'headline': 'CVX falls', 'polarity': -0.5, 'confidence': 0.75},
            
            # Healthcare (neutral)
            {'headline': 'JNJ steady', 'polarity': 0.1, 'confidence': 0.7},
            {'headline': 'PFE unchanged', 'polarity': -0.1, 'confidence': 0.65},
        ]
        
        result = aggregate_sector_sentiment(articles, '2025-09-30')
        
        # Should have 3 sectors
        self.assertEqual(len(result), 3)
        self.assertIn('XLK', result)  # Technology
        self.assertIn('XLE', result)  # Energy
        self.assertIn('XLV', result)  # Healthcare
        
        # Check sentiment is different across sectors
        xlk_sentiment = result['XLK']['mean_polarity']
        xle_sentiment = result['XLE']['mean_polarity']
        xlv_sentiment = result['XLV']['mean_polarity']
        
        # Technology should be positive
        self.assertGreater(xlk_sentiment, 0.5)
        
        # Energy should be negative
        self.assertLess(xle_sentiment, -0.4)
        
        # Healthcare should be near zero
        self.assertLess(abs(xlv_sentiment), 0.2)
        
        # Verify they're all different
        self.assertNotEqual(xlk_sentiment, xle_sentiment)
        self.assertNotEqual(xlk_sentiment, xlv_sentiment)
        self.assertNotEqual(xle_sentiment, xlv_sentiment)
        
        # Verify article counts
        self.assertEqual(result['XLK']['n_articles'], 3)
        self.assertEqual(result['XLE']['n_articles'], 2)
        self.assertEqual(result['XLV']['n_articles'], 2)
    
    def test_cross_sectional_variance(self):
        """Test that cross-sectional variance is non-zero"""
        import numpy as np
        
        # Create articles with sector-specific sentiment
        articles = []
        
        # Each sector gets different sentiment
        sector_sentiments = {
            'XLK': 0.8,   # Tech positive
            'XLE': -0.6,  # Energy negative
            'XLV': 0.2,   # Healthcare neutral
            'XLF': -0.3,  # Financials slightly negative
            'XLI': 0.5,   # Industrials positive
        }
        
        for sector, sentiment in sector_sentiments.items():
            # Get a ticker from this sector
            ticker = SECTOR_CONSTITUENTS[sector][0]
            for i in range(5):  # 5 articles per sector
                articles.append({
                    'headline': f'{ticker} news {i}',
                    'polarity': sentiment + (i - 2) * 0.05,  # Add small variation
                    'confidence': 0.8
                })
        
        result = aggregate_sector_sentiment(articles, '2025-09-30')
        
        # Extract sentiment values
        sentiments = [result[s]['mean_polarity'] for s in result.keys()]
        
        # Compute cross-sectional std
        std = np.std(sentiments)
        
        # Should be non-zero (not identical)
        self.assertGreater(std, 0.1, "Cross-sectional std should be > 0.1")
    
    def test_get_sector_for_ticker(self):
        """Test ticker-to-sector lookup"""
        self.assertEqual(get_sector_for_ticker('AAPL'), 'XLK')
        self.assertEqual(get_sector_for_ticker('XOM'), 'XLE')
        self.assertEqual(get_sector_for_ticker('JNJ'), 'XLV')
        self.assertIsNone(get_sector_for_ticker('UNKNOWN'))
    
    def test_constituent_mapping_complete(self):
        """Test that all 11 sectors have constituents"""
        expected_sectors = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
        
        for sector in expected_sectors:
            self.assertIn(sector, SECTOR_CONSTITUENTS)
            self.assertGreater(len(SECTOR_CONSTITUENTS[sector]), 0)
    
    def test_ticker_to_sector_reverse_mapping(self):
        """Test that reverse mapping is built correctly"""
        # Check a few known mappings
        self.assertEqual(TICKER_TO_SECTOR.get('AAPL'), 'XLK')
        self.assertEqual(TICKER_TO_SECTOR.get('XOM'), 'XLE')
        self.assertEqual(TICKER_TO_SECTOR.get('JPM'), 'XLF')
        
        # Should have many tickers mapped
        self.assertGreater(len(TICKER_TO_SECTOR), 100)


if __name__ == '__main__':
    unittest.main()

