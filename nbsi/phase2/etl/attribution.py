"""
Phase-2 Sentiment Attribution Module
Maps news articles to sector ETFs via constituent tickers or sector tags.
Excludes unassigned articles from sector aggregation.
"""
from typing import Optional, Dict, List, Set

# SPDR Sector ETF Constituent Mapping (representative top holdings)
# Source: SPDR sector ETF holdings (public data, simplified for attribution)
SECTOR_CONSTITUENTS: Dict[str, List[str]] = {
    'XLB': [  # Materials
        'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DD', 'DOW', 'ALB', 'PPG',
        'NUE', 'VMC', 'MLM', 'CTVA', 'IFF', 'CE', 'EMN', 'FMC', 'MOS', 'CF'
    ],
    'XLC': [  # Communications
        'META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS',
        'CHTR', 'EA', 'ATVI', 'TTWO', 'OMC', 'IPG', 'NWSA', 'NWS', 'FOXA', 'FOX', 'DISH'
    ],
    'XLE': [  # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'PXD', 'OXY',
        'WMB', 'KMI', 'HAL', 'BKR', 'HES', 'DVN', 'FANG', 'MRO', 'APA', 'CTRA'
    ],
    'XLF': [  # Financials
        'BRK.B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'SCHW', 'BLK', 'AXP',
        'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT', 'TROW', 'AFL', 'ALL', 'AIG'
    ],
    'XLI': [  # Industrials
        'UNP', 'HON', 'UPS', 'RTX', 'BA', 'CAT', 'GE', 'LMT', 'DE', 'MMM',
        'GD', 'NOC', 'ETN', 'EMR', 'ITW', 'CSX', 'NSC', 'WM', 'FDX', 'PCAR'
    ],
    'XLK': [  # Technology
        'AAPL', 'MSFT', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM', 'ACN', 'ORCL', 'AMD',
        'TXN', 'QCOM', 'INTC', 'IBM', 'INTU', 'NOW', 'AMAT', 'MU', 'ADI', 'LRCX'
    ],
    'XLP': [  # Consumer Staples
        'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
        'GIS', 'KHC', 'HSY', 'K', 'CLX', 'SJM', 'TSN', 'CAG', 'CPB', 'HRL'
    ],
    'XLRE': [  # Real Estate
        'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
        'EQR', 'VTR', 'SBAC', 'WY', 'ARE', 'INVH', 'ESS', 'MAA', 'UDR', 'EXR'
    ],
    'XLU': [  # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'WEC',
        'ES', 'PEG', 'EIX', 'AWK', 'DTE', 'PPL', 'FE', 'AEE', 'CMS', 'CNP'
    ],
    'XLV': [  # Healthcare
        'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'AMGN',
        'CVS', 'BMY', 'MDT', 'GILD', 'CI', 'ISRG', 'REGN', 'VRTX', 'ZTS', 'SYK'
    ],
    'XLY': [  # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG',
        'MAR', 'GM', 'F', 'ORLY', 'AZO', 'YUM', 'DHI', 'LEN', 'ROST', 'GPC'
    ],
}

# Build reverse mapping: ticker -> sector
TICKER_TO_SECTOR: Dict[str, str] = {}
for sector, tickers in SECTOR_CONSTITUENTS.items():
    for ticker in tickers:
        # Handle duplicates (e.g., GOOGL/GOOG both in XLC)
        if ticker not in TICKER_TO_SECTOR:
            TICKER_TO_SECTOR[ticker] = sector


def extract_tickers(text: str) -> Set[str]:
    """
    Extract potential ticker symbols from text (headline + summary).
    Simple heuristic: uppercase words 1-5 chars, common in financial news.
    """
    if not text:
        return set()
    
    # Split on whitespace and common delimiters
    words = text.upper().replace(',', ' ').replace('.', ' ').replace(':', ' ').split()
    
    # Filter for potential tickers (1-5 uppercase chars)
    tickers = set()
    for word in words:
        # Remove trailing punctuation
        word = word.strip('.,;:!?()')
        # Check if it's a valid ticker pattern
        if 1 <= len(word) <= 5 and word.isalpha() and word.isupper():
            tickers.add(word)
    
    return tickers


def assign_sector(article: Dict, ticker_to_sector: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Assign an article to a sector ETF based on:
    1. Tickers mentioned in headline/summary (mapped via constituent list)
    2. Explicit sector tag (if available)
    3. None if no match (excluded from sector aggregation)
    
    Args:
        article: Dict with keys like 'headline', 'summary', 'sector_tag' (optional)
        ticker_to_sector: Optional custom mapping (defaults to TICKER_TO_SECTOR)
    
    Returns:
        Sector ETF symbol (e.g., 'XLE') or None
    """
    if ticker_to_sector is None:
        ticker_to_sector = TICKER_TO_SECTOR
    
    # Strategy 1: Extract tickers from headline + summary
    text = ''
    if 'headline' in article:
        text += str(article['headline']) + ' '
    if 'summary' in article:
        text += str(article['summary'])
    
    tickers = extract_tickers(text)
    
    # Map tickers to sectors
    matched_sectors = set()
    for ticker in tickers:
        if ticker in ticker_to_sector:
            matched_sectors.add(ticker_to_sector[ticker])
    
    # If multiple sectors matched, pick the first (alphabetically for determinism)
    if matched_sectors:
        return sorted(matched_sectors)[0]
    
    # Strategy 2: Explicit sector tag (if available)
    if 'sector_tag' in article and article['sector_tag']:
        tag = str(article['sector_tag']).upper()
        if tag in SECTOR_CONSTITUENTS:
            return tag
    
    # No match: exclude from sector aggregation
    return None


def aggregate_sector_sentiment(articles: List[Dict], date_et: str) -> Dict[str, Dict]:
    """
    Aggregate sentiment per sector for a given date.
    
    Args:
        articles: List of article dicts with 'polarity', 'confidence', etc.
        date_et: Date string (YYYY-MM-DD)
    
    Returns:
        Dict[sector_etf, {mean_polarity, n_articles, conf_mean, ...}]
    """
    from collections import defaultdict
    import numpy as np
    
    # Group articles by sector
    sector_articles = defaultdict(list)
    
    for article in articles:
        sector = assign_sector(article)
        if sector:
            sector_articles[sector].append(article)
    
    # Aggregate per sector
    results = {}
    for sector, arts in sector_articles.items():
        polarities = [a.get('polarity', 0.0) for a in arts if 'polarity' in a]
        confidences = [a.get('confidence', 0.0) for a in arts if 'confidence' in a]
        
        results[sector] = {
            'date_et': date_et,
            'sector': sector,
            'n_articles': len(arts),
            'mean_polarity': np.mean(polarities) if polarities else 0.0,
            'conf_mean': np.mean(confidences) if confidences else 0.0,
        }
    
    return results


def get_sector_for_ticker(ticker: str) -> Optional[str]:
    """
    Get the sector ETF for a given ticker symbol.
    
    Args:
        ticker: Ticker symbol (e.g., 'AAPL')
    
    Returns:
        Sector ETF symbol (e.g., 'XLK') or None
    """
    return TICKER_TO_SECTOR.get(ticker.upper())


def get_constituents_for_sector(sector: str) -> List[str]:
    """
    Get the constituent tickers for a given sector ETF.
    
    Args:
        sector: Sector ETF symbol (e.g., 'XLK')
    
    Returns:
        List of ticker symbols
    """
    return SECTOR_CONSTITUENTS.get(sector.upper(), [])

