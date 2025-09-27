#!/usr/bin/env python3
"""
NBElastic Phase‑0 Diagnostic Script
===================================

This script performs the phase‑0 diagnostic outlined in the NBElastic
runbook.  It pulls news and price data from Polygon.io, fundamentals
and earnings data from Finnhub, computes a number of quality metrics
and writes out several CSV reports plus a summary markdown file.

The diagnostics validate the assumptions about news coverage,
diversity, staleness, deduplication, and the viability of a SPY
sentiment proxy.  The results are intended to guide whether the
system can proceed with the remaining phases or if fallback rules
should be engaged.

Usage:
    python diagnostic_phase0.py --config config.yaml --output-dir ./phase0_reports

The script expects a YAML configuration with API keys and
parameters (see nbsi_phase0/config.yaml for an example).  All
outputs will be written into the specified output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import threading
import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Any, Dict, Iterable, List, Optional, Tuple
from email.utils import parsedate_to_datetime

import pandas as pd
import numpy as np
import requests
import yaml

from utils import (
    deduplicate_articles,
    realised_volatility,
    to_eastern,
)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


######################################################################
# Configuration dataclasses
######################################################################


@dataclass
class Config:
    polygon_api_key: str
    finnhub_api_key: str
    news_lookback_days: int = 20
    news_per_page: int = 100
    prices_lookback_days: int = 90
    spy_top_n: int = 50
    spy_min_articles: int = 8
    finnhub_rate_limit: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)

        # Default rate limit configuration
        default_rate_limit = {
            "max_per_second": 25,
            "bucket_capacity": 30,
            "backoff_base_seconds": 0.5,
            "backoff_factor": 2.0,
            "backoff_max_seconds": 60
        }

        return Config(
            polygon_api_key=data["polygon_api_key"],
            finnhub_api_key=data["finnhub_api_key"],
            news_lookback_days=data.get("news", {}).get("lookback_days", 20),
            news_per_page=data.get("news", {}).get("per_page", 100),
            prices_lookback_days=data.get("prices", {}).get("lookback_days", 90),
            spy_top_n=data.get("spy", {}).get("top_n", 50),
            spy_min_articles=data.get("spy", {}).get("min_articles", 8),
            finnhub_rate_limit=data.get("finnhub_rate_limit", default_rate_limit),
        )


######################################################################
# Helper classes
######################################################################


class PolygonClient:
    """Minimal client for the Polygon.io REST API."""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        params["apiKey"] = self.api_key
        url = f"{self.BASE_URL}{path}"
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            logging.error("Polygon API error %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        return resp.json()

    def fetch_news(self, start_date: str, end_date: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch all news articles between start_date and end_date (inclusive).

        The API is paginated via next_url.  This method will iterate
        through all pages and return a list of article dictionaries.
        """
        logging.info("Fetching news from %s to %s", start_date, end_date)
        articles = []
        params = {
            "published_utc.gte": start_date,
            "published_utc.lte": end_date,
            "order": "asc",
            "limit": limit,
        }
        url_path = "/v2/reference/news"
        while True:
            if url_path.startswith("http"):
                # This is a full URL from next_url, add API key and make direct request
                separator = "&" if "?" in url_path else "?"
                url_with_key = f"{url_path}{separator}apiKey={self.api_key}"
                resp = requests.get(url_with_key, timeout=30)
                if resp.status_code != 200:
                    logging.error("Polygon API error %s: %s", resp.status_code, resp.text)
                    resp.raise_for_status()
                data = resp.json()
            else:
                # This is a path, use the normal get method
                data = self.get(url_path, params=params)

            results = data.get("results", [])
            articles.extend(results)
            next_url = data.get("next_url")
            if not next_url:
                break
            # next_url is a full URL, use it directly
            url_path = next_url
            params = None
        logging.info("Fetched %d news articles", len(articles))
        return articles

    def fetch_aggregates(self, ticker: str, timespan: str, multiplier: int, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch aggregate bars for a given ticker.

        Args:
            ticker: The ticker symbol (e.g. "SPY").
            timespan: e.g. "day".
            multiplier: e.g. 1 for daily bars.
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            A list of bar dictionaries.
        """
        logging.info("Fetching aggregates for %s from %s to %s", ticker, from_date, to_date)
        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        data = self.get(path, params)
        results = data.get("results", [])
        return results


class TokenBucket:
    """Thread-safe token bucket for rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful, False if not enough tokens."""
        with self.lock:
            now = time.time()
            # Add tokens based on time elapsed
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_for_tokens(self, tokens: int = 1) -> float:
        """Calculate how long to wait for tokens to be available."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            available_tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

            if available_tokens >= tokens:
                return 0.0

            needed_tokens = tokens - available_tokens
            wait_time = needed_tokens / self.refill_rate
            return wait_time


class FinnhubForbidden(Exception):
    pass


class FinnhubClient:
    """Rate-limited client for the Finnhub API with caching and robust error handling."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str, rate_limit_config: Optional[Dict[str, Any]] = None):
        self.api_key = api_key
        self.session = requests.Session()

        # Rate limiting configuration
        config = rate_limit_config or {}
        self.max_per_second = config.get("max_per_second", 25)
        bucket_capacity = config.get("bucket_capacity", 30)
        self.backoff_base = config.get("backoff_base_seconds", 0.5)
        self.backoff_factor = config.get("backoff_factor", 2.0)
        self.backoff_max = config.get("backoff_max_seconds", 60)

        # Token bucket for rate limiting
        self.token_bucket = TokenBucket(bucket_capacity, self.max_per_second)

        # In-memory cache for responses
        self.cache = {}

        # Track disabled endpoints (e.g., due to 403 errors)
        self.disabled_endpoints = set()

        # Request statistics
        self.request_count = 0
        self.start_time = time.time()
        self.sec_epoch = int(time.time())
        self.sec_count = 0

    def _make_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Create a cache key from endpoint and sorted parameters."""
        # Exclude token from cache key
        params_wo_token = {k: v for k, v in params.items() if k != "token"}
        sorted_params = sorted(params_wo_token.items())
        param_str = "&".join(f"{k}={v}" for k, v in sorted_params)
        cache_input = f"{endpoint}?{param_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _parse_retry_after(self, retry_after_header: str) -> float:
        """Parse Retry-After header (seconds or HTTP date)."""
        try:
            # Try parsing as seconds first
            return float(retry_after_header)
        except ValueError:
            try:
                # Try parsing as HTTP date
                retry_date = parsedate_to_datetime(retry_after_header)
                return max(0, (retry_date - datetime.now(retry_date.tzinfo)).total_seconds())
            except (ValueError, TypeError):
                return 0.0

    def _rate_limited_request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a rate-limited request with exponential backoff and caching."""
        params = params or {}
        params["token"] = self.api_key

        # Check if endpoint is disabled
        if path in self.disabled_endpoints:
            raise FinnhubForbidden(f"Endpoint {path} is disabled due to previous 403 error")

        # Check cache first
        cache_key = self._make_cache_key(path, params)
        if cache_key in self.cache:
            return self.cache[cache_key]

        url = f"{self.BASE_URL}{path}"
        attempt = 0
        backoff_time = self.backoff_base

        # For logging: params hash (exclude token)
        params_hash = self._make_cache_key(path, params)

        while attempt < 5:  # Max 5 attempts
            # Wait for rate limit token
            if not self.token_bucket.consume():
                wait_time = self.token_bucket.wait_for_tokens()
                if wait_time > 0:
                    logging.info(f"Rate limit: waiting {wait_time:.2f}s for token bucket")
                    time.sleep(wait_time)
                    self.token_bucket.consume()  # Consume after waiting

            # Per-second call rate logging
            now_sec = int(time.time())
            if now_sec != self.sec_epoch:
                logging.info(f"Finnhub call rate: {self.sec_count} req/s (sec epoch {self.sec_epoch})")
                self.sec_epoch = now_sec
                self.sec_count = 0

            try:
                self.request_count += 1
                self.sec_count += 1

                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 200:
                    data = resp.json()
                    # Cache successful response
                    self.cache[cache_key] = data
                    return data

                elif resp.status_code == 403:
                    # Disable this endpoint permanently for this run and short-circuit
                    self.disabled_endpoints.add(path)
                    logging.warning(f"Endpoint {path} returned 403, disabling for this run")
                    raise FinnhubForbidden(f"403 Forbidden: {resp.text}")

                elif resp.status_code == 429:
                    # Rate limited - use Retry-After if available
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        wait_time = self._parse_retry_after(retry_after)
                        logging.warning(f"429 on {path} (params_hash={params_hash}), Retry-After: {retry_after}, sleeping {wait_time:.2f}s (attempt {attempt + 1})")
                    else:
                        wait_time = backoff_time + random.uniform(0, backoff_time * 0.1)  # Add jitter
                        logging.warning(f"429 on {path} (params_hash={params_hash}), sleeping {wait_time:.2f}s (attempt {attempt + 1})")

                    time.sleep(wait_time)
                    backoff_time = min(self.backoff_max, backoff_time * self.backoff_factor)
                    attempt += 1
                    continue

                elif resp.status_code >= 500:
                    # Server error - exponential backoff
                    wait_time = backoff_time + random.uniform(0, backoff_time * 0.1)
                    logging.warning(f"Server error {resp.status_code} on {path}, sleeping {wait_time:.2f}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    backoff_time = min(self.backoff_max, backoff_time * self.backoff_factor)
                    attempt += 1
                    continue

                else:
                    # Other client errors
                    logging.error("Finnhub API error %s: %s", resp.status_code, resp.text)
                    resp.raise_for_status()

            except FinnhubForbidden:
                # Short-circuit retries for forbidden endpoints
                raise
            except requests.exceptions.RequestException as e:
                if attempt == 4:  # Last attempt
                    raise
                wait_time = backoff_time + random.uniform(0, backoff_time * 0.1)
                logging.warning(f"Request exception on {path}: {e}, sleeping {wait_time:.2f}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                backoff_time = min(self.backoff_max, backoff_time * self.backoff_factor)
                attempt += 1

        raise requests.exceptions.HTTPError(f"Failed to complete request to {path} after 5 attempts")

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request with rate limiting and error handling."""
        return self._rate_limited_request(path, params)

    def fetch_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch basic company profile (sector/industry) for the given symbol."""
        try:
            data = self.get("/stock/profile2", params={"symbol": symbol})
            return data
        except Exception as e:
            logging.warning("Failed to fetch profile for %s: %s", symbol, e)
            return None

    def fetch_etf_holdings(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch holdings for an ETF.

        This is used to derive the top constituents of SPY for the market
        sentiment proxy.  If the request fails the function returns an
        empty list.
        """
        try:
            data = self.get("/etf/holdings", params={"symbol": symbol})
            return data.get("holdings", [])
        except Exception as e:
            logging.warning("Failed to fetch ETF holdings for %s: %s", symbol, e)
            return []

    def fetch_earnings_calendar(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch earnings calendar between two dates."""
        try:
            data = self.get("/calendar/earnings", params={"from": start_date, "to": end_date})
            return data.get("earningsCalendar", [])
        except Exception as e:
            logging.warning("Failed to fetch earnings calendar: %s", e)
            return []


######################################################################
# Diagnostic Functions
######################################################################


def last_trading_days(n: int, end_date: date) -> List[date]:
    """Return the last n trading days up to end_date (inclusive).

    This helper assumes US markets are closed on weekends but does not
    account for holidays.  If the date falls on a weekend, it rolls
    backward to the previous Friday.

    Args:
        n: Number of trading days to return.
        end_date: The last date in the range.

    Returns:
        A list of date objects in ascending order.
    """
    days = []
    cur = end_date
    while len(days) < n:
        # Adjust for weekend
        while cur.weekday() >= 5:  # Saturday=5, Sunday=6
            cur = cur - timedelta(days=1)
        days.append(cur)
        cur = cur - timedelta(days=1)
    return sorted(days)


def parse_articles(raw_articles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Normalize raw news articles into a DataFrame.

    Args:
        raw_articles: List of article dictionaries as returned by Polygon.

    Returns:
        A DataFrame with columns: article_id, published_utc (datetime),
        source, title, description, tickers (list), url.
    """
    records = []
    for art in raw_articles:
        article_id = art.get("id") or art.get("article_id")
        ts = art.get("published_utc") or art.get("published_at")
        try:
            dt = pd.to_datetime(ts, utc=True)
        except Exception:
            continue
        source = art.get("publisher", {}).get("name") or art.get("source") or art.get("amp_url") or ""
        title = art.get("title") or ""
        description = art.get("description") or art.get("article_body") or ""
        tickers = art.get("tickers") or art.get("symbols") or []
        url = art.get("amp_url") or art.get("url") or ""
        records.append(
            {
                "article_id": article_id,
                "published_utc": dt,
                "source": source,
                "title": title,
                "description": description,
                "tickers": tickers,
                "url": url,
            }
        )
    df = pd.DataFrame.from_records(records)
    return df


def build_sector_mapping(tickers: Iterable[str], fh: FinnhubClient) -> Dict[str, str]:
    """Map tickers to sectors using Finnhub profiles.

    Args:
        tickers: Iterable of ticker symbols.
        fh: FinnhubClient instance.

    Returns:
        Dictionary mapping ticker symbol to sector string (uppercased).
    """
    mapping: Dict[str, str] = {}
    for ticker in set(tickers):
        if not ticker:
            continue
        profile = fh.fetch_company_profile(ticker)
        if profile and profile.get("finnhubIndustry"):
            sector = profile["finnhubIndustry"].upper()
        elif profile and profile.get("sector"):
            sector = profile["sector"].upper()
        else:
            sector = "UNKNOWN"
        mapping[ticker] = sector
    return mapping


def assign_sectors_to_articles(df: pd.DataFrame, ticker_sector_map: Dict[str, str]) -> pd.DataFrame:
    """Assign sectors to articles based on tickers.

    An article may be associated with multiple tickers; this function
    derives the set of sectors for each article.  Articles with no
    recognised tickers receive an empty list of sectors.

    Args:
        df: DataFrame with a 'tickers' column containing lists of tickers.
        ticker_sector_map: Dict mapping tickers to sectors.

    Returns:
        The input DataFrame with an added 'sectors' column (list of strings).
    """
    sectors_list = []
    for tickers in df["tickers"]:
        if isinstance(tickers, (list, tuple)):
            secs = {ticker_sector_map.get(t.upper()) for t in tickers if ticker_sector_map.get(t.upper())}
            secs.discard(None)
            sectors_list.append(sorted(secs))
        else:
            sectors_list.append([])
    df = df.copy()
    df["sectors"] = sectors_list
    return df


def compute_coverage_metrics(
    articles: pd.DataFrame,
    trading_days: List[date],
    sectors: List[str],
) -> Tuple[pd.DataFrame, Dict[date, float]]:
    """Compute per‑sector coverage, staleness and diversity metrics.

    Args:
        articles: DataFrame of articles with columns 'published_utc',
                  'source', 'sectors'.  Articles should be deduplicated
                  prior to calling this function.
        trading_days: Sorted list of trading day dates (datetime.date).
        sectors: List of sector names to evaluate.

    Returns:
        (coverage_df, daily_dup_rate)
        coverage_df: DataFrame with columns
          [date_et, sector, n_articles, top_source_share, stale_share, near_dup_rate]
        daily_dup_rate: mapping of date to near duplicate ratio (already computed during dedup)
    """
    records = []
    # Precompute per day near_dup_rate passed in external variable; will handle outside.
    # Convert published_utc to ET date
    if articles.empty:
        return pd.DataFrame(columns=["date_et", "sector", "n_articles", "top_source_share", "stale_share", "near_dup_rate"]), {}
    articles = articles.copy()
    # Determine article day in ET (floor to date)
    articles["date_et"] = articles["published_utc"].apply(lambda dt: to_eastern(dt).date())
    # Determine staleness: hours difference between 16:00 ET on that day and published time
    def compute_stale(row):
        publish_et = to_eastern(row["published_utc"])
        # 16:00 ET cutoff on the article date
        cutoff = publish_et.replace(hour=16, minute=0, second=0, microsecond=0)
        diff_hours = (cutoff - publish_et).total_seconds() / 3600.0
        return diff_hours
    articles["recency_hours"] = articles.apply(compute_stale, axis=1)
    articles["is_stale"] = articles["recency_hours"] > 12
    # For each trading day and sector, compute metrics
    for d in trading_days:
        day_articles = articles[articles["date_et"] == d]
        # daily duplicate ratio: this will be passed separately
        for sector in sectors:
            sector_arts = day_articles[day_articles["sectors"].apply(lambda lst: sector in lst)]
            n_articles = len(sector_arts)
            if n_articles > 0:
                # Source diversity: compute share of largest source
                src_counts = sector_arts["source"].value_counts(dropna=True)
                top_share = src_counts.iloc[0] / n_articles if not src_counts.empty else 0.0
                stale_share = sector_arts["is_stale"].mean() if n_articles > 0 else 0.0
            else:
                top_share = 0.0
                stale_share = 0.0
            # We'll fill near_dup_rate later when available
            records.append(
                {
                    "date_et": pd.to_datetime(d),
                    "sector": sector,
                    "n_articles": n_articles,
                    "top_source_share": top_share,
                    "stale_share": stale_share,
                }
            )
    coverage_df = pd.DataFrame.from_records(records)
    return coverage_df, {}


def compute_near_dup_rates(raw_articles: pd.DataFrame) -> Dict[date, float]:
    """Compute near duplicate ratios per date.

    Args:
        raw_articles: DataFrame of raw articles before deduplication.

    Returns:
        A mapping from date to duplicate ratio.
    """
    if raw_articles.empty:
        return {}
    raw_articles = raw_articles.copy()
    raw_articles["date_et"] = raw_articles["published_utc"].apply(lambda dt: to_eastern(dt).date())
    ratios: Dict[date, float] = {}
    for d, group in raw_articles.groupby("date_et"):
        dedup_df, dup_ratio = deduplicate_articles(group, text_col="title", date_col="date_et")
        ratios[d] = dup_ratio
    return ratios


def fetch_and_compute_price_health(
    tickers: List[str],
    trading_days: List[date],
    pc: PolygonClient,
    lookback_days: int,
) -> pd.DataFrame:
    """Fetch price data and compute bar health and realised vol for each ticker.

    Args:
        tickers: List of ticker symbols to fetch.
        trading_days: List of trading day dates for which to compute metrics.
        pc: PolygonClient instance.
        lookback_days: Number of calendar days of history to fetch.

    Returns:
        DataFrame with columns: date_et, ticker, bar_missing_flag, rv20, rv60
    """
    end_date = trading_days[-1]
    start_date = end_date - timedelta(days=lookback_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    all_records = []
    for ticker in tickers:
        bars = pc.fetch_aggregates(
            ticker=ticker,
            timespan="day",
            multiplier=1,
            from_date=start_str,
            to_date=end_str,
        )
        if not bars:
            logging.warning("No bars returned for %s", ticker)
            continue
        # Convert to DataFrame
        bars_df = pd.DataFrame(bars)
        # 't' is timestamp in milliseconds; convert to datetime (UTC) and floor to date
        bars_df["date"] = pd.to_datetime(bars_df["t"], unit="ms", utc=True).dt.date
        bars_df = bars_df.set_index("date").sort_index()
        # Extract closing price
        prices = bars_df["c"].astype(float)
        # Compute realised vol
        rv20 = realised_volatility(prices, window=20)
        rv60 = realised_volatility(prices, window=60)
        # Determine if any missing bars in the last 14 days
        # (Check difference between requested trading days and available dates)
        missing_dates = set(trading_days) - set(prices.index)
        missing_flag = True if missing_dates else False
        for d in trading_days:
            rv20_val = rv20.get(d, np.nan)
            rv60_val = rv60.get(d, np.nan)
            all_records.append(
                {
                    "date_et": pd.to_datetime(d),
                    "ticker": ticker,
                    "bar_missing_flag": missing_flag,
                    "rv20": rv20_val,
                    "rv60": rv60_val,
                }
            )
    return pd.DataFrame(all_records)


def fetch_spy_constituents(fh: FinnhubClient, top_n: int) -> List[str]:
    """Fetch the top N holdings of the SPY ETF from Finnhub.

    If the API call fails, returns an empty list which signals to the
    caller to fall back to SPY‑tagged news only.

    Args:
        fh: FinnhubClient instance.
        top_n: Number of constituents to return.

    Returns:
        A list of ticker symbols.
    """
    holdings = fh.fetch_etf_holdings("SPY")
    if not holdings:
        return []
    # Sort by weight descending if available
    holdings_sorted = sorted(holdings, key=lambda x: x.get("weight", 0), reverse=True)
    constituents = [h.get("symbol") for h in holdings_sorted[:top_n] if h.get("symbol")]
    return constituents


def compute_spy_sentiment(
    articles: pd.DataFrame,
    trading_days: List[date],
    spy_constituents: List[str],
    spy_min_articles: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute SPY sentiment and viability metrics.

    Args:
        articles: Deduplicated article DataFrame with columns 'date_et', 'tickers', 'title'.
        trading_days: List of trading day dates.
        spy_constituents: List of tickers considered SPY constituents.
        spy_min_articles: Minimum number of articles per day; fallback to SPY‑tagged only if below this.

    Returns:
        spy_df: DataFrame with columns [date_et, n_primary, n_fallback, used_fallback, spy_sentiment]
        spy_sent_series: Series of daily sentiment scores indexed by date.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from torch.nn.functional import softmax

    # Load FinBERT (or DistilBERT fallback) once
    model_name = "ProsusAI/finbert"
    fallback_name = "distilbert-base-uncased-finetuned-sst-2-english"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception:
        logging.warning("Falling back to DistilBERT for sentiment")
        tokenizer = AutoTokenizer.from_pretrained(fallback_name)
        model = AutoModelForSequenceClassification.from_pretrained(fallback_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    spy_records = []
    sentiments = []
    for d in trading_days:
        day_articles = articles[articles["date_et"] == d]
        # Primary set: those mentioning any of the top N constituents
        primary = day_articles[day_articles["tickers"].apply(lambda ts: any(t.upper() in spy_constituents for t in ts))]
        # Fallback set: those mentioning 'SPY' explicitly
        fallback = day_articles[day_articles["tickers"].apply(lambda ts: "SPY" in [t.upper() for t in ts])]
        use_primary = len(primary) >= spy_min_articles
        selected = primary if use_primary else fallback
        used_fallback = not use_primary
        n_primary = len(primary)
        n_fallback = len(fallback)
        # Compute average polarity
        if len(selected) == 0:
            avg_pol = 0.0
        else:
            texts = (selected["title"].fillna("") + ". " + selected["description"].fillna(""))
            encodings = tokenizer.batch_encode_plus(
                texts.tolist(),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = model(**{k: v.to(device) for k, v in encodings.items()})
            logits = outputs.logits
            probs = softmax(logits, dim=-1).cpu().numpy()
            # FinBERT label order: [negative, neutral, positive]; DistilBERT: [negative, positive]
            if probs.shape[1] == 3:
                p_neg = probs[:, 0]
                p_pos = probs[:, 2]
            else:
                # DistilBERT has 2 classes [negative, positive]; treat neutral as 0
                p_neg = probs[:, 0]
                p_pos = probs[:, 1]
            polarity_scores = p_pos - p_neg
            avg_pol = float(np.mean(polarity_scores))
        spy_records.append(
            {
                "date_et": pd.to_datetime(d),
                "n_primary": n_primary,
                "n_fallback": n_fallback,
                "used_fallback": used_fallback,
                "spy_sentiment": avg_pol,
            }
        )
        sentiments.append((d, avg_pol))
    spy_df = pd.DataFrame.from_records(spy_records)
    spy_sent_series = pd.Series({d: val for d, val in sentiments})
    return spy_df, spy_sent_series


def compute_spy_correlation(spy_sent_series: pd.Series, spy_returns: pd.Series) -> Dict[str, float]:
    """Compute correlation at lead/lag ±1 between SPY sentiment and returns.

    Returns a dictionary with keys 'corr_lag+1', 'corr_lag0', 'corr_lag-1'.
    """
    # Align indices to ensure identical date order
    common_dates = spy_sent_series.index.intersection(spy_returns.index)
    if common_dates.empty:
        return {"corr_lag+1": np.nan, "corr_lag0": np.nan, "corr_lag-1": np.nan}
    x = spy_sent_series.loc[common_dates]
    y = spy_returns.loc[common_dates]
    # Lag definitions: lag+1 means sentiment leads returns by 1 day (sentiment on day d predicts return on d+1)
    corr = {}
    # Lag +1
    x_lag = x[:-1]
    y_lag = y[1:]
    if len(x_lag) > 1:
        corr["corr_lag+1"] = float(np.corrcoef(x_lag, y_lag)[0, 1])
    else:
        corr["corr_lag+1"] = np.nan
    # Lag 0
    if len(x) > 1:
        corr["corr_lag0"] = float(np.corrcoef(x, y)[0, 1])
    else:
        corr["corr_lag0"] = np.nan
    # Lag -1: returns lead sentiment
    x_lagm = x[1:]
    y_lagp = y[:-1]
    if len(x_lagm) > 1:
        corr["corr_lag-1"] = float(np.corrcoef(x_lagm, y_lagp)[0, 1])
    else:
        corr["corr_lag-1"] = np.nan
    return corr


def compute_earnings_density(
    earnings: List[Dict[str, Any]],
    ticker_sector_map: Dict[str, str],
    trading_days: List[date],
    sectors: List[str],
) -> pd.DataFrame:
    """Compute earnings density per sector per day.

    An earnings event is considered relevant if it falls within ±2 days
    of the trading day.  The density is the count of unique tickers
    with such events divided by the total number of mapped tickers in
    that sector.  If a sector has no mapped tickers the density is
    reported as 0.

    Args:
        earnings: List of earnings events from Finnhub.
        ticker_sector_map: Mapping from ticker to sector.
        trading_days: Sorted list of trading day dates.
        sectors: List of sector names.

    Returns:
        DataFrame with columns: date_et, sector, pct_constituents_reporting
    """
    # Build mapping sector -> all tickers in mapping
    sector_constituents: Dict[str, List[str]] = {}
    for t, sec in ticker_sector_map.items():
        sector_constituents.setdefault(sec, []).append(t)
    # Build a list of events with (event_date, sector, symbol) for all earnings
    events_full: List[Tuple[date, str, str]] = []  # (event_date, sector, symbol)
    for ev in earnings:
        sym = ev.get("symbol") or ev.get("ticker")
        if not sym:
            continue
        ev_date = ev.get("date") or ev.get("endDate")
        try:
            edate = pd.to_datetime(ev_date).date()
        except Exception:
            continue
        sec = ticker_sector_map.get(sym.upper())
        if sec:
            events_full.append((edate, sec, sym.upper()))
    # For each trading day and sector, compute density of events within ±2 days
    records = []
    for d in trading_days:
        window_start = d - timedelta(days=2)
        window_end = d + timedelta(days=2)
        for sec in sectors:
            cons = sector_constituents.get(sec, [])
            denom = len(set(cons))
            if denom == 0:
                pct = 0.0
            else:
                tickers_in_window = {
                    sym
                    for (ev_date, ev_sec, sym) in events_full
                    if ev_sec == sec and window_start <= ev_date <= window_end
                }
                pct = len(tickers_in_window) / denom
            records.append(
                {
                    "date_et": pd.to_datetime(d),
                    "sector": sec,
                    "pct_constituents_reporting": pct,
                }
            )
    return pd.DataFrame.from_records(records)


def write_summary(
    out_dir: str,
    coverage_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    price_health_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
    near_dup_rates: Dict[date, float],
    spy_corr: Dict[str, float],
    sectors: List[str],
) -> None:
    """Write a markdown summary report for the phase‑0 diagnostic.

    The summary consolidates the findings from the coverage report,
    SPY proxy viability, price health, and earnings density into a
    human‑readable file.
    """
    # Compute coverage pass metric: sectors with ≥3 articles on ≥70% of days
    coverage_pass = True
    trading_days = sorted({d.date() for d in coverage_df["date_et"]})
    num_days = len(trading_days)
    sectors_pass_count = 0
    for sec in sectors:
        sec_df = coverage_df[(coverage_df["sector"] == sec)]
        days_with_full = (sec_df["n_articles"] >= 3).sum()
        if days_with_full / num_days >= 0.7:
            sectors_pass_count += 1
    coverage_pass = sectors_pass_count >= 10  # At least 10 of 11 sectors

    # SPY sentiment volume pass
    primary_counts = spy_df["n_primary"]
    median_primary = primary_counts.median() if not primary_counts.empty else 0
    spy_volume_pass = median_primary >= 10

    # Compose summary
    lines: List[str] = []
    lines.append("# Phase‑0 Diagnostic Summary\n")
    lines.append(f"Generated on {datetime.utcnow().isoformat()} UTC\n")

    lines.append("## Coverage\n")
    lines.append(
        f"Coverage pass: {'✅' if coverage_pass else '❌'} – {sectors_pass_count} of {len(sectors)} sectors meet the ≥3 articles/day on ≥70% of days criterion.\n"
    )
    lines.append("### Near duplicate ratios by date\n")
    for d in sorted(near_dup_rates.keys()):
        ratio = near_dup_rates[d]
        lines.append(f"- {d}: {ratio:.2%} duplicates removed\n")

    lines.append("\n## SPY Sentiment Proxy\n")
    lines.append(
        f"Median primary SPY‑constituent articles/day: {median_primary:.2f}; volume pass: {'✅' if spy_volume_pass else '❌'}.\n"
    )
    lines.append(
        "Correlation of SPY sentiment vs SPY returns (sentiment leading returns):\n"
        f"- Lag +1: {spy_corr.get('corr_lag+1', float('nan')):.3f}\n"
        f"- Lag 0:  {spy_corr.get('corr_lag0', float('nan')):.3f}\n"
        f"- Lag -1: {spy_corr.get('corr_lag-1', float('nan')):.3f}\n"
    )

    lines.append("\n## Price Health\n")
    # Compute bar missing summary
    missing_summary = (
        price_health_df.groupby("ticker")["bar_missing_flag"].mean().sort_index()
    )
    for ticker, missing_rate in missing_summary.items():
        lines.append(f"- {ticker}: missing data on {missing_rate * 100:.1f}% of days\n")

    lines.append("\n## Earnings Density\n")
    lines.append(
        "This section summarises the availability of earnings data across sectors.  "
        "Values represent the proportion of tickers in the sector with an earnings event within ±2 days.\n"
    )
    # Show average density per sector
    avg_density = earnings_df.groupby("sector")["pct_constituents_reporting"].mean()
    for sec in sectors:
        avg = avg_density.get(sec, 0.0)
        lines.append(f"- {sec}: {avg:.2%} average density\n")

    # Write summary file
    summary_path = os.path.join(out_dir, "phase0_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info("Wrote summary to %s", summary_path)


######################################################################
# Main entrypoint
######################################################################


def main():
    parser = argparse.ArgumentParser(description="Run NBElastic Phase‑0 diagnostic.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file containing API keys and parameters.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where diagnostic CSVs and summary will be written.",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    pc = PolygonClient(api_key=cfg.polygon_api_key)
    fh = FinnhubClient(api_key=cfg.finnhub_api_key, rate_limit_config=cfg.finnhub_rate_limit)

    # Determine date range: last N trading days ending yesterday (ET)
    # Use current date in ET and subtract one day to avoid partial day
    today_et = to_eastern(datetime.utcnow()).date()
    end_date = today_et - timedelta(days=1)
    trading_days = last_trading_days(14, end_date)

    # Fetch news over a wider lookback for staleness and dedup considerations
    start_news = (end_date - timedelta(days=cfg.news_lookback_days)).strftime("%Y-%m-%d")
    end_news = end_date.strftime("%Y-%m-%d")
    raw_articles = pc.fetch_news(start_news, end_news, limit=cfg.news_per_page)
    news_df = parse_articles(raw_articles)
    logging.info("Parsed %d raw articles", len(news_df))

    # Compute near duplicate ratios per date using titles
    near_dup_rates = compute_near_dup_rates(news_df)

    # Deduplicate articles globally
    dedup_df, _ = deduplicate_articles(news_df, text_col="title", date_col="published_utc")
    logging.info("After deduplication: %d articles", len(dedup_df))

    # Build ticker -> sector mapping from all tickers appearing in the deduplicated set
    all_tickers = [t for sub in dedup_df["tickers"] if isinstance(sub, (list, tuple)) for t in sub]
    ticker_sector_map = build_sector_mapping(all_tickers, fh)
    logging.info("Mapped %d tickers to sectors", len(ticker_sector_map))

    # Assign sectors to articles
    dedup_df = assign_sectors_to_articles(dedup_df, ticker_sector_map)

    # Determine list of sectors present in mapping
    sectors = sorted({sec for sec in ticker_sector_map.values() if sec})
    # Guarantee our core sectors list includes at least the SPDR sectors if present
    # If mapping is empty, provide placeholder sectors
    if not sectors:
        sectors = ["TECHNOLOGY", "FINANCIALS", "HEALTH CARE"]

    # Add date_et column to dedup_df for subsequent processing
    if not dedup_df.empty:
        dedup_df = dedup_df.copy()
        dedup_df["date_et"] = dedup_df["published_utc"].apply(lambda dt: to_eastern(dt).date())

    # Compute coverage metrics
    coverage_df, _ = compute_coverage_metrics(dedup_df, trading_days, sectors)
    # Add near_dup_rate per date
    coverage_df["near_dup_rate"] = coverage_df["date_et"].dt.date.map(lambda d: near_dup_rates.get(d, 0.0))
    coverage_path = os.path.join(args.output_dir, "coverage_report.csv")
    coverage_df.to_csv(coverage_path, index=False)
    logging.info("Wrote coverage report to %s", coverage_path)

    # Fetch SPY constituents for sentiment weighting
    spy_constituents = fetch_spy_constituents(fh, cfg.spy_top_n)

    # Fallback to common SPY constituents if API fails
    if not spy_constituents:
        spy_constituents = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "BRK.B", "UNH",
            "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "ABBV", "PFE",
            "BAC", "KO", "AVGO", "PEP", "TMO", "COST", "WMT", "DIS", "ABT", "DHR",
            "VZ", "ADBE", "CRM", "NFLX", "CMCSA", "ACN", "NKE", "TXN", "RTX", "QCOM",
            "NEE", "PM", "HON", "UPS", "T", "SPGI", "LOW", "ORCL", "IBM", "GS"
        ]
        logging.info("Using fallback SPY constituents due to API access issue")

    logging.info("Using %d SPY constituents", len(spy_constituents))

    # Compute SPY sentiment per day and fallback flags
    spy_df, spy_sent_series = compute_spy_sentiment(
        dedup_df, trading_days, spy_constituents, cfg.spy_min_articles
    )
    # Compute SPY price returns
    spy_bars = pc.fetch_aggregates(
        ticker="SPY",
        timespan="day",
        multiplier=1,
        from_date=(end_date - timedelta(days=cfg.prices_lookback_days)).strftime("%Y-%m-%d"),
        to_date=end_date.strftime("%Y-%m-%d"),
    )
    spy_prices = pd.Series(
        {pd.to_datetime(bar["t"], unit="ms", utc=True).date(): bar["c"] for bar in spy_bars}
    ).sort_index()
    spy_returns = spy_prices.pct_change()
    spy_corr = compute_spy_correlation(spy_sent_series, spy_returns)
    # Write SPY proxy report
    spy_df_path = os.path.join(args.output_dir, "spy_proxy_report.csv")
    spy_df.to_csv(spy_df_path, index=False)
    logging.info("Wrote SPY proxy report to %s", spy_df_path)

    # Compute price health for sector ETFs and SPY
    etf_tickers = ["XLK", "XLF", "XLY", "XLV", "XLE", "XLB", "XLI", "XLU", "XLRE", "XLC", "XLP", "SPY"]
    price_health_df = fetch_and_compute_price_health(etf_tickers, trading_days, pc, cfg.prices_lookback_days)
    price_health_path = os.path.join(args.output_dir, "price_health.csv")
    price_health_df.to_csv(price_health_path, index=False)
    logging.info("Wrote price health report to %s", price_health_path)

    # Compute earnings density
    earnings_start = (trading_days[0] - timedelta(days=2)).strftime("%Y-%m-%d")
    earnings_end = (trading_days[-1] + timedelta(days=2)).strftime("%Y-%m-%d")
    earnings_events = fh.fetch_earnings_calendar(earnings_start, earnings_end)
    earnings_df = compute_earnings_density(
        earnings_events, ticker_sector_map, trading_days, sectors
    )
    earnings_path = os.path.join(args.output_dir, "earnings_density.csv")
    earnings_df.to_csv(earnings_path, index=False)
    logging.info("Wrote earnings density report to %s", earnings_path)

    # Write summary
    write_summary(
        args.output_dir,
        coverage_df,
        spy_df,
        price_health_df,
        earnings_df,
        near_dup_rates,
        spy_corr,
        sectors,
    )


if __name__ == "__main__":
    main()