import os
import sys
import json
from datetime import date, timedelta
import pandas as pd
import requests

from nbsi.phase1.data.polygon_client import PolygonClient
from nbsi.phase1.data.finnhub_client import FinnhubClient
from nbsi.phase1.data.storage import write_df

BASE = os.path.dirname(os.path.dirname(__file__))
CFG_DIR = os.path.join(BASE, "phase1", "configs")


def load_secrets():
    # Prefer local secrets.yaml; fallback to phase0 config for local runs
    import yaml
    sec_path = os.path.join(CFG_DIR, "secrets.yaml")
    if os.path.exists(sec_path):
        with open(sec_path, "r") as fh:
            return yaml.safe_load(fh)
    p0 = os.path.join(os.path.dirname(BASE), "nbsi_phase0", "config.yaml")
    with open(p0, "r") as fh:
        data = yaml.safe_load(fh)
    return {
        "polygon_api_key": data.get("polygon_api_key") or data.get("polygon",{}).get("api_key"),
        "finnhub_api_key": data.get("finnhub_api_key") or data.get("finnhub",{}).get("api_key"),
        "alpaca": {
            "key_id": data.get("alpaca",{}).get("key_id"),
            "secret_key": data.get("alpaca",{}).get("secret_key"),
        }
    }


def main():
    secrets = load_secrets()
    poly_key = secrets["polygon_api_key"]
    fh_key = secrets["finnhub_api_key"]
    if not poly_key or not fh_key:
        print("Missing API keys; ensure configs/secrets.yaml is populated.")
        sys.exit(2)

    outdir = os.path.join("artifacts", "phase1_smoke")
    os.makedirs(outdir, exist_ok=True)

    # Dates: use last trading day from Phase-0 window end (yesterday ET proxy)
    # For smoke, we pull exactly one day window [D, D+1)
    D = date.today() - timedelta(days=1)
    d0 = D.strftime("%Y-%m-%d")
    d1 = (D + timedelta(days=1)).strftime("%Y-%m-%d")

    # Polygon news one day
    pc = PolygonClient(poly_key)
    news = pc.list_news(d0, d1, tickers=None, page_size=50)
    news_df = pd.DataFrame(news)
    if news_df.empty:
        print("Empty news_df; FAIL")
        sys.exit(3)
    write_df(news_df, os.path.join(outdir, f"news_{D.strftime('%Y%m%d')}.parquet"))

    # SPY and XLK aggs
    aggs_spy = pc.get_aggs_daily("SPY", d0, d1)
    aggs_xlk = pc.get_aggs_daily("XLK", d0, d1)
    spy_df = pd.DataFrame(aggs_spy)
    xlk_df = pd.DataFrame(aggs_xlk)
    if spy_df.empty or xlk_df.empty:
        print("Empty aggs; FAIL")
        sys.exit(4)
    write_df(spy_df, os.path.join(outdir, "aggs_SPY.parquet"))
    write_df(xlk_df, os.path.join(outdir, "aggs_XLK.parquet"))

    # Finnhub ping: profile for AAPL (exercise limiter and logs)
    rate_cfg = {
        "max_per_second": 25,
        "bucket_capacity": 30,
        "backoff_base_seconds": 0.5,
        "backoff_factor": 2.0,
        "backoff_max_seconds": 60,
    }
    fh = FinnhubClient(api_key=fh_key, rate_limit_config=rate_cfg)
    prof = fh.company_profile("AAPL")

    # Print shapes and heads/tails
    print("news_df:", news_df.shape)
    print(news_df.head(3).to_string(index=False))
    print(news_df.tail(3).to_string(index=False))

    print("spy_df:", spy_df.shape)
    print(spy_df.head(3).to_string(index=False))
    print(spy_df.tail(3).to_string(index=False))

    print("xlk_df:", xlk_df.shape)
    print(xlk_df.head(3).to_string(index=False))
    print(xlk_df.tail(3).to_string(index=False))

    print("Finnhub profile keys:", sorted(list(prof.keys()))[:10])

    print("SMOKE OK")


if __name__ == "__main__":
    main()

