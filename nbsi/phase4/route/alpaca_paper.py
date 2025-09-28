# nbsi/phase4/route/alpaca_paper.py
from __future__ import annotations
from dataclasses import dataclass
import os, json, time
import pandas as pd


@dataclass
class AlpacaCfg:
    base_url: str = "https://paper-api.alpaca.markets"
    key_id: str = os.getenv("ALPACA_KEY_ID", "")
    secret: str = os.getenv("ALPACA_SECRET_KEY", "")


def _headers(cfg: AlpacaCfg):
    if not (cfg.key_id and cfg.secret):
        raise RuntimeError("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY in environment.")
    return {
        "APCA-API-KEY-ID": cfg.key_id,
        "APCA-API-SECRET-KEY": cfg.secret,
        "Content-Type": "application/json",
    }


def preview_submissions(intents: pd.DataFrame) -> pd.DataFrame:
    """Return a minimal view of what we'd submit.
    Selects only the columns needed for order creation.
    """
    cols = ["date_et", "ticker", "action", "weight_delta"]
    return intents.loc[:, [c for c in cols if c in intents.columns]].copy()


def submit_opg_orders(intents: pd.DataFrame, cfg: AlpacaCfg) -> pd.DataFrame:
    """POST market-on-open (OPG) orders to Alpaca paper. Returns a table of results.

    Notes:
    - Lazy-imports requests to avoid adding a hard dependency for dry-run paths.
    - Simplistic lot sizing: qty ~ |weight_delta| * 100; adjust in future iterations.
    """
    # Lazy import to avoid requiring requests during dry paths
    import requests  # type: ignore

    url = f"{cfg.base_url}/v2/orders"
    H = _headers(cfg)
    out = []
    for _, r in intents.iterrows():
        side = "buy" if r["action"] == "buy" else "sell"
        qty = max(1, int(round(abs(float(r["weight_delta"])) * 100)))
        body = {
            "symbol": r["ticker"],
            "qty": qty,
            "side": side,
            "type": "market",
            "time_in_force": "opg",
            "client_order_id": f"opg-{r['ticker']}-{int(time.time()*1000)}",
        }
        resp = requests.post(url, headers=H, data=json.dumps(body))
        ok = 200 <= resp.status_code < 300
        out.append({
            "date_et": r["date_et"],
            "ticker": r["ticker"],
            "side": side,
            "qty": qty,
            "status_code": resp.status_code,
            "ok": ok,
            "resp": (resp.text or "")[:400],
        })
        if resp.status_code == 429:
            time.sleep(1.0)  # simple backoff
    return pd.DataFrame(out)

