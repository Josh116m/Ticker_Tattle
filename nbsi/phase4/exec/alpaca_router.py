"""
Phase-4 Alpaca paper broker router (scaffold).
- Builds OPG orders for t+1, batch submit when dry_run=False.
- Reads keys from env (ALPACA_KEY_ID / ALPACA_SECRET_KEY) first; then optional secrets.yaml fallback.
- For scaffolding, we log intents; no live submissions are performed by default (dry_run=True).
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, List


def load_secrets_from_yaml(path: Path) -> Dict[str, str]:
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in (data or {}).items()}
    except Exception:
        return {}


def get_alpaca_keys() -> Dict[str, str]:
    key = os.getenv("ALPACA_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if key and secret:
        return {"key_id": key, "secret_key": secret}
    # fallback to secrets.yaml if present
    secrets_path = Path("secrets.yaml")
    if secrets_path.exists():
        sec = load_secrets_from_yaml(secrets_path)
        if "ALPACA_KEY_ID" in sec and "ALPACA_SECRET_KEY" in sec:
            return {"key_id": sec["ALPACA_KEY_ID"], "secret_key": sec["ALPACA_SECRET_KEY"]}
    return {"key_id": "", "secret_key": ""}


def build_opg_orders(intents: List[Dict[str, Any]], dry_run: bool = True) -> Dict[str, Any]:
    """
    Given a list of order intents like {symbol, side, qty}, produce Alpaca OPG orders payloads.
    This scaffolding only returns the payloads and logs; no submissions.
    """
    orders = []
    for it in intents:
        orders.append(
            {
                "symbol": it["symbol"],
                "side": it["side"],
                "qty": int(it.get("qty", 0)),
                "type": "market",
                "time_in_force": "opg",
                "client_order_id": it.get("client_order_id", None),
                "extended_hours": False,
            }
        )
    return {"dry_run": dry_run, "orders": orders}

