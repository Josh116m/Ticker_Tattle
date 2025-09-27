import os
import sys
import pandas as pd

OUT_DIR = os.path.join('artifacts','phase0_rl_qa')
ARTS = {
    'coverage': os.path.join(OUT_DIR,'coverage_report.csv'),
    'spy': os.path.join(OUT_DIR,'spy_proxy_report.csv'),
    'price': os.path.join(OUT_DIR,'price_health.csv'),
    'earn': os.path.join(OUT_DIR,'earnings_density.csv'),
    'summary': os.path.join(OUT_DIR,'phase0_summary.md'),
}
LOG_PATH = os.path.join(OUT_DIR,'final_verifications.log')

REQ_COLS = {
    'coverage': ['date_et','sector','n_articles','top_source_share','stale_share','near_dup_rate'],
    'spy': ['date_et','n_primary','n_fallback','used_fallback','spy_sentiment'],
    'price': ['date_et','ticker','bar_missing_flag','rv20','rv60'],
}

lines = []

def log(msg):
    print(msg)
    lines.append(str(msg))

def check_exists():
    missing = [k for k,p in ARTS.items() if not os.path.exists(p)]
    if missing:
        log(f"MISSING artifacts: {missing}")
        return False
    return True

def check_schema(df, name):
    ok = True
    if name in REQ_COLS:
        cols = REQ_COLS[name]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            log(f"[{name}] missing columns: {missing}")
            ok = False
    log(f"[{name}] shape: {df.shape}")
    if 'date_et' in df.columns:
        dt = pd.to_datetime(df['date_et'], errors='coerce')
        log(f"[{name}] date_et range: {dt.min()} -> {dt.max()} (n unique dates: {dt.dt.normalize().nunique()})")
    log(f"[{name}] HEAD (5):\n{df.head(5).to_string(index=False)}")
    log(f"[{name}] TAIL (5):\n{df.tail(5).to_string(index=False)}")
    return ok

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    overall_ok = True

    if not check_exists():
        overall_ok = False
    else:
        cov = pd.read_csv(ARTS['coverage'])
        spy = pd.read_csv(ARTS['spy'])
        price = pd.read_csv(ARTS['price'])
        earn = pd.read_csv(ARTS['earn'])

        # Schemas + prints
        overall_ok &= check_schema(cov, 'coverage')
        overall_ok &= check_schema(spy, 'spy')
        overall_ok &= check_schema(price, 'price')
        log('[earn] shape: ' + str(earn.shape))
        if 'date_et' in earn.columns:
            dt = pd.to_datetime(earn['date_et'], errors='coerce')
            log(f"[earn] date_et range: {dt.min()} -> {dt.max()} (n unique dates: {dt.dt.normalize().nunique()})")
        log('[earn] HEAD (5):\n' + earn.head(5).to_string(index=False))
        log('[earn] TAIL (5):\n' + earn.tail(5).to_string(index=False))

        # Business checks
        # spy fallback logic
        fb = spy[(spy['n_primary'] < 8) & (spy['used_fallback'] != True)]
        if not fb.empty:
            log(f"[spy] VIOLATION: rows where n_primary<8 but used_fallback!=True -> {len(fb)}")
            overall_ok = False
        fb2 = spy[(spy['n_primary'] >= 8) & (spy['used_fallback'] != False)]
        if not fb2.empty:
            log(f"[spy] VIOLATION: rows where n_primary>=8 but used_fallback!=False -> {len(fb2)}")
            overall_ok = False

        # price completeness
        miss = price[price['bar_missing_flag'] == True]
        if not miss.empty:
            log(f"[price] VIOLATION: missing bars rows -> {len(miss)}")
            overall_ok = False
        if price['rv20'].isna().any():
            log(f"[price] VIOLATION: rv20 has {price['rv20'].isna().sum()} nulls")
            overall_ok = False
        # rv60 may be null early, just log counts
        log(f"[price] rv60 nulls: {price['rv60'].isna().sum()}")

        # coverage quick continuity signal (min/max by date only)
        dts = pd.to_datetime(cov['date_et']).dt.normalize().unique()
        log(f"[coverage] unique date count: {len(dts)} from {pd.Timestamp(dts[0]).date()} to {pd.Timestamp(dts[-1]).date()}")

    with open(LOG_PATH,'w',encoding='utf-8') as fh:
        fh.write('\n'.join(lines))

    if overall_ok:
        print('PHASE 0 RL+QA PASS — proceed to Phase-2')
    else:
        print('PHASE 0 RL+QA FAIL — block Phase-2 (see log)')

if __name__ == '__main__':
    main()

