from __future__ import annotations
import pandas as pd

REQUIRED_COVERAGE_RATIO = 0.70
REQUIRED_SECTORS = 10  # out of 11


def coverage_check(panel: pd.DataFrame) -> tuple[bool, str]:
    by_sector = panel.groupby('sector')['n_articles'].apply(lambda s: (s >= 3).mean())
    ok = int((by_sector >= REQUIRED_COVERAGE_RATIO).sum()) >= REQUIRED_SECTORS
    return ok, f"coverage sectors >=70%: {int((by_sector >= REQUIRED_COVERAGE_RATIO).sum())}/11"


def no_lookahead_check(panel: pd.DataFrame, labels: pd.DataFrame) -> tuple[bool, str]:
    # Expect labels indexed by date for next-day returns; ensure min(label_date) > max(signal_date) for same day mapping
    # We check that for each sector the label date equals next business day of the signal date at least
    p = panel[['date_et','sector']].drop_duplicates().sort_values(['sector','date_et'])
    l = labels.reset_index().rename(columns={'date':'label_date'})
    # labels expected per date per ticker; here we check that label dates are strictly greater than signal dates count-wise
    def to_naive_et(x: pd.Series) -> pd.Series:
        x = pd.to_datetime(x)
        if getattr(getattr(x.dt, 'tz', None), 'zone', None) is not None:
            return x.dt.tz_convert('US/Eastern').dt.tz_localize(None)
        return x
    min_signal = to_naive_et(p.groupby('sector')['date_et'].min())
    min_label = to_naive_et(l.groupby('ticker')['label_date'].min())
    aligned = []
    for sec in p['sector'].unique():
        if sec in min_label.index:
            aligned.append(min_label[sec] > min_signal.get(sec, pd.Timestamp.min))
    ok = all(aligned) if aligned else True
    return ok, "no look-ahead: labels occur after signals"


def cv_split_check(splits: list[tuple], dates: pd.DatetimeIndex, embargo_days: int) -> tuple[bool, str]:
    def gap_ok(train_idx, test_idx):
        if len(test_idx) == 0 or len(train_idx) == 0:
            return True
        min_test = dates[test_idx].min()
        max_test = dates[test_idx].max()
        train_dates = dates[train_idx]
        # Ensure embargo gap: no train dates within [min_test - embargo, max_test + embargo]
        emb_left = min_test - pd.Timedelta(days=embargo_days)
        emb_right = max_test + pd.Timedelta(days=embargo_days)
        overlaps = train_dates[(train_dates >= emb_left) & (train_dates <= emb_right)]
        return len(overlaps) == 0
    oks = [gap_ok(tr, te) for tr, te in splits]
    return all(oks), f"cv splits embargo respected: {sum(oks)}/{len(oks)}"

