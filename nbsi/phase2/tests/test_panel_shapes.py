import unittest
import pandas as pd
import numpy as np
from nbsi.phase2.features.panel_builder import build_panel, REQUIRED_PANEL_COLS

class PanelShapesTest(unittest.TestCase):
    def test_panel_has_required_columns_and_dates(self):
        # Build tiny synthetic inputs
        df_clean = pd.DataFrame({
            'article_id':['a','b','c','d'],
            'published_utc':['2025-09-24T12:00:00Z','2025-09-24T18:00:00Z','2025-09-25T12:00:00Z','2025-09-26T12:00:00Z'],
            'assigned_date_et':['2025-09-24','2025-09-24','2025-09-25','2025-09-26'],
            'source':['X','Y','X','Y'],
            'title':['t1','t2','t3','t4'],
            'description':['d1','d2','d3','d4'],
            'tickers':[['SPY'],['XLK'],['XLF'],['XLK']],
            'url':['u1','u2','u3','u4'],
        })
        df_sent = pd.DataFrame({
            'article_id':['a','b','c','d'],
            'polarity':[0.2,-0.1,0.6,0.0],
            'confidence':[0.9,0.8,0.95,0.7],
        })
        df_rel = pd.DataFrame({
            'article_id':['a','b','c','d'],
            'relevance_XLK':[0.9,0.7,0.1,0.8],
            'relevance_XLF':[0.2,0.1,0.9,0.2],
        })
        spy_daily = pd.DataFrame({
            'date_et':['2025-09-24','2025-09-25','2025-09-26'],
            'spy_sentiment':[0.1,0.2,0.3],
            'rv20':[0.05,0.06,0.07],
            'rv60':[np.nan,np.nan,0.08],
        })
        spy_daily['date_et'] = pd.to_datetime(spy_daily['date_et'])

        panel = build_panel(df_clean, df_sent, df_rel, spy_daily, half_life_hours=24.0, extreme_thr=0.5)
        for c in REQUIRED_PANEL_COLS:
            self.assertIn(c, panel.columns)
        # Dates covered
        got = set(pd.to_datetime(panel['date_et']).dt.normalize().unique())
        exp = set(pd.to_datetime(spy_daily['date_et']).dt.normalize().unique())
        self.assertTrue(exp.issubset(got))

if __name__ == '__main__':
    unittest.main()

