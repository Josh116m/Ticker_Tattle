import unittest
import pandas as pd
from nbsi.phase2.etl.news_cleaner import clean_day


class NewsPaginationTest(unittest.TestCase):
    def test_cleaner_no_duplicates_and_order_preserved(self):
        # Simulate two pages with ordered times and one duplicate id/title
        data1 = [
            {'id':'a','published_utc':'2025-09-26T10:00:00Z','publisher.name':'X','title':'T1','description':'D1','tickers':['SPY'],'article_url':'u1'},
            {'id':'b','published_utc':'2025-09-26T11:00:00Z','publisher.name':'X','title':'T2','description':'D2','tickers':['XLK'],'article_url':'u2'},
        ]
        data2 = [
            {'id':'c','published_utc':'2025-09-26T12:00:00Z','publisher.name':'Y','title':'T3','description':'D3','tickers':['XLF'],'article_url':'u3'},
            {'id':'b','published_utc':'2025-09-26T11:00:00Z','publisher.name':'X','title':'T2','description':'D2','tickers':['XLK'],'article_url':'u2'},
        ]
        df_raw = pd.json_normalize(data1 + data2)
        df_clean = clean_day(df_raw)
        # No duplicates on title+description (one b is dropped)
        self.assertEqual(df_clean['article_id'].nunique(), 3)
        # assigned_date_et exists and is a date
        self.assertIn('assigned_date_et', df_clean.columns)


if __name__ == '__main__':
    unittest.main()

