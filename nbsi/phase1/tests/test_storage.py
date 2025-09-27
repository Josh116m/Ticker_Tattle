import os
import shutil
import tempfile
import unittest
import pandas as pd

from nbsi.phase1.data.storage import write_df, read_df


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "test.parquet")
        # Detect pandas construction issues and skip if environment is broken
        self.pandas_ok = True
        try:
            _ = pd.DataFrame({"a": [1], "b": ["x"]})
        except Exception:
            self.pandas_ok = False

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_roundtrip_and_schema(self):
        if not self.pandas_ok:
            self.skipTest("Pandas environment is not functional for DataFrame construction")
        df = pd.DataFrame([(1, "x"), (2, "y"), (3, "z")], columns=["a","b"])
        write_df(df, self.path, schema=["a","b"], mode="overwrite")
        out = read_df(self.path)
        self.assertEqual(list(out.columns), ["a","b"])
        self.assertEqual(out.shape, (3,2))

    def test_overwrite_atomic(self):
        if not self.pandas_ok:
            self.skipTest("Pandas environment is not functional for DataFrame construction")
        df1 = pd.DataFrame([(1, "x")], columns=["a","b"])
        write_df(df1, self.path, schema=["a","b"], mode="overwrite")
        df2 = pd.DataFrame([(2, "y"), (3, "z")], columns=["a","b"])
        write_df(df2, self.path, schema=["a","b"], mode="overwrite")
        out = read_df(self.path)
        self.assertEqual(out.shape, (2,2))


if __name__ == "__main__":
    unittest.main()

