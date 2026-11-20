import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import numpy as np
import pandas as pd
import main

class CsvInputTests(unittest.TestCase):
    def sample(self):
        c=np.linspace(10,12,1600)
        return pd.DataFrame(dict(timestamp=1700000100000+np.arange(1600)*900000,open=c,high=c+.1,low=c-.1,close=c,volume=np.ones(1600)*100))
    def test_csv_runs_without_downloading(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);self.sample().to_csv(p/'input.csv',index=False)
            args=['main.py','--csv',str(p/'input.csv'),'--interval','15','--mode','rules','--out',str(p/'out')]
            with patch('sys.argv',args),patch.object(main,'fetch_klines',side_effect=AssertionError('network forbidden')),patch.object(main,'search',return_value=(pd.DataFrame(),pd.DataFrame())),patch('builtins.print'):
                main.main()
            meta=json.loads((p/'out/run_metadata.json').read_text());self.assertEqual(meta['source'],'uploaded_csv');self.assertFalse(meta['synthetic']);self.assertEqual(meta['rows'],1600)
    def test_gap_is_rejected(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);self.sample().drop(500).to_csv(p/'input.csv',index=False)
            with patch('sys.argv',['main.py','--csv',str(p/'input.csv'),'--interval','15','--out',str(p/'out')]):
                with self.assertRaisesRegex(ValueError,'gaps'):main.main()
