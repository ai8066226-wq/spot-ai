import unittest
import numpy as np
import pandas as pd
from technical_engine import indicators, signals, load_closed, RULES, exits

class TechnicalEngineTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(7);c=100*np.exp(np.cumsum(rng.normal(.0001,.01,1200)))
        return pd.DataFrame(dict(timestamp=np.arange(1200)*900000,open=c,high=c*1.01,low=c*.99,close=c,volume=rng.uniform(100,200,1200)))
    def test_no_future_leakage(self):
        d=self.data();f=indicators(d);s=signals(d,f)
        changed=d.copy();changed.loc[800:,['open','high','low','close']]*=2
        g=indicators(changed);t=signals(changed,g)
        pd.testing.assert_frame_equal(f.iloc[:800],g.iloc[:800])
        for name in s:np.testing.assert_array_equal(s[name][:800],t[name][:800])
    def test_registry_and_warmup(self):
        d=self.data();s=signals(d,indicators(d))
        self.assertEqual(set(s),set(RULES));self.assertEqual(len(s),5);self.assertEqual(len(exits()),12)
        for value in s.values():self.assertFalse(value[:199].any());self.assertEqual(value.dtype,bool)
    def test_prior_high_excludes_current(self):
        d=self.data();f=indicators(d)
        self.assertAlmostEqual(f.prior_high.iloc[500],d.high.iloc[480:500].max())
    def test_flat_data(self):
        d=self.data();d[['open','high','low','close']]=100.;f=indicators(d)
        self.assertEqual(f.rsi.iloc[-1],50)
        for s in signals(d,f).values():self.assertFalse(s.any())
    def test_excludes_open_candle_and_wrong_interval(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'d.csv';d=self.data();d.to_csv(path,index=False)
            self.assertEqual(len(load_closed(path,now_ms=int(d.timestamp.iloc[-1])+1)),1199)
            with self.assertRaises(ValueError):load_closed(path,interval=30)
