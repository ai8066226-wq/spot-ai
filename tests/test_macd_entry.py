import unittest
import numpy as np
import pandas as pd
from trailing_lab import entry_signals
from kdj_research import cross_up

class MACDEntryTests(unittest.TestCase):
    def test_above_is_continuous_not_only_cross(self):
        d=pd.DataFrame(dict(close=np.linspace(100,200,200)))
        signal=entry_signals(d,'macd_above')
        self.assertTrue(signal[1:].all())
        dif=d.close.ewm(span=12,adjust=False).mean()-d.close.ewm(span=26,adjust=False).mean()
        dea=dif.ewm(span=9,adjust=False).mean()
        self.assertGreater(signal.sum(),cross_up(dif,dea).sum())
    def test_equality_does_not_buy(self):
        d=pd.DataFrame(dict(close=np.ones(200)))
        self.assertFalse(entry_signals(d,'macd_above').any())
    def test_future_prices_do_not_change_past_signals(self):
        rng=np.random.default_rng(42);a=pd.DataFrame(dict(close=100+np.cumsum(rng.normal(size=200))))
        b=a.copy();b.loc[150:,'close']*=3
        np.testing.assert_equal(entry_signals(a,'macd_above')[:150],entry_signals(b,'macd_above')[:150])
