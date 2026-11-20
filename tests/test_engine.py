import unittest
import numpy as np
import pandas as pd
from engine import backtest, features, validate

class Tests(unittest.TestCase):
    def candles(self):
        c=np.arange(100.,300.)
        return pd.DataFrame(dict(timestamp=np.arange(200)*900000,open=c,high=c+2,low=c-2,close=c,volume=np.ones(200)))
    def test_no_future_features(self):
        a=self.candles(); b=a.copy(); b.loc[150:,'close']*=2
        pd.testing.assert_frame_equal(features(a).iloc[:150],features(b).iloc[:150])
    def test_next_open(self):
        d=self.candles().iloc[:3].copy(); d['open']=[100,200,300]; d['close']=[100,200,300]
        self.assertAlmostEqual(backtest(d,[True,False,False],0,0)['net_return'],.5)
    def test_costs(self):
        d=self.candles().iloc[:4].copy(); d[['open','close']]=100.
        r=backtest(d,[True,False,False,False],.001,.0005)
        self.assertLess(r['net_return'],0); self.assertEqual(r['trades'],1)
    def test_cash(self):
        self.assertEqual(backtest(self.candles(),np.zeros(200))['net_return'],0)
    def test_bad_data(self):
        d=self.candles(); d.loc[2,'timestamp']=d.loc[1,'timestamp']
        with self.assertRaises(ValueError): validate(d)
    def test_gap(self):
        with self.assertRaises(ValueError): validate(self.candles().drop(10))

if __name__=='__main__': unittest.main()
