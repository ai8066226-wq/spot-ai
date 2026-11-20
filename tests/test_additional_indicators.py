import unittest
import numpy as np
import pandas as pd
from additional_indicators import indicators, supertrend

class AdditionalIndicatorTests(unittest.TestCase):
    def data(self,c):
        c=np.asarray(c,dtype=float)
        return pd.DataFrame(dict(open=c,high=c+1,low=c-1,close=c,volume=np.ones(len(c))*10))
    def test_cci_mfi_obv_rising(self):
        d=self.data(np.arange(100,160));f=indicators(d)
        self.assertAlmostEqual(f.cci.iloc[-1],126.66666666666667)
        self.assertEqual(f.mfi.iloc[-1],100)
        self.assertEqual(f.obv.iloc[-1],590)
    def test_falling_money_flow(self):
        f=indicators(self.data(np.arange(160,100,-1)))
        self.assertEqual(f.mfi.iloc[-1],0)
        self.assertEqual(f.obv.iloc[-1],-590)
    def test_zero_volume_neutral(self):
        d=self.data(np.arange(100,160));d.volume=0
        f=indicators(d);self.assertEqual(f.mfi.iloc[-1],50);self.assertEqual(f.obv.iloc[-1],0)
    def test_supertrend_flips_after_large_move(self):
        d=self.data([100]*20+[120]*20+[80]*20)
        s=supertrend(d);self.assertFalse(s.iloc[19]);self.assertTrue(s.iloc[20]);self.assertFalse(s.iloc[40])
