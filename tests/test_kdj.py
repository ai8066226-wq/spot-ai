import unittest
import numpy as np
import pandas as pd
from kdj_research import kdj,cross_up
class KDJTests(unittest.TestCase):
    def candles(self,n=200):
        rng=np.random.default_rng(42);c=100+np.cumsum(rng.normal(size=n))
        return pd.DataFrame(dict(close=c,high=c+2,low=c-2))
    def test_crossover_equivalence(self):
        f=kdj(self.candles())
        np.testing.assert_allclose(f.J-f.K,2*(f.K-f.D),atol=1e-12)
        np.testing.assert_equal(cross_up(f.J,f.K),cross_up(f.K,f.D))
    def test_no_future_candles(self):
        a=self.candles();b=a.copy();b.loc[150:]*=10
        pd.testing.assert_frame_equal(kdj(a).iloc[:150],kdj(b).iloc[:150])
    def test_flat_prices(self):
        d=pd.DataFrame(dict(close=np.full(100,10),high=np.full(100,10),low=np.full(100,10)))
        f=kdj(d)
        np.testing.assert_allclose(f.iloc[8:],50)
        self.assertFalse(cross_up(f.J,f.K).any())
    def test_cross_only_on_transition(self):
        np.testing.assert_equal(cross_up([0,2,3,0,2],[1,1,1,1,1]),[False,True,False,False,True])
