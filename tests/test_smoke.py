import unittest
import numpy as np
import pandas as pd
from data_bybit import synthetic_ohlcv, fetch_klines
from indicators import add_all
from strategies import STRATEGIES
from backtest import run_backtest, metrics
from ml_engine import barrier_labels

class SmokeTests(unittest.TestCase):
    def test_all_strategies_spot(self):
        d=add_all(synthetic_ohlcv(n=1500)).iloc[200:]
        for name,(f,_) in STRATEGIES.items():
            s=f(d);self.assertTrue(np.isin(s,[0,1]).all(),name)
            t=run_backtest(d,s);self.assertIn('total_return',metrics(t))
            with self.assertRaises(ValueError):f(d,side='short')
    def test_future_does_not_change_past(self):
        raw=synthetic_ohlcv(n=700);d=add_all(raw)
        changed=raw.copy();changed.iloc[500:,:4]*=2
        pd.testing.assert_frame_equal(d.iloc[:500],add_all(changed).iloc[:500])
    def test_stop_first_and_gap(self):
        d=pd.DataFrame(dict(open=[100,100,90,100],high=[100,110,110,100],low=[100,90,80,100],close=[100,100,100,100],atr=[1]*4))
        t=run_backtest(d,[1,0,0,0],tp=.05,sl=.05,mode='pct');self.assertEqual(t.reason.iloc[0],'SL')
        with self.assertRaises(ValueError):run_backtest(d,[-1,0,0,0])
    def test_no_derivative_data(self):
        with self.assertRaises(ValueError):fetch_klines(category='linear')
    def test_initial_loss_in_drawdown(self):
        t=pd.DataFrame(dict(ret=[-.1],bars=[1]));self.assertAlmostEqual(metrics(t)['max_dd'],.1)
    def test_labels_shape(self):
        d=add_all(synthetic_ohlcv(n=400));ys=barrier_labels(d,2,1,24)
        for y in ys:self.assertEqual(len(y),len(d));self.assertTrue(np.isnan(y[-24:]).all())
