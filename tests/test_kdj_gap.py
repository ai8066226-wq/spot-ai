import unittest
import numpy as np
import pandas as pd
from kdj_gap_exit import simulate_gap
from research_v3 import Risk

def data():
    o=np.arange(100.,108.)
    d=pd.DataFrame(dict(timestamp=np.arange(8)*900000,open=o,high=o+.2,low=o-.2,close=o))
    return d,pd.DataFrame(dict(atr=np.full(8,.02)))

class GapExitTests(unittest.TestCase):
    def test_shrink_exit_at_next_open_not_peak(self):
        d,f=data();r=simulate_gap(d,f,[-1,2,5,4,3,2,1,0],Risk(fee=0,slippage=0),True)
        t=r['trade_log'][0]
        self.assertEqual(t['entry_index'],2);self.assertEqual(t['exit_index'],4)
        self.assertEqual(t['exit'],104);self.assertEqual(t['max_gap'],5)
        self.assertEqual(t['exit_signal_timestamp'],int(d.timestamp.iloc[3]))
    def test_future_larger_peak_does_not_postpone_exit(self):
        d,f=data();a=simulate_gap(d,f,[-1,2,5,4,3,2,1,0],Risk(),True)
        b=simulate_gap(d,f,[-1,2,5,4,3,1000,2000,3000],Risk(),True)
        self.assertEqual(a['trade_log'][0],b['trade_log'][0])
    def test_equal_distance_keeps_position(self):
        d,f=data();r=simulate_gap(d,f,[-1,2,5,5,5,4,3,2],Risk(),True)
        self.assertEqual(r['trade_log'][0]['exit_index'],6)
    def test_protective_stop_overrides_gap_wait(self):
        d,f=data();d.loc[2,'low']=90
        r=simulate_gap(d,f,[-1,2,5,6,7,8,9,10],Risk(),True)
        self.assertEqual(r['trade_log'][0]['reason'],'stop')
    def test_scheduled_exit_ignores_later_intrabar_low(self):
        d,f=data();d.loc[4,'low']=90
        r=simulate_gap(d,f,[-1,2,5,4,3,2,1,0],Risk(fee=0,slippage=0),True)
        self.assertEqual(r['trade_log'][0]['reason'],'gap_shrink_next_open')
        self.assertEqual(r['trade_log'][0]['exit'],104)
    def test_stop_gap_can_exceed_planned_risk(self):
        d,f=data();d.loc[3,['open','high','low','close']]=50
        r=simulate_gap(d,f,[-1,2,5,6,7,8,9,10],Risk(),True)
        t=r['trade_log'][0];self.assertEqual(t['reason'],'stop_gap')
        self.assertGreater(-t['pnl'],t['planned_stop_risk'])
    def test_flat_has_no_trades(self):
        d,f=data();self.assertEqual(simulate_gap(d,f,np.zeros(8))['trades'],0)
