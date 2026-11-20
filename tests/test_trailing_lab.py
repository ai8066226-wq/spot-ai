import unittest
import numpy as np
import pandas as pd
from trailing_lab import ExitSettings,setting_grid,simulate_trailing,updated_stop,parse_values
from research_v3 import Risk


def data():
    d=pd.DataFrame(dict(timestamp=np.arange(6)*900000,open=np.full(6,100.),high=np.full(6,101.),low=np.full(6,99.),close=np.full(6,100.)))
    return d,pd.DataFrame(dict(atr=np.full(6,.01)))

class TrailingLabTests(unittest.TestCase):
    def test_user_values_and_grid_limit(self):
        self.assertEqual(parse_values('0.5,1,1'),(.005,.01))
        self.assertEqual(parse_values('0,0.5',allow_zero=True),(0.,.005))
        for text in ('nan','-1','101',''):
            with self.assertRaises(ValueError):parse_values(text)
        with self.assertRaises(ValueError):setting_grid(stops=tuple([.01]*100))
    def test_grid_is_bounded_and_unique(self):
        settings=setting_grid();self.assertEqual(len(settings),315);self.assertEqual(len(set(settings)),315)
    def test_stop_never_moves_down(self):
        s=ExitSettings(.05,.5,'atr',2.,0.)
        first=updated_stop(95,110,108,.01,s)
        second=updated_stop(first,110,108,.1,s)
        self.assertEqual(second,first)
    def test_new_trailing_stop_not_applied_to_old_low(self):
        d,f=data();d.loc[1,['open','high','low','close']]=[100,110,97,109]
        d.loc[2,['open','high','low','close']]=[108,109,98,100]
        s=ExitSettings(.05,.5,'percent',.1,0.)
        r=simulate_trailing(d,f,[1,0,0,0,0,0],s,Risk(fee=0,slippage=0),True)
        t=r['trade_log'][0];self.assertEqual(t['exit_index'],2);self.assertEqual(t['exit'],99)
        self.assertEqual(r['stop_trace'][0]['event'],'close_update_effective_next_bar')
    def test_stop_wins_ambiguous_bar(self):
        d,f=data();d.loc[1,['high','low']]=[120,90]
        r=simulate_trailing(d,f,[1,0,0,0,0,0],ExitSettings(.05,.1,'percent',.01,0),Risk(fee=0,slippage=0),True)
        self.assertEqual(r['trade_log'][0]['reason'],'initial_stop');self.assertEqual(r['trade_log'][0]['exit'],95)
    def test_trailing_activation_not_early(self):
        d,f=data();s=ExitSettings(.1,.5,'percent',.01,.05)
        r=simulate_trailing(d,f,[1,0,0,0,0,0],s,Risk(fee=0,slippage=0),True)
        self.assertFalse(r['stop_trace'][0]['activated']);self.assertEqual(r['stop_trace'][0]['stop_after'],90)
    def test_gap_exceeds_stop_budget(self):
        d,f=data();d.loc[2,['open','high','low','close']]=[80,81,79,80]
        r=simulate_trailing(d,f,[1,0,0,0,0,0],ExitSettings(.05,.5,'percent',.1,.2),Risk(),True)
        t=r['trade_log'][0];self.assertEqual(t['reason'],'stop_gap');self.assertGreater(-t['pnl'],t['planned_stop_risk'])
    def test_price_gap_does_not_improve_target_fill(self):
        d,f=data();d.loc[2,['open','high','low','close']]=[125,130,120,125]
        r=simulate_trailing(d,f,[1,0,0,0,0,0],ExitSettings(.05,.1,'percent',.1,.2),Risk(fee=0,slippage=0),True)
        self.assertEqual(r['trade_log'][0]['reason'],'target_gap');self.assertAlmostEqual(r['trade_log'][0]['exit'],110)
    def test_trade_fills_at_next_open(self):
        d,f=data();d.loc[1,['open','high','low','close']]=[105,106,104,105]
        r=simulate_trailing(d,f,[1,0,0,0,0,0],ExitSettings(.05,.5,'percent',.1,.2),Risk(fee=0,slippage=0),True)
        self.assertEqual(r['trade_log'][0]['entry'],105)
    def test_same_prices_still_pay_costs(self):
        d,f=data();d[['open','high','low','close']]=100.
        r=simulate_trailing(d,f,[1,0,0,0,0,0],ExitSettings(.05,.5,'percent',.1,.2),Risk())
        self.assertLess(r['net_return'],0)
    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):ExitSettings(.01,.02,'percent',1.5,0)
