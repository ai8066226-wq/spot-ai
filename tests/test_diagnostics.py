import unittest
import numpy as np
import pandas as pd
from diagnose_icnt import event_outcomes,probability_metrics,fixed_trade_costs,breakeven
from research_v3 import Risk,enriched,labels

class DiagnosticTests(unittest.TestCase):
    def test_targets_match_existing_simulation_labels(self):
        rng=np.random.default_rng(7);c=100*np.exp(np.cumsum(rng.normal(0,.01,150)))
        o=np.r_[c[0],c[:-1]]
        d=pd.DataFrame(dict(timestamp=np.arange(150)*900000,open=o,high=np.maximum(o,c)*1.005,low=np.minimum(o,c)*.995,close=c,volume=np.ones(150)))
        f=enriched(d);f['atr']=f.atr.fillna(.01)
        expected,ends=labels(d,f,Risk());got=event_outcomes(d,f,Risk())
        np.testing.assert_allclose(expected,got['y'],equal_nan=True)
        np.testing.assert_equal(ends,got['ends'])
    def test_constant_baseline_has_no_skill(self):
        y=np.array([0,1,0,0,1]*20);p=np.full(len(y),.4)
        m=probability_metrics(y,p,.4)
        self.assertEqual(m['roc_auc'],.5);self.assertAlmostEqual(m['brier_skill'],0)
    def test_good_ranker_beats_bad_ranker(self):
        y=np.array([0,1]*50);good=np.where(y==1,.9,.1)
        a=probability_metrics(y,good,.5);b=probability_metrics(y,1-good,.5)
        self.assertGreater(a['roc_auc'],b['roc_auc']);self.assertGreater(a['brier_skill'],b['brier_skill'])
    def test_fixed_cost_accounting(self):
        r=Risk();q=2.;buy=100*(1+r.slippage);sell=110*(1-r.slippage)
        pnl=q*(sell*(1-r.fee)-buy*(1+r.fee))
        m=fixed_trade_costs([dict(pnl=pnl,quantity=q,entry=buy,exit=sell)],r)
        self.assertAlmostEqual(m['same_trades_without_fees_or_slippage_pnl'],20)
        self.assertAlmostEqual(m['actual_pnl']+m['fee_drag']+m['slippage_drag'],20)
    def test_payoff_breakeven(self):
        self.assertAlmostEqual(breakeven(np.array([.02,.02,-.01,-.01])),1/3)
        self.assertIsNone(breakeven(np.array([.01,.02])))
