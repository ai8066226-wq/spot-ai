import unittest
from tune_icnt import grid,score
from research_v3 import Risk

class TuningTests(unittest.TestCase):
    def test_search_preserves_risk_budget_and_costs(self):
        risks=grid();self.assertEqual(len(risks),12)
        for r in risks:
            self.assertEqual(r.fee,Risk().fee)
            self.assertEqual(r.slippage,Risk().slippage)
            self.assertEqual(r.risk_per_trade,.005)
            self.assertEqual(r.max_allocation,.25)
            self.assertEqual(r.max_drawdown,.10)
    def test_no_trade_configuration_not_qualified(self):
        m=dict(trades=0,net_return=0.,profit_factor=None,max_drawdown=0.,payoff_ratio=None,drawdown_halt=False)
        sufficient,qualified,_=score([m,m,m])
        self.assertFalse(sufficient);self.assertFalse(qualified)
    def test_positive_return_is_not_enough(self):
        m=dict(trades=20,net_return=.05,profit_factor=1.1,max_drawdown=.02,payoff_ratio=1.2,drawdown_halt=False)
        sufficient,qualified,_=score([m,m,m])
        self.assertTrue(sufficient);self.assertFalse(qualified)
    def test_unstable_results_penalized(self):
        m=dict(trades=20,net_return=.05,profit_factor=2.,max_drawdown=.02,payoff_ratio=2.,drawdown_halt=False)
        stable=score([m,m,m])[2]
        weaker=score([m,m,{**m,'net_return':-.05}])
        self.assertFalse(weaker[1]);self.assertLess(weaker[2],stable)
