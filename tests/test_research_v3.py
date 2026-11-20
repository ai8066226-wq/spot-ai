import unittest
import numpy as np
import pandas as pd
from research_v3 import Risk,bar_exit,simulate,labels,enriched,fit_bundle,predict,train_columns,block_bootstrap


def data(n=200,price=100.):
    return pd.DataFrame(dict(timestamp=1700000100000+np.arange(n)*900000,open=np.full(n,price),high=np.full(n,price),low=np.full(n,price),close=np.full(n,price),volume=np.ones(n)))

class ResearchTests(unittest.TestCase):
    def test_ambiguous_bar_stop_first(self):
        self.assertEqual(bar_exit(100,110,90,100,98,105,1,Risk()),(98.,'stop'))
    def test_stop_gap_worse_price(self):
        self.assertEqual(bar_exit(90,92,88,91,98,105,1,Risk()),(90.,'stop_gap'))
    def test_target_gap_no_improvement(self):
        self.assertEqual(bar_exit(110,112,108,111,98,105,1,Risk()),(105.,'target_gap'))
    def test_no_position_no_loss(self):
        d=data();f=enriched(d);f['atr']=.01
        self.assertEqual(simulate(d,f,np.zeros(len(d)),Risk())['net_return'],0)
    def test_sizing_and_costs(self):
        d=data(5);d.loc[1,'low']=90
        f=enriched(d);f['atr']=.01
        r=simulate(d,f,[1,0,0,0,0],Risk(),True)
        t=r['trade_log'][0]
        self.assertLessEqual(t['planned_stop_risk'],.005+1e-10)
        self.assertLessEqual(t['spend'],.25+1e-10)
        self.assertAlmostEqual(-t['pnl'],t['planned_stop_risk'])
    def test_gaps_can_exceed_risk(self):
        d=data(5);d.loc[2,['open','high','low','close']]=50
        f=enriched(d);f['atr']=.01
        r=simulate(d,f,[1,0,0,0,0],Risk(),True)
        t=r['trade_log'][0]
        self.assertGreater(-t['pnl'],t['planned_stop_risk'])
        self.assertTrue(r['drawdown_halt'])
    def test_next_open_entry(self):
        d=data(5);d.loc[1,['open','high','low','close']]=200
        f=enriched(d);f['atr']=.01
        r=simulate(d,f,[1,0,0,0,0],Risk(fee=0,slippage=0),True)
        self.assertEqual(r['trade_log'][0]['entry'],200)
    def test_labels_match_execution(self):
        d=data(30);d.loc[1,'high']=110
        f=enriched(d);f['atr']=.01;risk=Risk()
        y,ends=labels(d,f,risk)
        r=simulate(d,f,[1]+[0]*29,risk,True)
        self.assertEqual(y[0],int(r['trade_log'][0]['pnl']>0))
        self.assertEqual(ends[0],r['trade_log'][0]['exit_index'])
        self.assertTrue(np.isnan(y[-1]))
    def test_feature_causality(self):
        rng=np.random.default_rng(1);a=data();a['close']=100+np.cumsum(rng.normal(0,.1,len(a)))
        b=a.copy();b.loc[150:,'close']*=3
        pd.testing.assert_frame_equal(enriched(a).iloc[:150],enriched(b).iloc[:150])
    def test_redundancy_filter(self):
        f=pd.DataFrame(dict(a=np.arange(200),copy=np.arange(200)*2,constant=np.ones(200)))
        self.assertEqual(train_columns(f),['a'])
    def test_calibration_boundaries_and_future(self):
        rng=np.random.default_rng(1);n=1300
        f=pd.DataFrame(dict(a=rng.normal(size=n),b=rng.normal(size=n)))
        y=rng.integers(0,2,n).astype(float);ends=np.arange(n)+16
        b=fit_bundle(f,y,ends,1000,list(f),Risk())
        self.assertLess(ends[b['train_last']],b['calibration_first'])
        self.assertLess(ends[b['calibration_last']],b['evaluation_first'])
        newer=f.copy();newer.loc[1000:]*=100
        other=fit_bundle(newer,y,ends,1000,list(f),Risk())
        np.testing.assert_allclose(predict(b,f.iloc[1000:]),predict(other,f.iloc[1000:]))
    def test_small_bootstrap_rejected(self):
        self.assertFalse(block_bootstrap([])['available'])
    def test_daily_loss_blocks_new_entries(self):
        d=data(8);d['timestamp']=1700006400000+np.arange(8)*900000;d.loc[1:,'low']=90
        f=enriched(d);f['atr']=.01
        risk=Risk(risk_per_trade=.05,max_allocation=1.,daily_loss=.005,max_drawdown=.9,cooldown=0)
        r=simulate(d,f,np.ones(8),risk,True)
        self.assertEqual(r['trades'],1)
    def test_rejection_removes_old_model(self):
        import tempfile
        from pathlib import Path
        from research_v3 import run
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);d=data(2700);d.to_csv(path/'input.csv',index=False)
            (path/'model.pkl').write_bytes(b'old')
            (path/'predictions.csv').write_text('old')
            report=run(path/'input.csv',path)
            self.assertEqual(report['status'],'REJECTED_KEEP_CASH')
            self.assertEqual(report['model_final']['net_return'],0)
            self.assertFalse((path/'model.pkl').exists())

    def test_bad_risk(self):
        with self.assertRaises(ValueError): Risk(risk_per_trade=0)
        with self.assertRaises(ValueError): Risk(fee=float('nan'))

if __name__=='__main__': unittest.main()
