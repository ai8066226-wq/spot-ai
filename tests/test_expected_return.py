import pickle
import unittest
import numpy as np
import pandas as pd
from expected_return import EntryPolicy,entry_signal,temporal_indices,build_bundle,predict_bundle,predictive_metrics,gated_signal,FEATURES
from research_v3 import Risk

class ExpectedReturnTests(unittest.TestCase):
    def test_expected_gain_does_not_override_tail_risk(self):
        signal=entry_signal(np.array([.01,.01,.001,np.nan]),np.array([-.02,-.05,-.01,-.01]))
        np.testing.assert_equal(signal,[True,False,False,False])
    def test_fixed_baseline_zero_skill(self):
        y=np.array([-.01,.02]*50);p=np.full(100,.005);q=np.full(100,-.01)
        m=predictive_metrics(y,p,q,p,q)
        self.assertAlmostEqual(m['mse_skill'],0);self.assertAlmostEqual(m['q10_pinball_skill'],0)
    def test_purge_excludes_cross_boundary_outcomes(self):
        ends=np.arange(1300)+16
        train,cal=temporal_indices(ends,1000,16)
        self.assertLess(ends[train].max(),cal.min())
        self.assertLess(ends[cal].max(),1000)
    def test_no_partial_horizon_entries(self):
        p=dict(mean=np.full(30,.01),q10=np.full(30,-.02))
        s=gated_signal(p,30,16,EntryPolicy())
        self.assertTrue(s[:14].all());self.assertFalse(s[14:].any())
    def test_future_changes_do_not_change_fit_or_serialized_inference(self):
        rng=np.random.default_rng(42);n=1300
        f=pd.DataFrame({c:rng.normal(size=n) for c in FEATURES});target=rng.normal(0,.02,n);ends=np.arange(n)+16
        risk=Risk();b=build_bundle(f,target,ends,1000,risk)
        newer=f.copy();newer.loc[1000:]*=100
        changed_target=target.copy();changed_target[1000:]=99
        other=build_bundle(newer,changed_target,ends,1000,risk)
        a=predict_bundle(b,f.iloc[1000:]);c=predict_bundle(other,f.iloc[1000:]);restored=predict_bundle(pickle.loads(pickle.dumps(b)),f.iloc[1000:])
        for name in a:
            np.testing.assert_allclose(a[name]['mean'],c[name]['mean'])
            np.testing.assert_allclose(a[name]['q10'],c[name]['q10'])
            np.testing.assert_allclose(a[name]['mean'],restored[name]['mean'])
            self.assertTrue((a[name]['q10']<=a[name]['mean']).all())
