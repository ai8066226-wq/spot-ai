"""Bounded, reproducible parameter search. Final 20% excluded from tuning."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_v3 import (Risk,enriched,labels,fit_bundle,predict,simulate,passes,block_bootstrap)
from engine import validate


def grid():
    return [Risk(max_hold=h,stop_atr=s,reward_risk=r)
            for h in (16,32,64) for s in (1.,1.5) for r in (1.5,2.5)]


def score(folds):
    returns=np.array([f['net_return'] for f in folds])
    sufficient=len(folds)==3 and all(f['trades']>=5 for f in folds)
    qualified=len(folds)==3 and all(passes(f) for f in folds)
    value=float(np.median(returns)-2*max(f['max_drawdown'] for f in folds)-np.std(returns))
    return sufficient,qualified,value


def tune(csv,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    raw=validate(pd.read_csv(csv)); n=len(raw)-100
    if n<5000: raise ValueError('Need at least 5100 candles for this search')
    cut=int(n*.8)
    # Slice before computing features AND target labels. No test prices inform selection.
    dev_raw=raw.iloc[:cut+100].copy()
    f=enriched(dev_raw).iloc[100:].reset_index(drop=True)
    d=dev_raw.iloc[100:].reset_index(drop=True)
    groups={'trend':['return','body','range','volume_ratio','volume_change','ema_10','ema_20','ema_50','macd','momentum_12','momentum_48'],'all':list(f.columns)}
    edges=np.linspace(int(cut*.45),cut,4,dtype=int)
    candidates=[];diagnostics=[]
    for profile,risk in enumerate(grid()):
        y,ends=labels(d,f,risk)
        local={}
        for a,b in zip(edges[:-1],edges[1:]):
            for group,cols in groups.items():
                bundle=fit_bundle(f,y,ends,int(a),cols,risk)
                if bundle is None: continue
                for mode in ('boosting','ensemble'):
                    p=predict(bundle,f.iloc[a:b],mode)
                    diagnostics.append(dict(profile=profile,group=group,mode=mode,start=int(a),end=int(b),positive_label_rate=float(np.nanmean(y[:a-risk.max_hold-1])),probability_p50=float(np.quantile(p,.5)),probability_p95=float(np.quantile(p,.95)),probability_max=float(p.max())))
                    for threshold in (.30,.40,.50,.60):
                        key=(group,mode,threshold)
                        m=simulate(d.iloc[a:b],f.iloc[a:b],p>=threshold,risk)
                        local.setdefault(key,[]).append(m)
        for (group,mode,threshold),folds in local.items():
            sufficient,qualified,value=score(folds)
            candidates.append(dict(profile=profile,risk=asdict(risk),group=group,mode=mode,threshold=threshold,folds=folds,sufficient=sufficient,qualified=qualified,score=value,total_validation_trades=sum(x['trades'] for x in folds)))
        print(f'Risk profile {profile+1}/12 complete',flush=True)
    candidates.sort(key=lambda x:(x['qualified'],x['sufficient'],x['score']),reverse=True)
    qualified=[c for c in candidates if c['qualified']]
    active=[c for c in candidates if c['sufficient']]
    selected=qualified[0] if qualified else None
    report=dict(version='bounded-search-1',symbol='ICNTUSDT',market='spot',source='User-supplied Bybit CSV',rows=len(raw),development_rows=cut,untouched_final_rows=n-cut,final_first_timestamp=int(raw.timestamp.iloc[cut+100]),settings_tried=len(candidates),risk_profiles=12,rankings=candidates,diagnostics=diagnostics,selected=selected,best_exploratory=active[0] if active else None,status='NO_QUALIFIED_SETTINGS',live_enabled=False,notes=['Ranking uses development-only forward validation.','Every candidate must pass the original PF>=1.5 and payoff>=1.5 gates in all three windows.','The exploratory best is not approved and may lose money.','Previously explored development results influenced the choice of this new grid; a future paper test is still required.','A finite search is not a formal correction for multiple testing.'])
    if selected:
        risk=Risk(**selected['risk'])
        y,ends=labels(d,f,risk)
        bundle=fit_bundle(f,y,ends,cut,groups[selected['group']],risk)
        full_f=enriched(raw).iloc[100:].reset_index(drop=True)
        test=raw.iloc[100+cut:].reset_index(drop=True);test_f=full_f.iloc[cut:].reset_index(drop=True)
        p=predict(bundle,test_f,selected['mode']);signal=p>=selected['threshold']
        final=simulate(test,test_f,signal,risk,details=True)
        trades=final.pop('trade_log');curve=final.pop('equity_curve')
        stressed=simulate(test,test_f,signal,Risk(**{**selected['risk'],'fee':risk.fee*2,'slippage':risk.slippage*2}))
        boot=block_bootstrap(trades)
        accepted=passes(final,20) and stressed['net_return']>0 and boot['available'] and boot['return_p05']>0
        report.update(status='PAPER_CANDIDATE_ONLY' if accepted else 'FAILED_FINAL_TEST',final=final,cost_stress=stressed,bootstrap=boot)
        pd.DataFrame(trades).to_csv(out/'tuning_final_trades.csv',index=False)
        pd.DataFrame(dict(timestamp=test.timestamp.iloc[1:].to_numpy(),equity=curve[1:])).to_csv(out/'tuning_final_equity.csv',index=False)
    (out/'tuning_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    rows=[]
    for c in candidates:
        rows.append(dict(holding_candles=c['risk']['max_hold'],stop_atr=c['risk']['stop_atr'],reward_risk=c['risk']['reward_risk'],features=c['group'],model=c['mode'],threshold=c['threshold'],qualified=c['qualified'],sufficient=c['sufficient'],score=c['score'],trades=c['total_validation_trades'],**{f'fold_{i+1}_return':m['net_return'] for i,m in enumerate(c['folds'])}))
    pd.DataFrame(rows).to_csv(out/'settings_comparison.csv',index=False)
    print(json.dumps({k:v for k,v in report.items() if k in ('status','settings_tried','selected','best_exploratory','final','cost_stress')},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default='data/ICNTUSDT_spot_15m.csv');p.add_argument('--output',default='tuning_outputs');a=p.parse_args();tune(a.csv,a.output)
