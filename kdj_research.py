"""Fixed KDJ(9,3,3) crossover experiment, development windows only."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from engine import validate
from research_v3 import Risk,enriched,simulate,passes


def kdj(d,period=9,k_smoothing=3,d_smoothing=3):
    if min(period,k_smoothing,d_smoothing)<1:raise ValueError('Invalid period')
    low=d.low.rolling(period,min_periods=period).min()
    high=d.high.rolling(period,min_periods=period).max()
    span=high-low
    rsv=(100*(d.close-low)/span.replace(0,np.nan)).where(span!=0,50.)
    kval=np.full(len(d),np.nan);dval=np.full(len(d),np.nan);pk=pdv=50.
    for i,value in enumerate(rsv.to_numpy()):
        if not np.isfinite(value):continue
        pk=(1-1/k_smoothing)*pk+value/k_smoothing
        pdv=(1-1/d_smoothing)*pdv+pk/d_smoothing
        kval[i]=pk;dval[i]=pdv
    return pd.DataFrame(dict(K=kval,D=dval,J=3*kval-2*dval),index=d.index)


def cross_up(a,b):
    delta=np.asarray(a)-np.asarray(b)
    return np.r_[False,(delta[1:]>0)&(delta[:-1]<=0)]


def run(csv,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    raw=validate(pd.read_csv(csv));n=len(raw)-100;cut=int(n*.8)
    if n<5000:raise ValueError('Need 5100 candles')
    dev=raw.iloc[:cut+100]
    indicator=kdj(dev).iloc[100:].reset_index(drop=True)
    f=enriched(dev).iloc[100:].reset_index(drop=True);d=dev.iloc[100:].reset_index(drop=True)
    signal=cross_up(indicator.J,indicator.K)
    kd_signal=cross_up(indicator.K,indicator.D)
    if not np.array_equal(signal,kd_signal):raise ValueError('Numerical mismatch between equivalent crossovers')
    risk=Risk();edges=np.linspace(int(cut*.45),cut,4,dtype=int)
    rows=[];logs=[]
    for fold,(a,b) in enumerate(zip(edges[:-1],edges[1:]),1):
        selected=signal[a:b].copy()
        selected[np.arange(b-a)>=b-a-risk.max_hold]=False
        result=simulate(d.iloc[a:b],f.iloc[a:b],selected,risk,details=True)
        trades=result.pop('trade_log');result.pop('equity_curve')
        stress=simulate(d.iloc[a:b],f.iloc[a:b],selected,Risk(fee=risk.fee*2,slippage=risk.slippage*2))
        rows.append(dict(fold=fold,first_timestamp=int(d.timestamp.iloc[a]),last_timestamp=int(d.timestamp.iloc[b-1]),crossovers=int(selected.sum()),metrics=result,cost_stress=stress,qualified=passes(result)))
        for t in trades:logs.append(dict(fold=fold,**t))
    report=dict(symbol='ICNTUSDT',version='kdj-crossover-1',parameters=[9,3,3],entry='J crosses K upwards on closed candle; execute at following open',exit='Original risk engine only: 1.5 ATR stop, 2.5R target, max 16 candles; no crossover exit in this experiment',equivalent_to_k_cross_d=True,development_rows=cut,final_rows_excluded=n-cut,folds=rows,status='VALIDATION_CANDIDATE_ONLY' if all(r['qualified'] for r in rows) else 'REJECTED_KEEP_CASH',live_enabled=False,notes=['J=3K-2D implies J-K=2(K-D): the two upward crossovers coincide without clipping J.','K,D recursively start at 50; flat rolling ranges use RSV=50. Platforms may initialize differently.','No parameter search and no final-period evaluation.','Risk budget and assumed fees/slippage remain unchanged.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    pd.DataFrame(logs).to_csv(out/'trades.csv',index=False)
    pd.DataFrame([dict(fold=r['fold'],crossovers=r['crossovers'],**r['metrics']) for r in rows]).to_csv(out/'comparison.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default='data/ICNTUSDT_spot_15m.csv');p.add_argument('--output',default='kdj_outputs');a=p.parse_args();run(a.csv,a.output)
