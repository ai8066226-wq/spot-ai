"""Causal J-K peak-gap trailing exit, confirmed at close, filled next open."""
from dataclasses import asdict
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from engine import validate
from kdj_research import kdj,cross_up
from research_v3 import Risk,enriched,stop_fraction,metrics,passes,simulate


def simulate_gap(d,f,gap,risk=Risk(),details=False,prior_gap=None):
    gap=np.asarray(gap,dtype=float)
    if len(d)!=len(f) or len(d)!=len(gap):raise ValueError('Length mismatch')
    if not np.isfinite(gap).all() or not np.isfinite(f.atr).all():raise ValueError('Invalid indicator values')
    signal=cross_up(gap,np.zeros(len(gap)))
    if prior_gap is not None and len(gap):signal[0]=gap[0]>0 and prior_gap<=0
    o=d.open.to_numpy();h=d.high.to_numpy();l=d.low.to_numpy();c=d.close.to_numpy();ts=d.timestamp.to_numpy();atr=f.atr.to_numpy()
    cash=1.;units=0.;pos=None;pending=None;peak=1.;dd=0.;halted=False;blocked_until=0
    day=None;day_start=1.;day_block=False;equity=[1.];trades=[];exposure=0
    for j in range(1,len(d)):
        current_day=int(ts[j]//86400000)
        if current_day!=day:day=current_day;day_start=equity[-1];day_block=False
        exited=False
        def close_position(price,reason):
            nonlocal cash,units,pos,pending,blocked_until,exited
            proceeds=units*price*(1-risk.slippage)*(1-risk.fee);pnl=proceeds-pos['spend']
            cash+=proceeds
            trades.append(dict(**pos,exit_index=j,exit_timestamp=int(ts[j]),exit=float(price*(1-risk.slippage)),reason=reason,exit_signal_timestamp=pending,quantity=units,pnl=float(pnl),**{'return':float(pnl/pos['spend'])}))
            units=0.;pos=None;pending=None;blocked_until=j+risk.cooldown+1;exited=True
        # Any previously confirmed distance exit executes before this candle's high/low.
        if units:
            exposure+=1
            if o[j]<=pos['stop']:close_position(float(o[j]),'stop_gap')
            elif halted:close_position(float(o[j]),'drawdown_halt_next_open')
            elif pending is not None:close_position(float(o[j]),'gap_shrink_next_open')
        if not units and not exited and not halted and not day_block and j>=blocked_until and j<len(d)-1 and signal[j-1]:
            entry=float(o[j]*(1+risk.slippage));fraction=stop_fraction(atr[j-1],risk)
            stop=entry*(1-fraction)
            unit_loss=entry*(1+risk.fee)-stop*(1-risk.slippage)*(1-risk.fee)
            units=min(cash*risk.max_allocation/(entry*(1+risk.fee)),cash*risk.risk_per_trade/unit_loss)
            spend=units*entry*(1+risk.fee);cash-=spend
            pos=dict(entry_index=j,entry_timestamp=int(ts[j]),entry_signal_timestamp=int(ts[j-1]),entry=entry,stop=float(stop),spend=float(spend),entry_equity=equity[-1],planned_stop_risk=float(units*unit_loss),max_gap=float(gap[j-1]),gap_at_exit_signal=None)
            exposure+=1
        if units:
            adverse=float(o[j] if o[j]<=pos['stop'] else max(l[j],pos['stop']))
            low_mark=cash+units*adverse*(1-risk.slippage)*(1-risk.fee)
            dd=max(dd,1-low_mark/peak)
            if l[j]<=pos['stop']:close_position(float(pos['stop']),'stop')
            elif j==len(d)-1:close_position(float(c[j]),'end_of_period')
        if units:
            # Only gap values observed so far can update the peak.
            if gap[j]<pos['max_gap']-1e-10:
                pending=int(ts[j]);pos['gap_at_exit_signal']=float(gap[j])
            else:pos['max_gap']=max(pos['max_gap'],float(gap[j]))
        mark=cash+units*c[j]*(1-risk.slippage)*(1-risk.fee)
        peak=max(peak,mark);dd=max(dd,1-mark/peak);equity.append(float(mark))
        if dd>=risk.max_drawdown:halted=True
        if mark<=day_start*(1-risk.daily_loss):day_block=True
    result=metrics(trades,equity,dd,exposure,halted)
    if details:result.update(trade_log=trades,equity_curve=equity)
    return result


def run(csv,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    raw=validate(pd.read_csv(csv));n=len(raw)-100;cut=int(n*.8)
    if n<5000:raise ValueError('Need 5100 candles')
    dev=raw.iloc[:cut+100];indicator=kdj(dev).iloc[100:].reset_index(drop=True)
    f=enriched(dev).iloc[100:].reset_index(drop=True);d=dev.iloc[100:].reset_index(drop=True)
    gap=(indicator.J-indicator.K).to_numpy();risk=Risk();edges=np.linspace(int(cut*.45),cut,4,dtype=int)
    records=[];logs=[];curves=[]
    for fold,(a,b) in enumerate(zip(edges[:-1],edges[1:]),1):
        frame=d.iloc[a:b];ff=f.iloc[a:b];g=gap[a:b]
        result=simulate_gap(frame,ff,g,risk,True,prior_gap=gap[a-1]);trades=result.pop('trade_log');equity=result.pop('equity_curve')
        stressed=simulate_gap(frame,ff,g,Risk(fee=risk.fee*2,slippage=risk.slippage*2),prior_gap=gap[a-1])
        baseline_signal=cross_up(g,np.zeros(len(g)));baseline_signal[0]=g[0]>0 and gap[a-1]<=0;baseline_signal[np.arange(len(g))>=len(g)-risk.max_hold]=False
        baseline=simulate(frame,ff,baseline_signal,risk)
        records.append(dict(fold=fold,first_timestamp=int(frame.timestamp.iloc[0]),last_timestamp=int(frame.timestamp.iloc[-1]),metrics=result,cost_stress=stressed,previous_fixed_exit=baseline,qualified=passes(result),exit_reasons={reason:sum(t['reason']==reason for t in trades) for reason in sorted(set(t['reason'] for t in trades))}))
        logs.extend(dict(fold=fold,**t) for t in trades)
        curves.extend(dict(fold=fold,timestamp=int(timestamp),equity=value) for timestamp,value in zip(frame.timestamp.iloc[1:],equity[1:]))
    report=dict(version='kdj-gap-shrink-1',symbol='ICNTUSDT',parameters=[9,3,3],entry='Closed-candle upward J/K crossover; buy following open',exit='First closed candle with J-K below the running peak since entry signal; sell following open',safety='Fixed ATR stop, daily and drawdown risk gates retained; no target and no maximum holding duration',risk=asdict(risk),ignored_risk_fields=['reward_risk','max_hold'],development_rows=cut,final_rows_excluded=n-cut,folds=records,status='VALIDATION_CANDIDATE_ONLY' if all(r['qualified'] for r in records) else 'REJECTED_KEEP_CASH',live_enabled=False,notes=['Peak is updated from past and current closed candles only, never the future maximum.','Peak starts at the confirmed entry-signal gap; shrink on entry candle can schedule next-open exit.','Equality does not exit. Numerical tolerance is 1e-10 indicator units.','Prior fixed-exit baseline excludes entries lacking a full maximum horizon; gap exit has no time horizon and closes at evaluation end.','Initial position is flat in each evaluation window. No final-period evaluation or production activation.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    pd.DataFrame(logs).to_csv(out/'trades.csv',index=False);pd.DataFrame(curves).to_csv(out/'equity.csv',index=False)
    pd.DataFrame([dict(fold=r['fold'],**r['metrics']) for r in records]).to_csv(out/'comparison.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default='data/ICNTUSDT_spot_15m.csv');p.add_argument('--output',default='kdj_gap_outputs');a=p.parse_args();run(a.csv,a.output)
