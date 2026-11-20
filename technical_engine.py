"""Causal, rule-based spot research. No order placement or KDJ dependency."""
import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from engine import validate, fetch
from execution import ExitSettings, simulate_trailing
from research_v3 import Risk, passes

RULES = {
 'trend_pullback': 'EMA50 > EMA200, ADX >= 25, price reclaims EMA20 after pullback, RSI 45..70',
 'range_breakout': 'Close crosses previous 20-bar high, EMA50 > EMA200, volume >= 1.5x previous 20-bar mean',
 'mean_reversion': 'ADX < 20, price reclaims lower Bollinger band (20,2), RSI < 45',
 'macd_momentum': 'DIF crosses above DEA (12,26,9), EMA50 > EMA200, volume >= previous 20-bar mean',
 'volatility_breakout': 'Close crosses previous 20-bar high after prior Bollinger width <= prior rolling 100-bar 20th percentile, volume >= 1.5x mean',
}

def cross(a,b):
    return (a>b)&(a.shift(1)<=b.shift(1))

def indicators(d):
    c=d.close;f=pd.DataFrame(index=d.index)
    for n in (20,50,200):f[f'ema{n}']=c.ewm(span=n,adjust=False,min_periods=n).mean()
    delta=c.diff();up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean();down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    f['rsi']=100*up/(up+down).replace(0,np.nan);f.loc[(up+down)==0,'rsi']=50
    tr=pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean();f['atr']=atr/c
    rise=d.high.diff();fall=-d.low.diff()
    plus=rise.where((rise>fall)&(rise>0),0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    minus=fall.where((fall>rise)&(fall>0),0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    dx=100*(plus-minus).abs()/(plus+minus).replace(0,np.nan)
    f['adx']=dx.fillna(0).ewm(alpha=1/14,adjust=False,min_periods=28).mean()
    mid=c.rolling(20).mean();std=c.rolling(20).std(ddof=0)
    f['bb_lower']=mid-2*std;f['bb_upper']=mid+2*std
    width=4*std/mid;f['width']=width
    f['squeeze']=width.shift(1)<=width.shift(1).rolling(100,min_periods=100).quantile(.2)
    f['prior_high']=d.high.shift(1).rolling(20).max()
    f['volume_ratio']=d.volume/d.volume.shift(1).rolling(20).mean().replace(0,np.nan)
    f['dif']=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    f['dea']=f.dif.ewm(span=9,adjust=False).mean()
    f['regime']=np.select([(f.adx>=25)&(f.ema50>f.ema200),(f.adx>=25)&(f.ema50<f.ema200),f.adx<20],['uptrend','downtrend','range'],default='transition')
    return f

def signals(d,f):
    trend=(f.regime=='uptrend'); liquid=f.volume_ratio>=1
    ready=f[['ema200','atr','rsi','adx']].notna().all(axis=1)
    rules={
      'trend_pullback':trend&cross(d.close,f.ema20)&f.rsi.between(45,70)&liquid,
      'range_breakout':trend&cross(d.close,f.prior_high)&(f.volume_ratio>=1.5),
      'mean_reversion':(f.regime=='range')&cross(d.close,f.bb_lower)&(f.rsi<45)&liquid,
      'macd_momentum':trend&cross(f.dif,f.dea)&liquid,
      'volatility_breakout':f['squeeze']&cross(d.close,f.prior_high)&(f.volume_ratio>=1.5)&(f.regime!='downtrend'),
    }
    return {name:(v&ready).fillna(False).to_numpy(dtype=bool) for name,v in rules.items()}

def exits():
    return [ExitSettings(stop,target,'atr',trail,activation) for stop in (.01,.02) for target in (.02,.04,.06) for trail in (1.5,2.5) for activation in (.01,)]

def load_closed(csv, interval=15, now_ms=None):
    d=validate(pd.read_csv(csv));step=interval*60000
    if len(d)<1000:raise ValueError('At least 1000 candles required')
    if not (d.timestamp%step==0).all() or not (d.timestamp.diff().dropna()==step).all():raise ValueError('Dataset interval mismatch')
    now_ms=int(time.time()*1000) if now_ms is None else now_ms
    d=d[d.timestamp+step<=now_ms].reset_index(drop=True)
    if len(d)<1000:raise ValueError('Insufficient closed candles')
    return d

def run(csv,output,symbol='ICNTUSDT',interval=15):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Fresh output prevents stale final-test files appearing after a rejected run.
    for name in ('final_trades.csv','final_equity.csv'): (out/name).unlink(missing_ok=True)
    raw=load_closed(csv,interval);cut=int(len(raw)*.8);dev=raw.iloc[:cut].copy()
    f=indicators(dev);sig=signals(dev,f);warmup=250
    edges=np.linspace(warmup+int((cut-warmup)*.45),cut,4,dtype=int)
    risk=Risk();ranking=[]
    for name in RULES:
        for setting in exits():
            folds=[simulate_trailing(dev.iloc[a:b],f.iloc[a:b],sig[name][a:b],setting,risk) for a,b in zip(edges[:-1],edges[1:])]
            ranking.append(dict(strategy=name,settings=asdict(setting),folds=folds,qualified=all(passes(m) for m in folds),mean_return=float(np.mean([m['net_return'] for m in folds])),worst_return=float(min(m['net_return'] for m in folds)),total_trades=sum(m['trades'] for m in folds)))
    ranking.sort(key=lambda x:(x['mean_return'],x['worst_return']),reverse=True)
    selected=next((r for r in ranking if r['qualified']),None)
    report=dict(version='technical-engine-1.0',symbol=symbol,interval_minutes=interval,rows=len(raw),development_rows=cut,final_rows_reserved=len(raw)-cut,folds=[dict(start_timestamp=int(dev.timestamp.iloc[a]),end_timestamp=int(dev.timestamp.iloc[b-1])) for a,b in zip(edges[:-1],edges[1:])],rules=RULES,risk=asdict(risk),candidates=ranking,selected=selected,live_enabled=False,status='NO_QUALIFIED_STRATEGY',notes=['Rules are research hypotheses, not best-strategy claims.','Development history reused; obtain fresh future paper results.','Final 20% excluded from indicator computation and selection unless candidate qualifies.','Single position; next-open execution, conservative same-bar stop priority, close-updated trailing effective next candle.','Fees 0.1% and slippage 0.05% per side; no order book, exchange precision or minimum order model.','No machine learning training or live orders in this version.'])
    if selected:
        ff=indicators(raw);ss=signals(raw,ff)[selected['strategy']];setting=ExitSettings(**selected['settings'])
        result=simulate_trailing(raw.iloc[cut:],ff.iloc[cut:],ss[cut:],setting,risk,True)
        trades=result.pop('trade_log');equity=result.pop('equity_curve');result.pop('stop_trace')
        stress=simulate_trailing(raw.iloc[cut:],ff.iloc[cut:],ss[cut:],setting,Risk(fee=.002,slippage=.001))
        report.update(final=result,cost_stress=stress,status='PAPER_CANDIDATE_ONLY' if passes(result,20) and stress['net_return']>0 else 'FAILED_FINAL_TEST')
        pd.DataFrame(trades).to_csv(out/'final_trades.csv',index=False)
        pd.DataFrame({'timestamp':raw.timestamp.iloc[cut:].to_numpy(),'equity':equity}).to_csv(out/'final_equity.csv',index=False)
    best=ranking[0];logs=[]
    for i,(a,b) in enumerate(zip(edges[:-1],edges[1:]),1):
        m=simulate_trailing(dev.iloc[a:b],f.iloc[a:b],sig[best['strategy']][a:b],ExitSettings(**best['settings']),risk,True)
        logs.extend(dict(fold=i,**t) for t in m['trade_log'])
    pd.DataFrame(logs).to_csv(out/'best_validation_trades.csv',index=False)
    audit=pd.DataFrame({'timestamp':dev.timestamp,'regime':f.regime,**sig});audit.to_csv(out/'development_signals.csv',index=False)
    pd.DataFrame([dict(strategy=r['strategy'],**r['settings'],mean_return=r['mean_return'],worst_return=r['worst_return'],trades=r['total_trades'],qualified=r['qualified']) for r in ranking]).to_csv(out/'comparison.csv',index=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (r['strategy'],f"{100*r['mean_return']:.2f}%",f"{100*r['worst_return']:.2f}%",r['total_trades'],r['qualified'],r['settings']))+'</tr>' for r in ranking)
    (out/'index.html').write_text('<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>محرك التحليل الفني</title><style>body{font:16px system-ui;background:#101827;color:white;padding:20px}td,th{padding:12px;border-bottom:1px solid #657}section{overflow:auto}</style><h1>محرك التحليل الفني — '+html.escape(symbol)+'</h1><p>'+report['status']+'</p><p>خمس استراتيجيات بدون KDJ؛ نتائج بحثية بعد التكاليف، لا ضمان للربح. آخر 20% محجوزة. الترتيب حسب متوسط التحقق.</p><input placeholder="تصفية اسم الاستراتيجية" oninput="for(let r of document.querySelectorAll(\'tbody tr\'))r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase())"><section><table><thead><tr><th>الاستراتيجية</th><th>متوسط العائد</th><th>أسوأ فترة</th><th>الصفقات</th><th>مقبول</th><th>إعدادات الخروج</th></tr></thead><tbody>'+rows+'</tbody></table></section></html>')
    print(json.dumps(dict(status=report['status'],candidates=len(ranking),best=best),indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--symbol',choices=['ICNTUSDT','BTCUSDT'],default='ICNTUSDT');p.add_argument('--csv');p.add_argument('--output',default='technical_outputs');p.add_argument('--fetch-days',type=int,default=0);a=p.parse_args()
    csv=a.csv or ('data/ICNTUSDT_spot_15m.csv' if a.symbol=='ICNTUSDT' else 'data/candles.csv')
    if a.fetch_days:
        if not 11<=a.fetch_days<=730:p.error('fetch-days must be 11..730')
        Path(a.output).mkdir(parents=True,exist_ok=True);csv=str(Path(a.output)/'candles.csv');fetch(a.symbol,'15',a.fetch_days,csv)
    run(csv,a.output,a.symbol)
