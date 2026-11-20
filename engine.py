"""Research only: closed candles, long/flat, next-open execution."""
import argparse, json, pickle, time, urllib.request, urllib.parse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def validate(d):
    cols=['timestamp','open','high','low','close','volume']
    if not set(cols)<=set(d): raise ValueError('Missing OHLCV columns')
    d=d[cols].copy().apply(pd.to_numeric, errors='raise').sort_values('timestamp').reset_index(drop=True)
    if d.isna().any().any() or not np.isfinite(d.to_numpy()).all(): raise ValueError('Nonfinite data')
    if d.timestamp.duplicated().any() or (d.volume<0).any() or (d[['open','high','low','close']]<=0).any().any(): raise ValueError('Invalid prices, volume or duplicate timestamps')
    if ((d.high<d[['open','close','low']].max(axis=1)) | (d.low>d[['open','close','high']].min(axis=1))).any(): raise ValueError('Invalid candle range')
    if len(d)>2 and d.timestamp.diff().dropna().nunique()!=1: raise ValueError('Missing or irregular candles; repair source data first')
    return d


def fetch(symbol, interval, days, dest):
    step=int(interval)*60000
    end=int(time.time()*1000)//step*step-1
    start=end-days*86400000+1
    rows=[]; cursor=end
    while cursor>=start:
        q=urllib.parse.urlencode(dict(category='spot',symbol=symbol,interval=interval,start=start,end=cursor,limit=1000))
        with urllib.request.urlopen('https://api.bybit.com/v5/market/kline?'+q, timeout=30) as r: body=json.load(r)
        if body.get('retCode')!=0: raise RuntimeError(str(body))
        page=body['result']['list']
        if not page: break
        rows.extend([x[:6] for x in page]); oldest=min(int(x[0]) for x in page)
        if oldest>cursor: raise RuntimeError('Pagination failed')
        cursor=oldest-1; time.sleep(.15)
    d=pd.DataFrame(rows,columns=['timestamp','open','high','low','close','volume'])
    d=d.drop_duplicates('timestamp'); d=validate(d)
    if len(d)<1000: raise ValueError('Insufficient history')
    d.to_csv(dest,index=False)


def features(d):
    c=d.close; delta=c.diff(); f=pd.DataFrame(index=d.index)
    f['return']=c.pct_change(); f['body']=(c-d.open)/d.open
    f['range']=(d.high-d.low)/c
    f['volume_ratio']=d.volume/d.volume.rolling(20).mean().replace(0,np.nan)
    for n in (7,14,28):
        up=delta.clip(lower=0).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
        down=(-delta.clip(upper=0)).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
        f[f'rsi_{n}']=100*up/(up+down).replace(0,np.nan)
    for n in (10,20,50): f[f'ema_{n}']=c/c.ewm(span=n,adjust=False,min_periods=n).mean()-1
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    f['macd']=(macd-macd.ewm(span=9,adjust=False).mean())/c
    f['bollinger']=(c-c.rolling(20).mean())/c.rolling(20).std().replace(0,np.nan)
    tr=pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1)
    f['atr']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/c
    return f.replace([np.inf,-np.inf],np.nan)


def backtest(d, signal, fee=.001, slip=.0005):
    if not 0<=fee<.1 or not 0<=slip<.1: raise ValueError('Invalid cost assumption')
    # signal[i] uses closed candle i; trade at open i+1, mark at next open.
    cash=1.; units=0.; peak=1.; dd=0.; entry=None; trades=[]; curve=[1.]
    for i in range(len(d)-1):
        p=float(d.open.iloc[i+1]); want=bool(signal[i])
        if want and units==0:
            entry=cash; units=cash/(p*(1+slip)*(1+fee)); cash=0.
        elif not want and units>0:
            cash=units*p*(1-slip)*(1-fee); units=0.; trades.append(cash/entry-1)
        equity=cash+units*p; peak=max(peak,equity); dd=max(dd,1-equity/peak); curve.append(equity)
    if units>0:
        cash=units*float(d.close.iloc[-1])*(1-slip)*(1-fee); trades.append(cash/entry-1)
        peak=max(peak,cash); dd=max(dd,1-cash/peak)
    gains=sum(x for x in trades if x>0); losses=-sum(x for x in trades if x<0)
    return dict(net_return=cash-1,max_drawdown=dd,trades=len(trades),win_rate=sum(x>0 for x in trades)/len(trades) if trades else 0,profit_factor=gains/losses if losses else None)


def candidates(f):
    s={}
    for n in (7,14,28):
        for t in (45,55): s[f'rsi_{n}_above_{t}']=(f[f'rsi_{n}']>t).to_numpy()
    for n in (10,20,50): s[f'ema_{n}_trend']=(f[f'ema_{n}']>0).to_numpy()
    s['macd_positive']=(f.macd>0).to_numpy()
    s['bollinger_recovery']=(f.bollinger< -1).to_numpy()
    for name in list(s):
        if not name.startswith('ema'): s[name+'+ema_50']=s[name] & (f.ema_50>0).to_numpy()
    return s


def run(csv, out, fee=.001, slip=.0005):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    d=validate(pd.read_csv(csv)); f=features(d)
    d=d.iloc[60:].reset_index(drop=True); f=f.iloc[60:].reset_index(drop=True)
    if len(d)<1500: raise ValueError('Need at least 1560 contiguous candles')
    cut=int(len(d)*.8); dev=d.iloc[:cut]; sig=candidates(f)
    # Three chronological validation windows. Final 20% is never used to choose strategies.
    edges=np.linspace(int(cut*.4),cut,4,dtype=int)
    ranking=[]
    for name,s in sig.items():
        folds=[backtest(d.iloc[a:b].reset_index(drop=True),s[a:b],fee,slip) for a,b in zip(edges[:-1],edges[1:])]
        acceptable=all(x['net_return']>0 and x['max_drawdown']<=.2 and x['trades']>=5 for x in folds)
        score=float(np.mean([x['net_return']-x['max_drawdown'] for x in folds]))
        ranking.append(dict(name=name,score=score,accepted=acceptable,reason='positive each validation fold, drawdown <=20%, >=5 trades/fold' if acceptable else 'failed profitability, drawdown or sample-size filter',folds=folds))
    ranking.sort(key=lambda x:x['score'],reverse=True)
    chosen=next((r['name'] for r in ranking if r['accepted']),None)
    rule_final=backtest(d.iloc[cut:].reset_index(drop=True),sig[chosen][cut:],fee,slip) if chosen else None
    # Predict executable next-open to open four bars later. Purge future labels at boundaries.
    horizon=4; target_return=d.open.shift(-(horizon+1))/d.open.shift(-1)-1
    cost=2*(fee+slip); y=np.where(target_return>cost,2,np.where(target_return< -cost,0,1))
    base=['return','body','range','volume_ratio']
    # Group ablation selects an indicator family on pre-final validation only.
    groups={'price_volume':base,'rsi':base+[x for x in f if x.startswith('rsi')], 'ema':base+[x for x in f if x.startswith('ema')], 'macd':base+['macd'], 'bollinger':base+['bollinger'],'atr':base+['atr'],'all':list(f.columns)}
    split=int(cut*.7); train_end=split-horizon-1; val_end=cut-horizon-1
    models=[]
    for name,cols in groups.items():
        m=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,l2_regularization=5,random_state=42)
        m.fit(f[cols].iloc[:train_end],y[:train_end])
        p=m.predict_proba(f[cols].iloc[split:val_end]); buy=p[:,list(m.classes_).index(2)]>.6 if 2 in m.classes_ else np.zeros(len(p),bool)
        metrics=backtest(d.iloc[split:val_end].reset_index(drop=True),buy,fee,slip)
        models.append((metrics['net_return']-metrics['max_drawdown'],name,cols,metrics))
    models.sort(key=lambda x:x[0],reverse=True); _,name,cols,vm=models[0]
    model=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,l2_regularization=5,random_state=42)
    model.fit(f[cols].iloc[:cut-horizon-1],y[:cut-horizon-1])
    p=model.predict_proba(f[cols].iloc[cut:]); buy=p[:,list(model.classes_).index(2)]>.6 if 2 in model.classes_ else np.zeros(len(p),bool)
    final=backtest(d.iloc[cut:].reset_index(drop=True),buy,fee,slip)
    report=dict(data_rows=len(d),synthetic='UNKNOWN: check data provenance',fee_per_side=fee,slippage_per_side=slip,final_start_timestamp=int(d.timestamp.iloc[cut]),rankings=ranking,selected_rule=chosen,rule_final=rule_final,model_family=name,model_validation=[dict(group=n,metrics=v) for _,n,c,v in models],model_final=final,model_approved=vm['net_return']>0 and vm['trades']>=5 and final['net_return']>0 and final['trades']>=5 and final['max_drawdown']<=.2,buy_hold_final=backtest(d.iloc[cut:].reset_index(drop=True),np.ones(len(d)-cut,bool),fee,slip),notes=['Research only; no order execution.','Reusing the final period for tuning invalidates its independence.','Minimum samples and thresholds are research assumptions, not guarantees.','No precision, liquidity or minimum-order constraints modeled.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (out/'model.pkl').open('wb') as h: pickle.dump(dict(model=model,columns=cols,horizon=horizon,threshold=.6),h)
    print(json.dumps(dict(report=str(out/'report.json'),selected_rule=chosen,model_family=name,model_approved=report['model_approved'])))


def demo(dest):
    rng=np.random.default_rng(42); n=6000; c=100*np.exp(np.cumsum(rng.normal(0,.003,n))); o=np.r_[c[0],c[:-1]]
    pd.DataFrame(dict(timestamp=1700000100000+np.arange(n)*900000,open=o,high=np.maximum(o,c)*1.002,low=np.minimum(o,c)*.998,close=c,volume=rng.uniform(10,100,n))).to_csv(dest,index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('fetch'); a.add_argument('--symbol',default='BTCUSDT'); a.add_argument('--interval',choices=['1','3','5','15','30','60','120','240','360','720'],default='15'); a.add_argument('--days',type=int,default=180); a.add_argument('--output',default='candles.csv')
    a=sub.add_parser('demo'); a.add_argument('--output',default='SYNTHETIC.csv')
    a=sub.add_parser('research'); a.add_argument('csv'); a.add_argument('--output',default='results'); a.add_argument('--fee',type=float,default=.001); a.add_argument('--slippage',type=float,default=.0005)
    args=p.parse_args()
    if args.cmd=='fetch':
        if args.days<=0: p.error('days must be positive')
        fetch(args.symbol,args.interval,args.days,args.output)
    elif args.cmd=='demo': demo(args.output)
    else: run(args.csv,args.output,args.fee,args.slippage)
