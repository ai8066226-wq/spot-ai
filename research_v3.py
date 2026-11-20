"""Temporal model research with a long-only, risk-budgeted OHLC simulator.
No exchange orders. All risk limits are research assumptions, not guarantees.
"""
from dataclasses import asdict, dataclass
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from engine import validate, features


@dataclass(frozen=True)
class Risk:
    fee: float = .001
    slippage: float = .0005
    risk_per_trade: float = .005
    max_allocation: float = .25
    stop_atr: float = 1.5
    reward_risk: float = 2.5
    min_stop: float = .004
    max_stop: float = .04
    max_hold: int = 16
    daily_loss: float = .02
    max_drawdown: float = .10
    cooldown: int = 2

    def __post_init__(self):
        for key,value in asdict(self).items():
            if not np.isfinite(value): raise ValueError(f'Nonfinite risk value: {key}')
        if not 0<=self.fee<.1 or not 0<=self.slippage<.1: raise ValueError('Invalid costs')
        if not 0<self.risk_per_trade<=1 or not 0<self.max_allocation<=1: raise ValueError('Invalid sizing')
        if not 0<self.min_stop<=self.max_stop<1 or self.stop_atr<=0 or self.reward_risk<=0: raise ValueError('Invalid stops')
        if not 0<self.daily_loss<1 or not 0<self.max_drawdown<1: raise ValueError('Invalid risk gates')
        if not isinstance(self.max_hold,int) or self.max_hold<1 or self.cooldown<0: raise ValueError('Invalid holding period')


def enriched(d):
    f=features(d)
    for n in (3,12,48):
        f[f'momentum_{n}']=d.close.pct_change(n)
        f[f'volatility_{n}']=d.close.pct_change().rolling(n).std()
    f['upper_wick']=(d.high-d[['open','close']].max(axis=1))/d.close
    f['lower_wick']=(d[['open','close']].min(axis=1)-d.low)/d.close
    f['volume_change']=d.volume.pct_change()
    return f.replace([np.inf,-np.inf],np.nan)


def stop_fraction(atr,risk):
    return float(np.clip(atr*risk.stop_atr,risk.min_stop,risk.max_stop))


def bar_exit(o,h,l,c,stop,target,age,risk):
    # Stop gaps fill at the worse open. Profit gaps get no favorable improvement.
    if o<=stop: return float(o),'stop_gap'
    if o>=target: return float(target),'target_gap'
    # OHLC does not reveal ordering: assume stop first when both are touched.
    if l<=stop: return float(stop),'stop'
    if h>=target: return float(target),'target'
    if age>=risk.max_hold: return float(c),'timeout'
    return None,None


def labels(d,f,risk):
    n=len(d); y=np.full(n,np.nan); ends=np.full(n,-1,dtype=int)
    o=d.open.to_numpy(); h=d.high.to_numpy(); l=d.low.to_numpy(); c=d.close.to_numpy()
    atr=f.atr.to_numpy()
    for t in range(n-risk.max_hold):
        j=t+1; entry=o[j]*(1+risk.slippage)
        frac=stop_fraction(atr[t],risk)
        stop=entry*(1-frac); target=entry*(1+risk.reward_risk*frac)
        for k in range(j,j+risk.max_hold):
            price,_=bar_exit(o[k],h[k],l[k],c[k],stop,target,k-j+1,risk)
            if price is not None:
                net=price*(1-risk.slippage)*(1-risk.fee)/(entry*(1+risk.fee))-1
                y[t]=int(net>0); ends[t]=k; break
    return y,ends


def metrics(trades,equity,drawdown,exposure,halted):
    gains=sum(t['pnl'] for t in trades if t['pnl']>0)
    losses=-sum(t['pnl'] for t in trades if t['pnl']<0)
    wins=[t['return'] for t in trades if t['pnl']>0]
    lost=[-t['return'] for t in trades if t['pnl']<0]
    return dict(net_return=float(equity[-1]-1),max_drawdown=float(drawdown),trades=len(trades),win_rate=len(wins)/len(trades) if trades else 0.,profit_factor=gains/losses if losses else None,total_profit=float(gains),total_loss=float(losses),average_win=float(np.mean(wins)) if wins else 0.,average_loss=float(np.mean(lost)) if lost else 0.,payoff_ratio=float(np.mean(wins)/np.mean(lost)) if wins and lost else None,expectancy_per_trade=float(sum(t['pnl'] for t in trades)/len(trades)) if trades else 0.,exposure=float(exposure/max(1,len(equity)-1)),drawdown_halt=bool(halted))


def simulate(d,f,signal,risk,details=False):
    if len(d)!=len(f) or len(signal)!=len(d): raise ValueError('Length mismatch')
    o=d.open.to_numpy(); h=d.high.to_numpy(); l=d.low.to_numpy(); c=d.close.to_numpy()
    ts=d.timestamp.to_numpy(); atr=f.atr.to_numpy()
    cash=1.; units=0.; position=None; equity=[1.]; trades=[]; peak=1.; dd=0.
    blocked_until=0; halted=False; exposure=0; day=None; day_start=1.; day_block=False
    for j in range(1,len(d)):
        date=int(ts[j]//86400000)
        if date!=day:
            day=date; day_start=equity[-1]; day_block=False
        # Decisions/gates use only previous marks. New trades execute at next open.
        if not units and not halted and not day_block and j>=blocked_until and bool(signal[j-1]):
            frac=stop_fraction(atr[j-1],risk)
            entry=float(o[j]*(1+risk.slippage))
            stop=entry*(1-frac); target=entry*(1+risk.reward_risk*frac)
            per_unit_loss=entry*(1+risk.fee)-stop*(1-risk.slippage)*(1-risk.fee)
            units=min(cash*risk.max_allocation/(entry*(1+risk.fee)),cash*risk.risk_per_trade/per_unit_loss)
            spend=units*entry*(1+risk.fee); cash-=spend
            position=dict(entry_index=j,entry_timestamp=int(ts[j]),entry=entry,stop=stop,target=target,spend=spend,entry_equity=equity[-1],quantity=units,planned_stop_risk=units*per_unit_loss)
        if units:
            exposure+=1
            # Conservative intrabar risk mark, capped by stop unless opening below it.
            adverse=float(o[j] if o[j]<=position['stop'] else max(l[j],position['stop']))
            low_equity=cash+units*adverse*(1-risk.slippage)*(1-risk.fee)
            dd=max(dd,1-low_equity/peak)
            price,reason=bar_exit(o[j],h[j],l[j],c[j],position['stop'],position['target'],j-position['entry_index']+1,risk)
            if halted: price,reason=float(o[j]),'drawdown_halt_next_open'
            if j==len(d)-1 and price is None: price,reason=float(c[j]),'end_of_period'
            if price is not None:
                proceeds=units*price*(1-risk.slippage)*(1-risk.fee)
                pnl=proceeds-position['spend']; cash+=proceeds
                trades.append(dict(**position,exit_index=j,exit_timestamp=int(ts[j]),exit=float(price*(1-risk.slippage)),reason=reason,pnl=float(pnl),return_value=float(pnl/position['spend'])))
                trades[-1]['return']=trades[-1].pop('return_value')
                units=0.; position=None; blocked_until=j+risk.cooldown+1
        mark=cash+units*c[j]*(1-risk.slippage)*(1-risk.fee)
        peak=max(peak,mark); dd=max(dd,1-mark/peak); equity.append(float(mark))
        # Intrabar breach blocks further trades, any open position closes next open.
        if dd>=risk.max_drawdown: halted=True
        if mark<=day_start*(1-risk.daily_loss): day_block=True
    result=metrics(trades,equity,dd,exposure,halted)
    if details: result.update(trade_log=trades,equity_curve=equity)
    return result


def train_columns(f):
    # No labels or future rows used. Remove redundant/constant indicators within training.
    usable=[x for x in f if f[x].notna().sum()>=50 and f[x].std()>1e-12]
    corr=f[usable].corr(method='spearman').abs()
    kept=[]
    for col in usable:
        if all(not np.isfinite(corr.loc[col,k]) or corr.loc[col,k]<.95 for k in kept): kept.append(col)
    return kept


def make_model(kind):
    if kind=='boosting':
        return HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=42)
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.1,max_iter=500,random_state=42))


def fit_bundle(f,y,ends,stop,columns,risk):
    # Chronological internal calibration, with purges at both boundaries.
    cal_start=int(stop*.75); train_end=cal_start-risk.max_hold-1; cal_end=stop-risk.max_hold-1
    train_idx=np.arange(max(0,train_end)); train_idx=train_idx[(ends[train_idx]>=0)&(ends[train_idx]<cal_start)]
    cal_idx=np.arange(cal_start,max(cal_start,cal_end)); cal_idx=cal_idx[(ends[cal_idx]>=0)&(ends[cal_idx]<stop)]
    train_idx=train_idx[np.isfinite(y[train_idx])]; cal_idx=cal_idx[np.isfinite(y[cal_idx])]
    if len(train_idx)<300 or len(cal_idx)<80 or len(np.unique(y[train_idx]))<2 or len(np.unique(y[cal_idx]))<2: return None
    cols=train_columns(f.iloc[train_idx][columns])
    if not cols: return None
    models=[]
    for kind in ('boosting','logistic'):
        m=make_model(kind); m.fit(f.iloc[train_idx][cols],y[train_idx])
        p=m.predict_proba(f.iloc[cal_idx][cols])[:,1]
        # Sigmoid fitted only on an independent calibration time block.
        z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
        calibration=LogisticRegression(C=1.,random_state=42).fit(z,y[cal_idx])
        models.append((m,calibration))
    return dict(columns=cols,models=models,train_rows=len(train_idx),calibration_rows=len(cal_idx),train_last=int(train_idx[-1]),calibration_first=int(cal_idx[0]),calibration_last=int(cal_idx[-1]),evaluation_first=stop)


def predict(bundle,f,mode='ensemble'):
    probs=[]
    for m,cal in bundle['models']:
        p=m.predict_proba(f[bundle['columns']])[:,1]
        z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
        probs.append(cal.predict_proba(z)[:,1])
    return probs[0] if mode=='boosting' else np.mean(probs,axis=0)


def passes(m,min_trades=5):
    return bool(m['trades']>=min_trades and m['net_return']>0 and m['profit_factor'] is not None and m['profit_factor']>=1.5 and m['max_drawdown']<=.10 and m['payoff_ratio'] is not None and m['payoff_ratio']>=1.5 and not m['drawdown_halt'])


def block_bootstrap(trades,repeats=500):
    # Contiguous trade blocks partially preserve serial dependence. Not a guarantee.
    p=np.array([t['pnl']/t['entry_equity'] for t in trades],dtype=float)
    if len(p)<10: return dict(available=False,reason='fewer than 10 trades')
    rng=np.random.default_rng(42); block=max(2,int(np.sqrt(len(p)))); samples=[]
    for _ in range(repeats):
        indices=[]
        while len(indices)<len(p):
            start=int(rng.integers(len(p))); indices.extend((start+np.arange(block))%len(p))
        samples.append(float(np.prod(1+p[np.array(indices[:len(p)])])-1))
    return dict(available=True,replicates=repeats,block_length=block,return_p05=float(np.quantile(samples,.05)),return_p50=float(np.quantile(samples,.5)),return_p95=float(np.quantile(samples,.95)),warning='Conditional on observed trades; excludes unseen regimes and model-selection uncertainty.')


def run(csv,out,fee=.001,slip=.0005):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    for old in ('model.pkl','report.json','trades.csv','equity.csv','predictions.csv'):
        (out/old).unlink(missing_ok=True)
    risk=Risk(fee=fee,slippage=slip)
    raw=validate(pd.read_csv(csv)); f=enriched(raw)
    d=raw.iloc[100:].reset_index(drop=True); f=f.iloc[100:].reset_index(drop=True)
    if len(d)<2500: raise ValueError('Need at least 2600 contiguous closed candles')
    y,ends=labels(d,f,risk); cut=int(len(d)*.8)
    base=['return','body','range','upper_wick','lower_wick','volume_ratio','volume_change']
    groups=dict(price_volume=base,trend=base+['ema_10','ema_20','ema_50','macd','momentum_12','momentum_48'],momentum=base+['rsi_7','rsi_14','rsi_28','bollinger','momentum_3'],all=list(f.columns))
    edges=np.linspace(int(cut*.45),cut,4,dtype=int)
    results={}; folds_audit=[]
    for fold,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        for group,cols in groups.items():
            bundle=fit_bundle(f,y,ends,int(a),cols,risk)
            if bundle is None: continue
            folds_audit.append(dict(fold=fold,group=group,**{k:v for k,v in bundle.items() if k not in ('models',)}))
            valid_y=np.isfinite(y[a:b]) & (ends[a:b]<b)
            for mode in ('boosting','ensemble'):
                p=predict(bundle,f.iloc[a:b],mode)
                calibration_error=float(brier_score_loss(y[a:b][valid_y],p[valid_y])) if valid_y.any() else None
                for threshold in (.55,.65,.75):
                    key=f'{group}/{mode}/{threshold}'
                    sig=p>=threshold
                    m=simulate(d.iloc[a:b],f.iloc[a:b],sig,risk)
                    results.setdefault(key,dict(group=group,mode=mode,threshold=threshold,folds=[]))['folds'].append(dict(fold=fold,metrics=m,brier_score=calibration_error))
        print(f'Walk-forward fold {fold+1}/3 completed',flush=True)
    ranking=[]
    for key,item in results.items():
        fs=[x['metrics'] for x in item['folds']]
        accepted=len(fs)==3 and all(passes(m) for m in fs)
        score=float(np.median([m['net_return']-2*m['max_drawdown'] for m in fs]))
        ranking.append(dict(name=key,**item,score=score,accepted=accepted,reason='passed all three forward validation windows' if accepted else 'requires >=5 trades, positive return, PF>=1.5, payoff>=1.5, drawdown<=10% and no halt in every fold'))
    ranking.sort(key=lambda x:x['score'],reverse=True)
    winner=next((x for x in ranking if x['accepted']),None)
    final_d=d.iloc[cut:]; final_f=f.iloc[cut:]
    flat=simulate(final_d,final_f,np.zeros(len(final_d),bool),risk)
    report=dict(version='0.3',data_rows=len(d),risk=asdict(risk),data_source='unknown',synthetic=None,final_start_timestamp=int(d.timestamp.iloc[cut]),final_end_timestamp=int(d.timestamp.iloc[-1]),final_start_index=cut,candidate_count=len(ranking),model_family=winner['name'] if winner else 'CASH',selected_rule=None,rankings=ranking,folds_audit=folds_audit,model_final=flat,model_approved=False,status='REJECTED_KEEP_CASH',notes=['No guarantee of profit or maximum loss.','OHLC ambiguity resolves stop before target; gaps can exceed risk budget.','Three walk-forward windows select candidates; final 20% never selects or tunes candidates.','Reusing the same final time range in subsequent design changes invalidates independence.','Bootstrap is conditional, not a formal correction for multiple testing.','No exchange precision, depth or minimum-order modeling.','All thresholds and risk budgets are research defaults.'])
    # Baselines use same full period. Buy-and-hold has a different risk budget (100%).
    first=float(final_d.open.iloc[1]); last=float(final_d.close.iloc[-1])
    report['buy_hold_final']=dict(net_return=last*(1-slip)*(1-fee)/(first*(1+slip)*(1+fee))-1,allocation=1.)
    report['cash_final']=flat
    if winner:
        bundle=fit_bundle(f,y,ends,cut,groups[winner['group']],risk)
        if bundle is not None:
            p=predict(bundle,final_f,winner['mode']); sig=p>=winner['threshold']
            final=simulate(final_d,final_f,sig,risk,details=True)
            trades=final.pop('trade_log'); equity=final.pop('equity_curve')
            stress_risk=Risk(fee=fee*2,slippage=slip*2)
            stressed=simulate(final_d,final_f,sig,stress_risk)
            boot=block_bootstrap(trades)
            report.update(model_final=final,cost_stress_final=stressed,bootstrap=boot,selected_columns=bundle['columns'])
            approved=passes(final,20) and stressed['net_return']>0 and boot['available'] and boot['return_p05']>0
            report['model_approved']=bool(approved)
            report['status']='PAPER_CANDIDATE_ONLY' if approved else 'REJECTED_KEEP_CASH'
            pd.DataFrame(trades).to_csv(out/'trades.csv',index=False)
            pd.DataFrame(dict(timestamp=final_d.timestamp.iloc[1:].to_numpy(),equity=equity[1:])).to_csv(out/'equity.csv',index=False)
            pd.DataFrame(dict(timestamp=final_d.timestamp,probability_profit=p,entry_signal=sig)).to_csv(out/'predictions.csv',index=False)
            # Selected pre-test research model retained even if rejected. Explicit gate travels with it.
            with (out/'model.pkl').open('wb') as handle:
                pickle.dump(dict(bundle=bundle,mode=winner['mode'],threshold=winner['threshold'],risk=asdict(risk),paper_candidate_only=approved,live_enabled=False,feature_version='0.3'),handle)
    # If no candidate passes, do not leave an old model in the output directory.
    if not (out/'predictions.csv').exists():
        (out/'model.pkl').unlink(missing_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(status=report['status'],selected=report['model_family'],candidate_count=len(ranking))),flush=True)
    return report
