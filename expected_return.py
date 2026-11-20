"""Experimental expected-net-return models. Never sends exchange orders."""
from dataclasses import asdict,dataclass
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_pinball_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from diagnose_icnt import event_outcomes
from research_v3 import Risk,enriched,train_columns,simulate,passes,block_bootstrap
from engine import validate

FEATURES=['return','body','range','volume_ratio','ema_20','ema_50','rsi_14','macd','atr','momentum_12','volatility_12']

@dataclass(frozen=True)
class EntryPolicy:
    min_expected_net: float=.002
    max_q10_loss: float=.03
    min_mean_to_q10_loss: float=.10


def entry_signal(mean,q10,policy=EntryPolicy()):
    mean=np.asarray(mean);q10=np.asarray(q10)
    loss=np.maximum(-q10,.001)
    return np.isfinite(mean)&np.isfinite(q10)&(mean>=policy.min_expected_net)&(q10>=-policy.max_q10_loss)&(mean/loss>=policy.min_mean_to_q10_loss)


def temporal_indices(ends,stop,horizon):
    calibration_start=int(stop*.75)
    train=np.arange(max(0,calibration_start-horizon-1))
    calibration=np.arange(calibration_start,max(calibration_start,stop-horizon-1))
    train=train[(ends[train]>=0)&(ends[train]<calibration_start)]
    calibration=calibration[(ends[calibration]>=0)&(ends[calibration]<stop)]
    if len(train)<300 or len(calibration)<80:raise ValueError('Insufficient purged training/calibration data')
    return train,calibration


def build_bundle(f,target,ends,stop,risk):
    train,calibration=temporal_indices(ends,stop,risk.max_hold)
    if not np.isfinite(target[train]).all() or not np.isfinite(target[calibration]).all():raise ValueError('Invalid targets')
    columns=train_columns(f.iloc[train][FEATURES])
    if not columns:raise ValueError('No usable features')
    x=f.iloc[train][columns];xc=f.iloc[calibration][columns]
    shared=dict(max_iter=60,max_leaf_nodes=3,max_depth=2,min_samples_leaf=128,l2_regularization=50.,learning_rate=.05,early_stopping=False,random_state=42)
    models=dict(ridge=make_pipeline(SimpleImputer(strategy='median'),RobustScaler(),Ridge(alpha=1000.)),small_boosting=HistGradientBoostingRegressor(loss='squared_error',**shared))
    shifts={}
    for name,m in models.items():
        m.fit(x,target[train]);shifts[name]=float(np.mean(target[calibration]-m.predict(xc)))
    # Q10 means a lower conditional outcome quantile, not a guaranteed loss bound.
    lower=HistGradientBoostingRegressor(loss='quantile',quantile=.1,**shared).fit(x,target[train])
    lower_shift=float(np.quantile(target[calibration]-lower.predict(xc),.1))
    dummy=DummyRegressor(strategy='mean').fit(xc,target[calibration])
    return dict(columns=columns,models=models,mean_bias_shifts=shifts,lower_model=lower,lower_shift=lower_shift,baseline=dummy,baseline_q10=float(np.quantile(target[calibration],.1)),training_mean=float(target[train].mean()),audit=dict(train_last=int(train[-1]),train_label_last=int(ends[train].max()),calibration_first=int(calibration[0]),calibration_last=int(calibration[-1]),calibration_label_last=int(ends[calibration].max()),evaluation_first=int(stop),train_rows=len(train),calibration_rows=len(calibration)))


def predict_bundle(bundle,f):
    x=f[bundle['columns']]
    means={name:m.predict(x)+bundle['mean_bias_shifts'][name] for name,m in bundle['models'].items()}
    means['constant_recent_mean']=bundle['baseline'].predict(x)
    lower=bundle['lower_model'].predict(x)+bundle['lower_shift']
    # Keep the lower quantile at or below each model's mean as a conservative guard.
    return {name:dict(mean=mu,q10=np.minimum(lower,mu) if name!='constant_recent_mean' else np.minimum(np.full(len(x),bundle['baseline_q10']),mu)) for name,mu in means.items()}


def predictive_metrics(actual,mean,q10,baseline_mean,baseline_q10):
    mse=float(mean_squared_error(actual,mean));base=float(mean_squared_error(actual,baseline_mean))
    pinball=float(mean_pinball_loss(actual,q10,alpha=.1))
    base_pinball=float(mean_pinball_loss(actual,baseline_q10,alpha=.1))
    return dict(mse=mse,baseline_mse=base,mse_skill=1-mse/base if base else None,mae=float(mean_absolute_error(actual,mean)),prediction_bias=float(np.mean(mean-actual)),actual_mean=float(np.mean(actual)),predicted_mean=float(np.mean(mean)),q10_pinball=pinball,q10_baseline_pinball=base_pinball,q10_pinball_skill=1-pinball/base_pinball if base_pinball else None,q10_violation_rate=float(np.mean(actual<q10)),q10_mean=float(np.mean(q10)))


def gated_signal(prediction,length,horizon,policy):
    signal=entry_signal(prediction['mean'],prediction['q10'],policy)
    # Do not initiate trades whose maximum duration exceeds a fixed evaluation boundary.
    signal[np.arange(length)>=length-horizon]=False
    return signal


def compare(csv,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for name in ('research_models.pkl','report.json','comparison.csv','validation_predictions.csv','final_trades.csv','final_equity.csv'):
        (out/name).unlink(missing_ok=True)
    raw=validate(pd.read_csv(csv));n=len(raw)-100;cut=int(n*.8)
    if n<5000:raise ValueError('Need at least 5100 candles')
    risk=Risk();policy=EntryPolicy()
    # The final 20% is sliced off BEFORE feature and target generation.
    dev=raw.iloc[:cut+100].copy();f=enriched(dev).iloc[100:].reset_index(drop=True);d=dev.iloc[100:].reset_index(drop=True)
    outcomes=event_outcomes(d,f,risk);target=outcomes['net'];ends=outcomes['ends']
    edges=np.linspace(int(cut*.45),cut,4,dtype=int)
    rows=[];audits=[];predictions=[]
    for fold,(a,b) in enumerate(zip(edges[:-1],edges[1:]),1):
        bundle=build_bundle(f,target,ends,int(a),risk)
        audits.append(dict(fold=fold,columns=bundle['columns'],**bundle['audit']))
        pred=predict_bundle(bundle,f.iloc[a:b])
        valid=np.isfinite(target[a:b])&(ends[a:b]>=0)&(ends[a:b]<b)
        for name,p in pred.items():
            baseline=pred['constant_recent_mean']
            predictive=predictive_metrics(target[a:b][valid],p['mean'][valid],p['q10'][valid],baseline['mean'][valid],baseline['q10'][valid])
            signal=gated_signal(p,len(p['mean']),risk.max_hold,policy)
            trading=simulate(d.iloc[a:b],f.iloc[a:b],signal,risk)
            stress=simulate(d.iloc[a:b],f.iloc[a:b],signal,Risk(fee=risk.fee*2,slippage=risk.slippage*2))
            rows.append(dict(fold=fold,model=name,predictive=predictive,trading=trading,cost_stress=stress,entry_signals=int(signal.sum())))
            predictions.append(pd.DataFrame(dict(timestamp=d.timestamp.iloc[a:b].to_numpy(),fold=fold,model=name,expected_net_return=p['mean'],predicted_q10_net_return=p['q10'],entry_signal=signal)))
        print(f'Expected-return fold {fold}/3 completed',flush=True)
    summaries=[]
    for name in ('ridge','small_boosting','constant_recent_mean'):
        selected=[r for r in rows if r['model']==name]
        has_skill=all(r['predictive']['mse_skill'] is not None and r['predictive']['mse_skill']>0 for r in selected)
        loss_skill=all(r['predictive']['q10_pinball_skill'] is not None and r['predictive']['q10_pinball_skill']>=0 for r in selected)
        coverage_stable=all(.05<=r['predictive']['q10_violation_rate']<=.15 for r in selected)
        trade_pass=all(passes(r['trading']) and r['cost_stress']['net_return']>0 for r in selected)
        approved=name!='constant_recent_mean' and has_skill and loss_skill and coverage_stable and trade_pass
        summaries.append(dict(model=name,mean_mse_skill=float(np.mean([r['predictive']['mse_skill'] for r in selected])),mean_q10_skill=float(np.mean([r['predictive']['q10_pinball_skill'] for r in selected])),q10_coverage_stable=coverage_stable,q10_violation_rates=[r['predictive']['q10_violation_rate'] for r in selected],validation_trades=sum(r['trading']['trades'] for r in selected),validation_returns=[r['trading']['net_return'] for r in selected],qualified=approved,score=float(np.median([r['trading']['net_return']-2*r['trading']['max_drawdown'] for r in selected]))))
    candidates=sorted([s for s in summaries if s['qualified']],key=lambda s:s['score'],reverse=True)
    chosen=candidates[0]['model'] if candidates else None
    research=build_bundle(f,target,ends,cut,risk)
    report=dict(version='expected-return-1',symbol='ICNTUSDT',rows=len(raw),development_rows=cut,final_rows_reserved=n-cut,risk=asdict(risk),entry_policy=asdict(policy),features=research['columns'],summary=summaries,folds=rows,audit=audits,selected=chosen,status='REJECTED_KEEP_CASH',live_enabled=False,notes=['The model estimates per-position net return after assumed fees and slippage, not portfolio return.','Q10 is a learned lower outcome quantile, not a guaranteed maximum loss.','Mean predictions are bias-adjusted on a separate purged chronological calibration block.','Q10 residual shift is estimated on that calibration block; temporal nonstationarity can invalidate coverage.','Final 20% is excluded from feature/target construction during model selection.','Two mean models, one fixed policy and original risk settings were declared before this experiment.','Research artifact is saved even when rejected; never interpret file existence as approval.','Development data have been reused for earlier diagnostics and searches; future paper testing is required.'])
    if chosen:
        full_f=enriched(raw).iloc[100:].reset_index(drop=True)
        test=raw.iloc[cut+100:].reset_index(drop=True);test_f=full_f.iloc[cut:].reset_index(drop=True)
        p=predict_bundle(research,test_f)[chosen]
        signal=gated_signal(p,len(test),risk.max_hold,policy)
        final=simulate(test,test_f,signal,risk,details=True)
        trades=final.pop('trade_log');curve=final.pop('equity_curve')
        stress=simulate(test,test_f,signal,Risk(fee=risk.fee*2,slippage=risk.slippage*2));boot=block_bootstrap(trades)
        passed=passes(final,20) and stress['net_return']>0 and boot['available'] and boot['return_p05']>0
        report.update(status='PAPER_CANDIDATE_ONLY' if passed else 'FAILED_FINAL_TEST',final=final,cost_stress=stress,bootstrap=boot)
        pd.DataFrame(trades).to_csv(out/'final_trades.csv',index=False)
        pd.DataFrame(dict(timestamp=test.timestamp.iloc[1:].to_numpy(),equity=curve[1:])).to_csv(out/'final_equity.csv',index=False)
    with (out/'research_models.pkl').open('wb') as handle:
        pickle.dump(dict(bundle=research,selected=chosen,policy=asdict(policy),risk=asdict(risk),status=report['status'],live_enabled=False,feature_version='expected-return-1'),handle)
    pd.concat(predictions,ignore_index=True).to_csv(out/'validation_predictions.csv',index=False)
    comparison=[]
    for r in rows:comparison.append(dict(fold=r['fold'],model=r['model'],**r['predictive'],**{'trading_'+k:v for k,v in r['trading'].items()}))
    pd.DataFrame(comparison).to_csv(out/'comparison.csv',index=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(status=report['status'],selected=chosen,summary=summaries),indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default='data/ICNTUSDT_spot_15m.csv');p.add_argument('--output',default='expected_return_outputs');a=p.parse_args();compare(a.csv,a.output)
