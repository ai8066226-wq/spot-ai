"""Development-only diagnostics. Does not tune settings or evaluate final 20%."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score,average_precision_score,brier_score_loss,log_loss)
from research_v3 import Risk,enriched,bar_exit,stop_fraction,fit_bundle,simulate
from engine import validate


def event_outcomes(d,f,risk):
    """Hypothetical overlapping entries, not a tradable portfolio ledger."""
    n=len(d); y=np.full(n,np.nan); ends=np.full(n,-1,dtype=int)
    gross=np.full(n,np.nan); net=np.full(n,np.nan); reasons=np.full(n,'',dtype=object)
    o=d.open.to_numpy();h=d.high.to_numpy();l=d.low.to_numpy();c=d.close.to_numpy();atr=f.atr.to_numpy()
    for t in range(n-risk.max_hold):
        j=t+1;entry=o[j]*(1+risk.slippage);fraction=stop_fraction(atr[t],risk)
        stop=entry*(1-fraction);target=entry*(1+risk.reward_risk*fraction)
        for k in range(j,j+risk.max_hold):
            exit_price,reason=bar_exit(o[k],h[k],l[k],c[k],stop,target,k-j+1,risk)
            if exit_price is not None:
                gross[t]=exit_price/o[j]-1
                net[t]=exit_price*(1-risk.slippage)*(1-risk.fee)/(entry*(1+risk.fee))-1
                y[t]=int(net[t]>0);ends[t]=k;reasons[t]=reason;break
    return dict(y=y,ends=ends,gross=gross,net=net,reasons=reasons)


def probability_metrics(y,p,reference):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    baseline=np.full(len(y),reference)
    brier=float(brier_score_loss(y,p));base_brier=float(brier_score_loss(y,baseline))
    top=np.argsort(-p,kind='stable')[:max(1,int(np.ceil(.1*len(p))))]
    prevalence=float(y.mean())
    return dict(rows=len(y),positive_rate=prevalence,roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,average_precision=float(average_precision_score(y,p)) if y.any() else None,brier=brier,baseline_brier=base_brier,brier_skill=1-brier/base_brier if base_brier else None,log_loss=float(log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1])),probability_mean=float(p.mean()),probability_p05=float(np.quantile(p,.05)),probability_p50=float(np.quantile(p,.5)),probability_p95=float(np.quantile(p,.95)),probability_max=float(p.max()),top_decile_positive_rate=float(y[top].mean()),top_decile_lift=float(y[top].mean()/prevalence) if prevalence else None)


def reliability(y,p):
    bins=[]
    for lower in np.arange(0,1,.1):
        mask=(p>=lower)&(p<(lower+.1) if lower<.9 else p<=1)
        if mask.any():bins.append(dict(lower=float(lower),upper=float(min(1,lower+.1)),count=int(mask.sum()),predicted_mean=float(p[mask].mean()),observed_positive_rate=float(y[mask].mean())))
    return bins


def auc_block_interval(y,p,block,repeats=200):
    rng=np.random.default_rng(42);n=len(y);scores=[]
    for _ in range(repeats):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        ix=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        if len(np.unique(y[ix]))==2:scores.append(roc_auc_score(y[ix],p[ix]))
    return dict(block_length=block,replicates=len(scores),p05=float(np.quantile(scores,.05)) if scores else None,p95=float(np.quantile(scores,.95)) if scores else None,scope='Conditional moving-block resampling of this validation window; not multiple-testing adjusted.')


def fixed_trade_costs(trades,risk):
    # Keep actual quantities and entry/exit events fixed. Not a reinvested counterfactual.
    actual=sum(t['pnl'] for t in trades)
    fee_only=0.;cost_free=0.
    for t in trades:
        q=t['quantity'];buy=t['entry'];sell=t['exit']
        fee_only+=q*(sell-buy)
        cost_free+=q*(sell/(1-risk.slippage)-buy/(1+risk.slippage))
    return dict(trades=len(trades),actual_pnl=float(actual),same_trades_without_fees_pnl=float(fee_only),same_trades_without_fees_or_slippage_pnl=float(cost_free),fee_drag=float(fee_only-actual),slippage_drag=float(cost_free-fee_only),scope='Fixed quantities and events, amounts in initial-wealth units. Not an alternative executable portfolio.')


def breakeven(returns):
    wins=returns[returns>0];losses=-returns[returns<0]
    if not len(wins) or not len(losses):return None
    return float(losses.mean()/(wins.mean()+losses.mean()))


def diagnostic(csv,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=validate(pd.read_csv(csv));n=len(raw)-100;cut=int(n*.8)
    if n<5000:raise ValueError('Need 5100 candles')
    # No test prices used to form features or labels.
    dev=raw.iloc[:cut+100].copy();f=enriched(dev).iloc[100:].reset_index(drop=True);d=dev.iloc[100:].reset_index(drop=True)
    profiles=dict(original=Risk(),shorter_target=Risk(max_hold=32,stop_atr=1.5,reward_risk=1.5),longer_horizon=Risk(max_hold=64,stop_atr=1.,reward_risk=2.5))
    groups=dict(trend=['return','body','range','volume_ratio','volume_change','ema_10','ema_20','ema_50','macd','momentum_12','momentum_48'],all=list(f.columns))
    edges=np.linspace(int(cut*.45),cut,4,dtype=int)
    records=[];targets=[];audit=[];importance=[];trading=[]
    for profile,risk in profiles.items():
        outcomes=event_outcomes(d,f,risk);y=outcomes['y'];ends=outcomes['ends']
        for fold,(a,b) in enumerate(zip(edges[:-1],edges[1:]),1):
            valid=np.isfinite(y[a:b])&(ends[a:b]>=0)&(ends[a:b]<b)
            eval_y=y[a:b][valid].astype(int)
            gross=outcomes['gross'][a:b][valid];net=outcomes['net'][a:b][valid];why=outcomes['reasons'][a:b][valid]
            targets.append(dict(profile=profile,fold=fold,rows=len(eval_y),net_positive_rate=float(eval_y.mean()),gross_positive_rate=float((gross>0).mean()),cost_flipped_fraction=float(((gross>0)&(net<=0)).mean()),gross_return_mean=float(gross.mean()),net_return_mean=float(net.mean()),timeout_fraction=float((why=='timeout').mean()),exit_counts={str(k):int(v) for k,v in zip(*np.unique(why,return_counts=True))},note='Overlapping hypothetical entries, not independent trades or portfolio returns.'))
            for group,cols in groups.items():
                bundle=fit_bundle(f,y,ends,int(a),cols,risk)
                if bundle is None:raise ValueError('Not enough train/calibration classes')
                train_ix=np.arange(bundle['train_last']+1);cal_ix=np.arange(bundle['calibration_first'],bundle['calibration_last']+1)
                reference=float(y[cal_ix].mean());training_reference=float(y[train_ix].mean())
                be=breakeven(outcomes['net'][cal_ix])
                audit.append(dict(profile=profile,fold=fold,group=group,train_last=int(train_ix[-1]),calibration_first=int(cal_ix[0]),calibration_last=int(cal_ix[-1]),evaluation_first=int(a),evaluation_end_exclusive=int(b),max_train_label_end=int(ends[train_ix].max()),max_calibration_label_end=int(ends[cal_ix].max()),train_positive_rate=training_reference,calibration_positive_rate=reference,calibration_payoff_breakeven_probability=be,selected_columns=bundle['columns']))
                fv=f.iloc[a:b];predictions={}
                for kind,(model,cal) in zip(('boosting','logistic'),bundle['models']):
                    p=model.predict_proba(fv[bundle['columns']])[:,1]
                    z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
                    calibrated=cal.predict_proba(z)[:,1]
                    predictions[kind+'_raw']=p;predictions[kind+'_calibrated']=calibrated
                    train_p=model.predict_proba(f.iloc[train_ix][bundle['columns']])[:,1]
                    audit[-1][kind+'_training_auc']=float(roc_auc_score(y[train_ix],train_p))
                    audit[-1][kind+'_calibration_slope']=float(cal.coef_[0,0])
                predictions['ensemble_calibrated']=(predictions['boosting_calibrated']+predictions['logistic_calibrated'])/2
                for name,p in predictions.items():
                    metrics=probability_metrics(eval_y,p[valid],reference)
                    row=dict(profile=profile,group=group,fold=fold,model=name,**metrics,reliability=reliability(eval_y,p[valid]))
                    if name=='boosting_calibrated':row['auc_resampling']=auc_block_interval(eval_y,p[valid],max(64,risk.max_hold*2))
                    records.append(row)
                    # Diagnostic accounting at fixed, already explored thresholds; no selection here.
                    for threshold in (.30,.55):
                        m=simulate(d.iloc[a:b],fv,p>=threshold,risk,details=True)
                        trades=m.pop('trade_log');m.pop('equity_curve')
                        trading.append(dict(profile=profile,group=group,fold=fold,model=name,threshold=threshold,metrics=m,cost_accounting=fixed_trade_costs(trades,risk)))
                # Past-calibration-only constant prediction benchmarks, never test prevalence.
                for name,constant in (('constant_calibration_prior',reference),('constant_training_prior',training_reference)):
                    records.append(dict(profile=profile,group=group,fold=fold,model=name,**probability_metrics(eval_y,np.full(len(eval_y),constant),reference)))
                # Circularly shift whole feature families within the validation window.
                # It is a diagnostic perturbation, never used to fit or select production features.
                model=bundle['models'][0][0];base=roc_auc_score(eval_y,predictions['boosting_raw'][valid])
                families=dict(price=['return','body','range','upper_wick','lower_wick'],volume=['volume_ratio','volume_change'],trend=['ema_10','ema_20','ema_50','macd'],momentum=['rsi_7','rsi_14','rsi_28','bollinger','momentum_3','momentum_12','momentum_48'],volatility=['atr','volatility_3','volatility_12','volatility_48'])
                for family,names in families.items():
                    keep=[x for x in names if x in bundle['columns']]
                    if not keep:continue
                    drops=[]
                    for fraction in (.25,.5,.75):
                        changed=fv[bundle['columns']].copy();changed[keep]=np.roll(changed[keep].to_numpy(),max(risk.max_hold+1,int(len(changed)*fraction)),axis=0)
                        perturbed=model.predict_proba(changed)[:,1]
                        drops.append(float(base-roc_auc_score(eval_y,perturbed[valid])))
                    importance.append(dict(profile=profile,group=group,fold=fold,family=family,columns=keep,auc_drop_mean=float(np.mean(drops)),auc_drop_shifts=drops,note='Circular group shifts; diagnostic only, correlated families can mask importance.'))
        print(f'Diagnosed {profile}',flush=True)
    summary=[]
    for profile in profiles:
        for group in groups:
            for name in predictions.keys():
                chosen=[r for r in records if r['profile']==profile and r['group']==group and r['model']==name]
                summary.append(dict(profile=profile,group=group,model=name,mean_auc=float(np.mean([r['roc_auc'] for r in chosen])),min_auc=float(min(r['roc_auc'] for r in chosen)),mean_brier_skill=float(np.mean([r['brier_skill'] for r in chosen])),mean_top_decile_lift=float(np.mean([r['top_decile_lift'] for r in chosen])),windows_auc_above_half=sum(r['roc_auc']>.5 for r in chosen),windows_brier_skill_positive=sum(r['brier_skill']>0 for r in chosen)))
    report=dict(symbol='ICNTUSDT',version='diagnostic-1',development_rows=cut,final_rows_excluded=n-cut,development_last_timestamp=int(d.timestamp.iloc[-1]),profiles={k:asdict(v) for k,v in profiles.items()},summary=summary,targets=targets,audit=audit,classification=records,feature_perturbations=importance,trading=trading,live_enabled=False,notes=['No production model, threshold or risk setting changed.','Final 20% not evaluated.','Brier score combines calibration and discrimination, not calibration alone.','ROC-AUC confidence ranges are descriptive block-resampling, not formal multiple-test corrected significance.','Top-decile results are descriptive and not a new deployment threshold.','Target returns overlap; they must not be summed as independent trades.'])
    (out/'diagnostic_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    pd.DataFrame(summary).to_csv(out/'classification_summary.csv',index=False)
    pd.DataFrame(targets).to_csv(out/'target_diagnostics.csv',index=False)
    print(json.dumps(summary,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default='data/ICNTUSDT_spot_15m.csv');p.add_argument('--output',default='diagnostic_outputs');a=p.parse_args();diagnostic(a.csv,a.output)
