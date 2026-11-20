"""KDJ entry laboratory: fixed take profit plus causally updated trailing stop."""
from dataclasses import dataclass,asdict
import argparse,html,json
from pathlib import Path
import numpy as np
import pandas as pd
from engine import validate
from kdj_research import kdj,cross_up
from research_v3 import Risk,enriched,metrics,passes,block_bootstrap

from execution import ExitSettings, updated_stop, simulate_trailing


def setting_grid(stops=(.005,.01,.02),targets=(.005,.01,.02,.03,.05),percent=(.003,.005,.01,.02),atr=(1.,1.5,2.),activations=(0.,.005,.01)):
    count=len(stops)*len(targets)*(len(percent)+len(atr))*len(activations)
    if not 1<=count<=1000:raise ValueError('Choose between 1 and 1000 settings')
    return [ExitSettings(stop,target,mode,distance,activation)
            for stop in stops for target in targets
            for mode,distances in (('percent',percent),('atr',atr))
            for distance in distances for activation in activations]


def parse_values(text,percent=True,allow_zero=False):
    values=list(dict.fromkeys(float(x.strip()) for x in text.split(',') if x.strip()))
    if not values or len(values)>12:raise ValueError('Provide 1..12 comma-separated values')
    cap=100 if percent else 20
    if not all(np.isfinite(x) and (x>=0 if allow_zero else x>0) and x<=cap for x in values):raise ValueError('Values outside allowed range')
    return tuple(x/100 if percent else x for x in values)


def entry_signals(d,rule='kdj_cross'):
    if rule=='kdj_cross':
        indicator=kdj(d)
        return cross_up(indicator.J,indicator.K)
    if rule=='macd_above':
        dif=d.close.ewm(span=12,adjust=False).mean()-d.close.ewm(span=26,adjust=False).mean()
        dea=dif.ewm(span=9,adjust=False).mean()
        return (dif>dea).to_numpy()
    raise ValueError('Unknown entry rule')



def html_report(report,out):
    # Self-contained and usable offline on a phone; no network or third-party JS.
    payload=json.dumps(report,ensure_ascii=False).replace('<','\\u003c')
    template='''<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>مختبر KDJ</title><style>body{font:16px system-ui;background:#101827;color:#e8edf7;margin:20px}h1{font-size:25px}section{background:#1d293e;padding:16px;border-radius:12px;margin:12px 0}button,select{font:inherit;padding:8px;margin:5px}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #41516c;text-align:right}.scroll{overflow:auto}a{color:#80c8ff}.bad{color:#ffb1a9}</style><h1>مختبر الأهداف والوقف المتحرك</h1><section id="summary"></section><section>نتائج تاريخية للتطوير، لا ضمان أرباح. الدخول بعد تأكيد تقاطع J/K عند الإغلاق. يُحدّث الوقف بعد الإغلاق ويصبح ساريًا في الشمعة التالية؛ لا نفترض معرفة ترتيب الحركة داخل الشمعة.</section><section><label>نوع الوقف <select id="mode"><option value="all">الكل</option><option value="percent">نسبة مئوية</option><option value="atr">ATR</option></select></label><label><input type="checkbox" id="qualified"> المقبول فقط</label><label>الترتيب <select id="sort"><option value="mean_return">متوسط العائد الأعلى</option><option value="worst_return">أسوأ فترة أفضل</option><option value="total_trades">عدد الصفقات</option></select></label><button onclick="downloadCSV()">تنزيل جدول CSV</button><div id="count"></div><div class="scroll"><table><thead><tr><th>الترتيب</th><th>هدف الربح</th><th>وقف البداية</th><th>الوقف المتحرك</th><th>التفعيل عند ربح</th><th>متوسط العائد</th><th>أسوأ فترة</th><th>الصفقات</th><th>فترات رابحة</th><th>القرار</th></tr></thead><tbody id="rows"></tbody></table></div></section><script>const report=PAYLOAD;const pct=x=>(100*x).toFixed(2)+'%';document.getElementById('summary').textContent=report.symbol+' | '+report.candidates.length+' إعدادًا | '+report.status+' | آخر '+report.final_rows_reserved+' شمعة خارج الاختيار';function render(){let a=report.candidates.filter(x=>(mode.value==='all'||x.settings.trailing_mode===mode.value)&&(!qualified.checked||x.qualified));a.sort((x,y)=>y[sort.value]-x[sort.value]);count.textContent=a.length+' إعدادًا';rows.innerHTML=a.map((x,i)=>`<tr><td>${i+1}</td><td>${pct(x.settings.take_profit)}</td><td>${pct(x.settings.initial_stop)}</td><td>${x.settings.trailing_mode==='percent'?pct(x.settings.trailing_distance):x.settings.trailing_distance+' ATR'}</td><td>${pct(x.settings.activation_gain)}</td><td>${pct(x.mean_return)}</td><td>${pct(x.worst_return)}</td><td>${x.total_trades}</td><td>${x.positive_folds}/3</td><td>${x.qualified?'مرشح تحقق فقط':'مرفوض'}</td></tr>`).join('')}function downloadCSV(){let lines=['target,initial_stop,trailing_mode,trailing_distance,activation,mean_return,worst_return,trades,qualified'];for(let x of report.candidates){let s=x.settings;lines.push([s.take_profit,s.initial_stop,s.trailing_mode,s.trailing_distance,s.activation_gain,x.mean_return,x.worst_return,x.total_trades,x.qualified].join(','))}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')],{type:'text/csv'}));a.download='trailing_lab.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}for(let id of ['mode','qualified','sort'])document.getElementById(id).onchange=render;render();</script></html>'''
    # Explicit bindings avoid browser-dependent window named-element behavior.
    template=template.replace("const report=PAYLOAD;","const report=PAYLOAD;const mode=document.getElementById('mode'),qualified=document.getElementById('qualified'),sort=document.getElementById('sort'),count=document.getElementById('count'),rows=document.getElementById('rows');")
    description='الدخول عند DIF أكبر من DEA في إغلاق الشمعة، مع إعادة الدخول عند السماح.' if report.get('entry_rule')=='macd_above' else 'الدخول بعد تأكيد تقاطع J/K عند الإغلاق.'
    template=template.replace('الدخول بعد تأكيد تقاطع J/K عند الإغلاق.',description)
    template=template.replace('<title>مختبر KDJ</title>','<title>مختبر MACD</title>' if report.get('entry_rule')=='macd_above' else '<title>مختبر KDJ</title>')
    (out/'lab.html').write_text(template.replace('PAYLOAD',payload))


def run(csv,output,symbol='ICNTUSDT',experiments=None,entry_rule='kdj_cross'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    raw=validate(pd.read_csv(csv));n=len(raw)-100;cut=int(n*.8)
    if n<5000:raise ValueError('Need 5100 candles')
    dev=raw.iloc[:cut+100]
    d=dev.iloc[100:].reset_index(drop=True);f=enriched(dev).iloc[100:].reset_index(drop=True)
    signal=entry_signals(dev,entry_rule)[100:];edges=np.linspace(int(cut*.45),cut,4,dtype=int);risk=Risk()
    candidates=[]
    experiments=setting_grid() if experiments is None else experiments
    for i,settings in enumerate(experiments):
        folds=[simulate_trailing(d.iloc[a:b],f.iloc[a:b],signal[a:b],settings,risk) for a,b in zip(edges[:-1],edges[1:])]
        candidates.append(dict(id=i,settings=asdict(settings),folds=folds,mean_return=float(np.mean([m['net_return'] for m in folds])),worst_return=float(min(m['net_return'] for m in folds)),max_drawdown=float(max(m['max_drawdown'] for m in folds)),total_trades=sum(m['trades'] for m in folds),positive_folds=sum(m['net_return']>0 for m in folds),qualified=all(passes(m) for m in folds)))
        if (i+1)%45==0:print(f'Tested {i+1}/{len(experiments)} exit settings',flush=True)
    candidates.sort(key=lambda x:x['mean_return'],reverse=True)
    qualified=[x for x in candidates if x['qualified']];winner=qualified[0] if qualified else None
    report=dict(version='trailing-lab-1.1',symbol=symbol,entry_rule=entry_rule,parameters=[9,3,3] if entry_rule=='kdj_cross' else [12,26,9],development_rows=cut,final_rows_reserved=n-cut,risk=asdict(risk),ignored_risk_fields=['stop_atr','min_stop','max_stop','reward_risk','max_hold'],best_by_validation_profit=candidates[0],selected=winner,status='NO_QUALIFIED_SETTINGS' if not winner else 'VALIDATION_CANDIDATE_ONLY',candidates=candidates,live_enabled=False,execution='Closed-candle entry condition, next-open entry. Stops ratchet only upward at completed-bar close and become effective next bar. Existing stop wins same-bar stop/target ambiguity.',notes=['Grid fixed before this run; final 20% excluded from feature generation and parameter choice.','Highest mean validation return does not imply acceptance or future profitability.','Original acceptance gates remain unchanged: positive each window, >=5 trades/window, PF>=1.5, payoff>=1.5, no drawdown halt.','No J-K distance exit, target and trailing stop are the exits in this experiment.','No intrabar entry or continuously updated trailing execution is claimed.','Development data reused across previous research; fresh future paper testing is required.'])
    # Save trades and stop traces of the highest-profit validation setting, even if rejected.
    best=ExitSettings(**candidates[0]['settings']);logs=[];trace=[]
    for fold,(a,b) in enumerate(zip(edges[:-1],edges[1:]),1):
        result=simulate_trailing(d.iloc[a:b],f.iloc[a:b],signal[a:b],best,risk,True)
        logs.extend(dict(fold=fold,**t) for t in result['trade_log']);trace.extend(dict(fold=fold,**t) for t in result['stop_trace'])
    pd.DataFrame(logs).to_csv(out/'best_validation_trades.csv',index=False);pd.DataFrame(trace).to_csv(out/'best_stop_trace.csv',index=False)
    if winner:
        settings=ExitSettings(**winner['settings'])
        full_f=enriched(raw).iloc[100:].reset_index(drop=True)
        full_signal=entry_signals(raw,entry_rule)[100:]
        test=raw.iloc[cut+100:].reset_index(drop=True);ff=full_f.iloc[cut:].reset_index(drop=True)
        final=simulate_trailing(test,ff,full_signal[cut:],settings,risk)
        stress=simulate_trailing(test,ff,full_signal[cut:],settings,Risk(fee=risk.fee*2,slippage=risk.slippage*2))
        report.update(final=final,cost_stress=stress,status='PAPER_CANDIDATE_ONLY' if passes(final,20) and stress['net_return']>0 else 'FAILED_FINAL_TEST')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    pd.DataFrame([dict(id=x['id'],**x['settings'],mean_return=x['mean_return'],worst_return=x['worst_return'],max_drawdown=x['max_drawdown'],trades=x['total_trades'],positive_folds=x['positive_folds'],qualified=x['qualified']) for x in candidates]).to_csv(out/'comparison.csv',index=False)
    html_report(report,out)
    print(json.dumps(dict(status=report['status'],best_settings=candidates[0]['settings'],best_folds=candidates[0]['folds'],qualified=len(qualified)),indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--symbol',choices=['ICNTUSDT','BTCUSDT'],default='ICNTUSDT');p.add_argument('--entry',choices=['kdj_cross','macd_above'],default='kdj_cross');p.add_argument('--output',default='trailing_lab_outputs');p.add_argument('--targets',default='0.5,1,2,3,5');p.add_argument('--initial-stops',default='0.5,1,2');p.add_argument('--trailing-percent',default='0.3,0.5,1,2');p.add_argument('--trailing-atr',default='1,1.5,2');p.add_argument('--activations',default='0,0.5,1');a=p.parse_args()
    path='data/ICNTUSDT_spot_15m.csv' if a.symbol=='ICNTUSDT' else 'data/candles.csv'
    settings=setting_grid(stops=parse_values(a.initial_stops),targets=parse_values(a.targets),percent=parse_values(a.trailing_percent),atr=parse_values(a.trailing_atr,percent=False),activations=parse_values(a.activations,allow_zero=True))
    run(path,a.output,a.symbol,settings,a.entry)
