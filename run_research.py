"""Manual GitHub Actions entry point. No account secrets or order access."""
import hashlib
import json
import math
import os
import platform
import re
import sys
import traceback
import shutil
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
import numpy
import pandas
import sklearn
import engine
import research_v3


def config(env):
    source=env.get('RESEARCH_SOURCE','synthetic')
    symbol=env.get('RESEARCH_SYMBOL','BTCUSDT').strip().upper()
    interval=env.get('RESEARCH_INTERVAL','15')
    days=int(env.get('RESEARCH_DAYS','180'))
    fee=float(env.get('RESEARCH_FEE','.001'))
    slip=float(env.get('RESEARCH_SLIPPAGE','.0005'))
    if source not in ('synthetic','bybit','csv'): raise ValueError('Unknown data source')
    if not re.fullmatch(r'[A-Z0-9]{5,24}',symbol): raise ValueError('Invalid symbol')
    if interval not in ('5','15','30','60','240'): raise ValueError('Invalid interval')
    if not 30<=days<=730: raise ValueError('Days must be 30..730')
    if not all(math.isfinite(x) and 0<=x<=.02 for x in (fee,slip)): raise ValueError('Costs must be finite fractions from 0 to .02')
    data_file=env.get('RESEARCH_DATA_FILE','data/candles.csv')
    if source=='csv':
        path=Path(data_file)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='data' or path.suffix!='.csv':
            raise ValueError('CSV must be a relative data/*.csv path')
    return dict(source=source,symbol=symbol,interval=interval,days=days,fee=fee,slippage=slip,data_file=data_file)


def main():
    out=Path('outputs'); out.mkdir(exist_ok=True)
    try:
        for old in ('error.txt','run.json','SUMMARY.md','report.json','model.pkl','trades.csv','equity.csv','predictions.csv','candles.csv'):
            (out/old).unlink(missing_ok=True)
        c=config(os.environ)
        metadata=dict(status='running',config=c,started_utc=datetime.now(timezone.utc).isoformat(),commit=os.environ.get('GITHUB_SHA'),run_id=os.environ.get('GITHUB_RUN_ID'),python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__,sklearn=sklearn.__version__)
        (out/'run.json').write_text(json.dumps(metadata,indent=2))
        csv=out/'candles.csv'
        if c['source']=='synthetic': engine.demo(csv)
        elif c['source']=='csv':
            source=Path(c['data_file']).resolve()
            if not source.is_relative_to(Path('data').resolve()): raise ValueError('CSV must stay inside data/')
            imported=engine.validate(pandas.read_csv(source))
            step=int(c['interval'])*60000
            if len(imported)>1 and not (imported.timestamp.diff().dropna()==step).all(): raise ValueError('CSV interval does not match configured interval')
            if len(imported) and imported.timestamp.iloc[-1]+step>int(time.time()*1000): raise ValueError('CSV contains unclosed or future candles')
            imported.to_csv(csv,index=False)
        else:
            try: engine.fetch(c['symbol'],c['interval'],c['days'],csv)
            except urllib.error.HTTPError as exc:
                if exc.code==403: raise RuntimeError('Bybit returned HTTP 403 to this runner. Use source=csv with lawfully obtained closed spot candles in data/candles.csv. No synthetic fallback was used.') from exc
                raise
        metadata['dataset_sha256']=hashlib.sha256(csv.read_bytes()).hexdigest()
        research_v3.run(csv,out,c['fee'],c['slippage'])
        report=json.loads((out/'report.json').read_text())
        report['data_source']=c['source']; report['synthetic']=c['source']=='synthetic'
        report['symbol']=c['symbol'] if c['source']!='synthetic' else None
        report['interval_minutes']=int(c['interval']) if c['source']!='synthetic' else 15
        if c['source']=='synthetic':
            report['model_approved']=False
            report['status']='SYNTHETIC_TEST_ONLY'
            (out/'model.pkl').unlink(missing_ok=True)
        report['source_verified']=c['source']=='bybit'
        if c['source']=='csv': report['notes'].append('CSV symbol and venue are user-declared, not independently verified.')
        report['approval_scope']='Research filter only. No authorization for live trading.'
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        header='SYNTHETIC DATA — SOFTWARE TEST ONLY' if report['synthetic'] else 'Historical spot research — not a guarantee of profit'
        summary=f"# {header}\n\nDecision: `{report['status']}`\n\nSelected feature group: `{report['model_family']}`\n\nFinal model metrics:\n```json\n{json.dumps(report['model_final'],indent=2)}\n```\n\nSee report.json for folds, rejections, baseline and limitations.\n"
        (out/'SUMMARY.md').write_text(summary)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f: f.write(summary)
        metadata['status']='succeeded'; metadata['finished_utc']=datetime.now(timezone.utc).isoformat()
        (out/'run.json').write_text(json.dumps(metadata,indent=2))
    except Exception as exc:
        (out/'error.txt').write_text(traceback.format_exc())
        manifest=out/'run.json'
        m=json.loads(manifest.read_text()) if manifest.exists() else {}
        m.update(status='failed',error=str(exc))
        manifest.write_text(json.dumps(m,indent=2))
        raise

if __name__=='__main__': main()
