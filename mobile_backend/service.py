import hashlib,json,math,os,pickle,sys,threading,time,uuid
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent/'engine'))
from indicators import add_all,ml_features
from ml_engine import barrier_labels,make_model,_pick_threshold,MODEL_NAME
from backtest import run_backtest,metrics,years_of
from broker import Bybit,floor_step

INTERVALS=['1','3','5','15','30','60','120','240','360','720','D']
def config_id(c):
    return hashlib.sha256(json.dumps({k:c[k] for k in ['symbols','interval','start','end']},sort_keys=True).encode()).hexdigest()
def frame(rows,interval):
    d=pd.DataFrame(rows,columns=['timestamp','open','high','low','close','volume'])
    if len(d)<300:raise ValueError('بيانات غير كافية')
    if not np.isfinite(d.to_numpy()).all() or (d.volume<0).any() or (d[['open','high','low','close']]<=0).any().any():raise ValueError('بيانات غير صالحة')
    step=(1440 if interval=='D' else int(interval))*60000
    if d.timestamp.duplicated().any() or not (d.timestamp.diff().dropna()==step).all():raise ValueError('فجوات في التاريخ؛ لا يمكن اعتماد الاختبار')
    if ((d.high<d[['open','close','low']].max(axis=1))|(d.low>d[['open','close','high']].min(axis=1))).any():raise ValueError('نطاق شمعة غير صالح')
    d.index=pd.to_datetime(d.pop('timestamp'),unit='ms',utc=True);return d

def train_symbol(rows,c):
    raw=frame(rows,c['interval'])
    if len(raw)<2000:raise ValueError('يلزم2000 شمعة مغلقة على الأقل')
    # Frozen single configuration from prior research, no new search on final slice.
    d=add_all(raw).iloc[200:];x=ml_features(d);H=24;tp=1.5;sl=1.
    y,_=barrier_labels(d,tp,sl,H);split=int(len(d)*.7);cal=int(split*.8);valid=x.notna().all(axis=1).to_numpy()&np.isfinite(y)
    fit=np.flatnonzero(valid[:cal-H]);calidx=np.arange(cal,split-H)
    if len(fit)<500 or len(np.unique(y[fit]))<2:raise ValueError('عينة التدريب أو الفئات غير كافية')
    model=make_model().fit(x.iloc[fit],y[fit]);p=np.zeros(len(calidx));ok=valid[calidx]
    p[ok]=model.predict_proba(x.iloc[calidx[ok]])[:,1]
    threshold=_pick_threshold(p,d.iloc[cal:split-H],1,tp,sl,H)
    if threshold is None:return dict(status='REJECTED',reason='لا توجد عتبة معايرة مربحة',rows=len(raw),passed=False,win_rate=None),None
    idx=np.flatnonzero(valid[:split-H]);model=make_model().fit(x.iloc[idx],y[idx])
    test=d.iloc[split:];xt=x.iloc[split:];ok=xt.notna().all(axis=1).to_numpy();p=np.zeros(len(test));p[ok]=model.predict_proba(xt.iloc[np.flatnonzero(ok)])[:,1]
    trades=run_backtest(test,(p>threshold).astype(int),tp=tp,sl=sl,max_bars=H)
    m=metrics(trades,years=years_of(test));stress=metrics(run_backtest(test,(p>threshold).astype(int),tp=tp,sl=sl,max_bars=H,fee=.002,slippage=.0004))
    passed=m['trades']>=20 and m['profit_factor']>=1.5 and m['total_return']>0 and m['max_dd']<=.1 and stress['total_return']>0
    report=dict(status='PASSED_RESEARCH' if passed else 'REJECTED',reason='شروط الاختبار والاستقرار' if passed else 'لم يستوف جميع شروط القبول',model=MODEL_NAME,rows=len(raw),train_end=str(d.index[split-H-1]),test_start=str(test.index[0]),test_end=str(test.index[-1]),threshold=float(threshold),passed=bool(passed),**m,stress_return=stress['total_return'],note='نسبة النجاح=الصفقات الرابحة، وليست ضمانًا. بيانات تاريخية قد تكون فُحصت سابقًا. التراجع عند إغلاق الصفقات.')
    report={k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in report.items()}
    return report,dict(model=model,columns=list(x.columns),threshold=threshold,config_id=config_id(c),tp=tp,sl=sl,H=H)

class Engine:
    def __init__(self,store):
        self.s=store;self.lock=threading.RLock();self.cancel=threading.Event();self.thread=None;self.training=False
        self.models={};self.s.put('running',False)
        if self.s.get('snapshot'):
            for k,v in self.s.get('snapshot').items():self.s.put(k,v)
        if self.s.get('pending'):self.s.log('طلب منصة معلق من تشغيل سابق: يجب المصالحة قبل التشغيل')
    def broker(self,c):
        k,s=self.s.secret(mode='demo' if c['mode']=='demo' else 'real');return Bybit(k,s,c['mode'])
    def config(self):return self.s.get('config')
    def train(self):
        with self.lock:
            if self.training or self.s.get('running'):raise ValueError('أوقف التشغيل قبل التدريب')
            c=self.config()
            if not c:raise ValueError('احفظ الإعدادات أولًا')
            self.training=True
        def job():
            try:
                for symbol in c['symbols']:
                    self.s.put('training_progress',{'symbol':symbol,'stage':'جلب البيانات'})
                    try:
                        rows=Bybit().candles(symbol,c['interval'],c['start'],c['end']);self.s.put('training_progress',{'symbol':symbol,'stage':'تدريب ومعايرة واختبار'})
                        report,bundle=train_symbol(rows,c);report['config_id']=config_id(c)
                        self.s.put('report_'+symbol,report)
                        if bundle:
                            with open(self.s.root/(symbol+'.pkl'),'wb') as f:pickle.dump(bundle,f)
                            self.models[symbol]=bundle
                        self.s.log(symbol+': '+report['status'])
                    except Exception as e:
                        self.s.put('report_'+symbol,dict(passed=False,status='ERROR',reason=str(e)[:250],config_id=config_id(c)));self.s.log(symbol+': فشل التدريب؛ راجع التقرير')
            finally:self.training=False;self.s.put('training_progress',{'stage':'انتهى'})
        threading.Thread(target=job,daemon=True).start()
    def bundle(self,symbol,c):
        if symbol not in self.models:
            with open(self.s.root/(symbol+'.pkl'),'rb') as f:self.models[symbol]=pickle.load(f)
        b=self.models[symbol]
        if b['config_id']!=config_id(c):raise ValueError('أعد التدريب بعد تعديل العملات أو التاريخ أو الإطار')
        return b
    def start(self,live_confirm=False):
        with self.lock:
            c=self.config()
            if self.training:raise ValueError('التدريب لم يكتمل')
            if self.s.get('running'):return
            if self.thread is not None and self.thread.is_alive():raise ValueError('انتظر اكتمال إيقاف الجلسة السابقة')
            if self.s.get('pending'):raise ValueError('طلب منصة معلق؛ اضغط مصالحة الطلب')
            if c['mode']=='real' and (not live_confirm or os.getenv('ENABLE_REAL_TRADING')!='true'):raise ValueError('الحقيقي يحتاج تأكيدًا صريحًا وتمكين ENABLE_REAL_TRADING على الخادم')
            if self.s.get('positions') and self.s.get('portfolio_mode')!=c['mode']:raise ValueError('لا تبدل الوضع مع مراكز مفتوحة')
            for symbol in c['symbols']:
                r=self.s.get('report_'+symbol,{})
                if not r.get('passed') or r.get('config_id')!=config_id(c):raise ValueError(symbol+': لم يجتز الاختبار الحالي')
                self.bundle(symbol,c)
            if c['mode']!='paper':self.broker(c).credentials_check()
            if not self.s.get('portfolio') or self.s.get('portfolio_mode')!=c['mode']:
                self.s.put('portfolio',{'cash':c['amount'],'initial':c['amount'],'realized':0.,'wins':0,'closed':0});self.s.put('portfolio_mode',c['mode'])
            self.cancel.clear();self.s.put('running',True);self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();self.s.log('تشغيل '+c['mode'])
    def stop(self):
        self.cancel.set();self.s.put('running',False);self.s.log('إيقاف المحرك. المراكز المفتوحة لا تُباع تلقائيًا عند الإيقاف؛ راجعها قبل مغادرة الحساب.')
    def loop(self):
        try:
            while not self.cancel.is_set():
                with self.lock:
                    c=self.config()
                    if not self.s.get('running'):break
                    for symbol in c['symbols']:
                        if self.cancel.is_set():break
                        self.tick(symbol,c)
                self.cancel.wait(15)
        except Exception:
            self.s.put('running',False);self.s.log('توقف بسبب خطأ اتصال/تنفيذ. راجع الطلب المعلق أو تقرير الحالة؛ لن تُكرر أوامر الشراء تلقائيًا.')
        finally:self.s.put('running',False)
    def tick(self,symbol,c):
        broker=self.broker(c);price=Bybit().price(symbol);self.s.put('price_'+symbol,price)
        positions=self.s.get('positions',{});p=positions.get(symbol);portfolio=self.s.get('portfolio');budget=min(c['amount'],portfolio['initial'])/len(c['symbols'])
        if p:
            if price<=p['stop'] or price>=p['target'] or time.time()-p['opened']>=24*(1440 if c['interval']=='D' else int(c['interval']))*60:
                self.execute(symbol,'Sell',p['qty'],price,c,broker);return
            return
        step=(1440 if c['interval']=='D' else int(c['interval']))*60000;now=int(time.time()*1000);end=now//step*step-1
        if self.s.get('last_'+symbol)==end:return
        rows=Bybit().candles(symbol,c['interval'],end-1000*step,end);d=add_all(frame(rows,c['interval']));b=self.bundle(symbol,c);x=ml_features(d)[b['columns']].iloc[-1:]
        self.s.put('last_'+symbol,end)
        if not x.notna().all(axis=1).iloc[0]:return
        probability=float(b['model'].predict_proba(x)[0,1]);self.s.put('signal_'+symbol,{'probability':probability,'threshold':b['threshold'],'at':end})
        # New buys stop at 10% session equity loss. Existing holdings still processed first.
        equity=portfolio['cash']+sum(v['qty']*self.s.get('price_'+k,v['entry']) for k,v in positions.items())
        if equity<=portfolio['initial']*.9:return
        spend=min(budget,portfolio['cash'])*.995
        if probability>b['threshold'] and spend>=5:
            self.execute(symbol,'Buy',spend,price,c,broker,atr=float(d.atr.iloc[-1]))
    def execute(self,symbol,side,qty,price,c,broker,atr=0.):
        with self.lock:
            if self.cancel.is_set():return
            if c['mode']=='paper':
                if side=='Buy':amount=qty;quantity=amount/(price*1.0002*1.001);cost=amount
                else:quantity=qty;amount=quantity*price*.9998*.999;cost=0.
                self.apply_fill(symbol,side,quantity,amount,c,atr)
                return
            instruments={x['symbol']:x for x in broker.instruments()};inst=instruments[symbol];lot=inst['lotSizeFilter']
            normalized=floor_step(qty,lot['quotePrecision'] if side=='Buy' else lot['basePrecision'])
            if side=='Buy' and float(normalized)<float(lot['minOrderAmt']):raise ValueError('الميزانية أقل من الحد الأدنى')
            if side=='Sell' and float(normalized)*price<float(lot['minOrderAmt']):raise ValueError('القيمة أقل من الحد الأدنى؛ قد يكون رصيد غبار')
            if side=='Sell' and float(normalized)>float(lot.get('maxMarketOrderQty', '1e30')):raise ValueError('الكمية فوق حد أمر السوق')
            link='app'+uuid.uuid4().hex[:28]
            self.s.put('pending',dict(symbol=symbol,side=side,qty=normalized,link=link,mode=c['mode'],atr=atr))
            broker.order(symbol,side,normalized,link)
            # Acknowledgement is not a fill; never resubmit an uncertain request.
            for _ in range(10):
                if self.reconcile():return
                time.sleep(1)
            raise ValueError('طلب غير محسوم')
    def reconcile(self):
        with self.lock:
            pending=self.s.get('pending')
            if not pending:return True
            c=self.config();broker=self.broker({**c,'mode':pending['mode']});r=broker.order_status(pending['symbol'],pending['link'])
            if not r or r['orderStatus'] not in ['Filled','Cancelled','Rejected','PartiallyFilledCanceled']:return False
            filled=float(r.get('cumExecQty') or 0)
            if filled>0:
                executions=broker.executions(pending['symbol'],pending['link'])
                if not executions:raise ValueError('تفاصيل التنفيذ غير جاهزة')
                inst=next(x for x in broker.instruments() if x['symbol']==pending['symbol']);base=inst['baseCoin'];quote=inst['quoteCoin']
                value=float(r['cumExecValue']);base_fee=quote_fee=0.
                for ex in executions:
                    fee=float(ex.get('execFee') or 0);coin=ex.get('feeCurrency')
                    if fee and coin not in [base,quote]:raise ValueError('عملة عمولة غير مدعومة؛ راجع الطلب يدويًا')
                    if coin==base:base_fee+=fee
                    elif coin==quote:quote_fee+=fee
                quantity=filled-base_fee if pending['side']=='Buy' else filled+base_fee
                cash=value+quote_fee if pending['side']=='Buy' else value-quote_fee
                self.apply_fill(pending['symbol'],pending['side'],quantity,cash,c,pending['atr'],pending['link'])
            self.s.put('pending',None);return True
    def apply_fill(self,symbol,side,qty,value,c,atr,link=None):
        # Single persisted journal object makes a filled order idempotent across restarts.
        ledger=self.s.get('ledger',[])
        if link and any(x.get('link')==link for x in ledger):return
        portfolio=self.s.get('portfolio');positions=self.s.get('positions',{})
        if side=='Buy':
            entry=value/qty;positions[symbol]=dict(qty=qty,cost=value,entry=entry,stop=max(entry-atr,entry*.5),target=entry+1.5*atr,opened=time.time());portfolio['cash']-=value
            pnl=None
        else:
            pos=positions[symbol];used=min(qty,pos['qty']);cost=pos['cost']*used/pos['qty'];pnl=value-cost
            portfolio['cash']+=value;portfolio['realized']+=pnl;portfolio['closed']+=1;portfolio['wins']+=int(pnl>0)
            remaining=pos['qty']-used
            if remaining>1e-12:pos['qty']=remaining;pos['cost']-=cost
            else:del positions[symbol]
        ledger.append(dict(at=time.time(),symbol=symbol,side=side,qty=qty,value=value,pnl=pnl,mode=c['mode'],link=link))
        # Persist one transactional snapshot before mirrored compatibility keys.
        snapshot=dict(portfolio=portfolio,positions=positions,ledger=ledger)
        self.s.put('snapshot',snapshot);self.s.put('portfolio',portfolio);self.s.put('positions',positions);self.s.put('ledger',ledger)
        self.s.log(symbol+' '+side+' '+c['mode'])
    def status(self):
        c=self.config() or {};positions=self.s.get('positions',{});portfolio=self.s.get('portfolio',{})
        unreal=sum(v['qty']*self.s.get('price_'+k,v['entry'])-v['cost'] for k,v in positions.items())
        return dict(running=self.s.get('running',False),training=self.training,progress=self.s.get('training_progress',{}),config=c,portfolio=portfolio,positions=positions,unrealized=unreal,reports={x:self.s.get('report_'+x,{}) for x in c.get('symbols',[])},signals={x:self.s.get('signal_'+x,{}) for x in c.get('symbols',[])},pending=self.s.get('pending'),events=self.s.events(),trades=self.s.get('ledger',[])[-100:],server_time=time.time())
