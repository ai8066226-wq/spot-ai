import hmac,os,time,threading,base64,hashlib
from datetime import datetime,timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI,Depends,Header,HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,field_validator,model_validator
from store import Store
from service import Engine,INTERVALS
from broker import Bybit
TOKEN=os.environ.get('ENGINE_TOKEN','')
KEY=os.environ.get('ENGINE_ENCRYPTION_KEY','')
SEED=os.environ.get('ENGINE_ENCRYPTION_SEED','')
if not KEY and len(SEED)>=32:
    KEY=base64.urlsafe_b64encode(hashlib.sha256(SEED.encode()).digest()).decode()
if len(TOKEN)<32 or not KEY:raise RuntimeError('Set ENGINE_TOKEN (>=32 chars) and ENGINE_ENCRYPTION_KEY (Fernet key) before starting')
s=Store(os.getenv('ENGINE_DATA','state'),KEY.encode());engine=Engine(s)
if s.get('snapshot'):
    for k,v in s.get('snapshot').items():s.put(k,v)
@asynccontextmanager
async def lifespan(app):
    yield
    engine.stop()
app=FastAPI(title='Bybit Spot Engine',version='1.0',lifespan=lifespan,docs_url=None,redoc_url=None)
def auth(authorization:str=Header(default='')):
    if not hmac.compare_digest(authorization,'Bearer '+TOKEN):raise HTTPException(401,'Unauthorized')
@app.get('/healthz')
def health():return {'status':'ok'}

class Config(BaseModel):
    symbols:list[str]=Field(min_length=1,max_length=10)
    interval:str='15'
    amount:float=Field(gt=0,le=1000000)
    mode:str='paper'
    start:int
    end:int
    @field_validator('amount')
    @classmethod
    def finite(cls,v):
        import math
        if not math.isfinite(v):raise ValueError('مبلغ غير صالح')
        return v
    @field_validator('symbols')
    @classmethod
    def symbols_valid(cls,v):
        import re
        if len(set(v))!=len(v) or any(not re.fullmatch('[A-Z0-9]{2,24}USDT',x) for x in v):raise ValueError('اختر رموز USDT فريدة')
        return v
    @field_validator('interval')
    @classmethod
    def interval_valid(cls,v):
        if v not in INTERVALS:raise ValueError('زمن غير مدعوم')
        return v
    @field_validator('mode')
    @classmethod
    def mode_valid(cls,v):
        if v not in ['paper','demo','real']:raise ValueError('وضع غير مدعوم')
        if os.getenv('PAPER_ONLY')=='true' and v!='paper':raise ValueError('هذه الاستضافة للتجربة الورقية فقط')
        return v
    @model_validator(mode='after')
    def dates(self):
        if not 0<self.start<self.end<=int(time.time()*1000):raise ValueError('تواريخ غير صالحة أو مستقبلية')
        if (self.end-self.start)/((1440 if self.interval=='D' else int(self.interval))*60000)>100000:raise ValueError('الفترة تتجاوز100000 شمعة')
        return self
class Keys(BaseModel):
    key:str=Field(min_length=8,max_length=512)
    secret:str=Field(min_length=8,max_length=512)
    mode:str='real'
class Start(BaseModel):live_confirm:bool=False
@app.exception_handler(ValueError)
async def value_error(request,exc):return JSONResponse(status_code=400,content={'detail':str(exc)})
@app.exception_handler(RuntimeError)
async def runtime_error(request,exc):return JSONResponse(status_code=502,content={'detail':str(exc)[:300]})
@app.get('/v1/status',dependencies=[Depends(auth)])
def status():return engine.status()
@app.get('/v1/symbols',dependencies=[Depends(auth)])
def symbols():return {'symbols':[dict(symbol=x['symbol'],base=x['baseCoin'],quote=x['quoteCoin']) for x in Bybit().instruments()]}
@app.post('/v1/config',dependencies=[Depends(auth)])
def config(c:Config):
    with engine.lock:
        if engine.training or s.get('running'):raise ValueError('أوقف المحرك وانتظر التدريب قبل تعديل الإعدادات')
        old=s.get('config');new=c.model_dump()
        if old and s.get('positions') and old!=new:raise ValueError('توجد مراكز مفتوحة؛ أغلقها قبل تغيير إعدادات الجلسة')
        available={x['symbol'] for x in Bybit().instruments()}
        if not set(c.symbols)<=available:raise ValueError('إحدى العملات غير متاحة للتداول الفوري USDT')
        s.put('config',new);return {'ok':True}
@app.post('/v1/credentials',dependencies=[Depends(auth)])
def credentials(k:Keys):
    if os.getenv('PAPER_ONLY')=='true':raise ValueError('المفاتيح غير مطلوبة للاستضافة الورقية')
    if k.mode not in ['real','demo']:raise ValueError('حدد حساب demo أوreal')
    if s.get('running') or engine.training:raise ValueError('أوقف المحرك قبل تغيير المفاتيح')
    Bybit(k.key,k.secret,k.mode).credentials_check();s.secret(k.key,k.secret,k.mode);return {'ok':True,'message':'تم التحقق وحفظ المفاتيح مشفرة'}
@app.delete('/v1/credentials/{mode}',dependencies=[Depends(auth)])
def delete_keys(mode:str):
    if s.get('running') or s.get('pending') or s.get('positions'):raise ValueError('لا تحذف المفاتيح مع مراكز أو أوامر معلقة')
    if mode not in ['real','demo']:raise ValueError('Invalid mode')
    s.put('credentials_'+mode,None);return {'ok':True}
@app.post('/v1/train',dependencies=[Depends(auth)])
def train():engine.train();return {'ok':True}
@app.post('/v1/start',dependencies=[Depends(auth)])
def start(body:Start):
    if not engine.config():raise ValueError('احفظ الإعدادات')
    engine.start(body.live_confirm);return {'ok':True}
@app.post('/v1/stop',dependencies=[Depends(auth)])
def stop():engine.stop();return {'ok':True}
@app.post('/v1/reconcile',dependencies=[Depends(auth)])
def reconcile():return {'resolved':engine.reconcile()}
