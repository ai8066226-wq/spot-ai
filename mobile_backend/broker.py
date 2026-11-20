import hashlib,hmac,json,time,uuid
from decimal import Decimal,ROUND_DOWN
import requests

class Bybit:
    def __init__(self,key='',secret='',mode='paper'):
        self.key=key;self.secret=secret;self.host='https://api-demo.bybit.com' if mode=='demo' else 'https://api.bybit.com'
    def request(self,method,path,data=None,private=False):
        data=data or {};payload=requests.models.RequestEncodingMixin._encode_params(sorted(data.items())) if method=='GET' else json.dumps(data,separators=(',',':'))
        headers={}
        if private:
            if not self.key or not self.secret:raise ValueError('مفاتيح الحساب غير محفوظة')
            ts=str(int(time.time()*1000));window='5000'
            sign=hmac.new(self.secret.encode(),(ts+self.key+window+payload).encode(),hashlib.sha256).hexdigest()
            headers={'X-BAPI-API-KEY':self.key,'X-BAPI-TIMESTAMP':ts,'X-BAPI-RECV-WINDOW':window,'X-BAPI-SIGN':sign,'Content-Type':'application/json'}
        url=self.host+path+('?' + payload if method=='GET' and payload else '')
        r=requests.request(method,url,headers=headers,data=payload if method!='GET' else None,timeout=20);r.raise_for_status();j=r.json()
        if j.get('retCode')!=0:raise RuntimeError('Bybit '+str(j.get('retCode'))+': '+str(j.get('retMsg')))
        return j['result']
    def instruments(self):
        # Spot endpoint does not use cursor pagination.
        rows=self.request('GET','/v5/market/instruments-info',{'category':'spot'})['list']
        return [r for r in rows if r['status']=='Trading' and r['quoteCoin']=='USDT']
    def candles(self,symbol,interval,start,end):
        rows=[];cursor=end;step=(1440 if interval=='D' else int(interval))*60000
        while cursor>=start:
            page=self.request('GET','/v5/market/kline',dict(category='spot',symbol=symbol,interval=interval,start=start,end=cursor,limit=1000))['list']
            if not page:break
            rows.extend(page);old=min(int(x[0]) for x in page)
            if old>cursor:raise ValueError('Pagination stalled')
            cursor=old-1
            if len(rows)>100000:raise ValueError('اختر فترة أقصر: الحد100000 شمعة لكل عملة')
            time.sleep(.12)
        now=int(time.time()*1000)
        return sorted({int(x[0]):[int(x[0])]+list(map(float,x[1:6])) for x in rows if start<=int(x[0])<=end and int(x[0])+step<=now}.values())
    def price(self,symbol):return float(self.request('GET','/v5/market/tickers',dict(category='spot',symbol=symbol))['list'][0]['lastPrice'])
    def credentials_check(self):return self.request('GET','/v5/account/wallet-balance',{'accountType':'UNIFIED','coin':'USDT'},True)
    def order(self,symbol,side,qty,link):
        return self.request('POST','/v5/order/create',dict(category='spot',symbol=symbol,side=side,orderType='Market',qty=str(qty),marketUnit='quoteCoin' if side=='Buy' else 'baseCoin',isLeverage=0,orderFilter='Order',orderLinkId=link,slippageToleranceType='Percent',slippageTolerance='0.50'),True)
    def order_status(self,symbol,link):
        q=dict(category='spot',symbol=symbol,orderLinkId=link)
        rows=self.request('GET','/v5/order/realtime',q,True)['list']
        if not rows:rows=self.request('GET','/v5/order/history',q,True)['list']
        return rows[0] if rows else None
    def executions(self,symbol,link):
        return self.request('GET','/v5/execution/list',dict(category='spot',symbol=symbol,orderLinkId=link,limit=100),True)['list']

def floor_step(value,step):
    d=Decimal(str(value));s=Decimal(str(step));return format((d/s).to_integral_value(rounding=ROUND_DOWN)*s,'f')
