"""Additional causal entry hypotheses, fixed before validation."""
import numpy as np
import pandas as pd
RULES={
 'supertrend_flip':'Supertrend(10,3) flips up, close > EMA200, volume ratio >=1',
 'cci_recovery':'CCI20 crosses -100 upwards, not downtrend, volume ratio >=1',
 'mfi_recovery':'MFI14 crosses 20 upwards, not downtrend, volume ratio >=1',
 'obv_breakout':'OBV exceeds previous 20-bar maximum, close > EMA50 > EMA200, volume ratio >=1',
}

def supertrend(d,period=10,multiplier=3):
    c=d.close.to_numpy();hl=(d.high+d.low).to_numpy()/2
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/period,adjust=False,min_periods=period).mean().to_numpy()
    upper=hl+multiplier*atr;lower=hl-multiplier*atr;bull=np.zeros(len(d),dtype=bool)
    start=period-1
    for i in range(start+1,len(d)):
        upper[i]=upper[i] if upper[i]<upper[i-1] or c[i-1]>upper[i-1] else upper[i-1]
        lower[i]=lower[i] if lower[i]>lower[i-1] or c[i-1]<lower[i-1] else lower[i-1]
        bull[i]=(c[i]>=lower[i]) if bull[i-1] else (c[i]>upper[i])
    return pd.Series(bull,index=d.index)

def indicators(d):
    tp=(d.high+d.low+d.close)/3;mean=tp.rolling(20).mean()
    deviation=tp.rolling(20).apply(lambda x:np.mean(np.abs(x-x.mean())),raw=True)
    cci=(tp-mean)/(.015*deviation.replace(0,np.nan));cci=cci.mask(deviation==0,0)
    flow=tp*d.volume;change=tp.diff()
    positive=flow.where(change>0,0).rolling(14).sum();negative=flow.where(change<0,0).rolling(14).sum()
    mfi=100*positive/(positive+negative).replace(0,np.nan);mfi=mfi.mask((positive+negative)==0,50)
    obv=(np.sign(d.close.diff()).fillna(0)*d.volume).cumsum()
    return pd.DataFrame(dict(cci=cci,mfi=mfi,obv=obv,obv_high=obv.shift(1).rolling(20).max(),supertrend_up=supertrend(d)),index=d.index)

def signals(d,f,cross):
    liquid=f.volume_ratio>=1;not_down=f.regime!='downtrend'
    level=lambda x:pd.Series(x,index=d.index)
    return {
      'supertrend_flip':f.supertrend_up&~f.supertrend_up.shift(1,fill_value=False)&(d.close>f.ema200)&liquid,
      'cci_recovery':cross(f.cci,level(-100))&not_down&liquid,
      'mfi_recovery':cross(f.mfi,level(20))&not_down&liquid,
      'obv_breakout':cross(f.obv,f.obv_high)&(d.close>f.ema50)&(f.ema50>f.ema200)&liquid,
    }
