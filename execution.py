"""Shared spot-only execution; no indicator dependencies."""
from dataclasses import dataclass
import numpy as np
from research_v3 import Risk, metrics

@dataclass(frozen=True)
class ExitSettings:
    initial_stop: float
    take_profit: float
    trailing_mode: str
    trailing_distance: float
    activation_gain: float
    def __post_init__(self):
        values=(self.initial_stop,self.take_profit,self.trailing_distance,self.activation_gain)
        if not all(np.isfinite(x) for x in values):raise ValueError('Nonfinite setting')
        if not 0<self.initial_stop<1 or self.take_profit<=0 or self.activation_gain<0:raise ValueError('Invalid price setting')
        if self.trailing_mode not in ('percent','atr') or self.trailing_distance<=0:raise ValueError('Invalid trailing setting')
        if self.trailing_mode=='percent' and self.trailing_distance>=1:raise ValueError('Trailing percent must be a fraction')



def updated_stop(old_stop,highest,close,atr_fraction,settings):
    trail=highest*(1-settings.trailing_distance) if settings.trailing_mode=='percent' else highest-close*atr_fraction*settings.trailing_distance
    return float(max(old_stop,trail))


def simulate_trailing(d,f,entry_signal,settings,risk=Risk(),details=False):
    signal=np.asarray(entry_signal,dtype=bool)
    if len(d)!=len(f) or len(d)!=len(signal):raise ValueError('Length mismatch')
    if not np.isfinite(f.atr).all():raise ValueError('Missing ATR')
    o=d.open.to_numpy();h=d.high.to_numpy();l=d.low.to_numpy();c=d.close.to_numpy();ts=d.timestamp.to_numpy();atr=f.atr.to_numpy()
    cash=1.;units=0.;pos=None;equity=[1.];trades=[];trace=[];peak=1.;dd=0.;halted=False;blocked_until=0;exposure=0
    day=None;day_start=1.;day_block=False
    for j in range(1,len(d)):
        current_day=int(ts[j]//86400000)
        if current_day!=day:day=current_day;day_start=equity[-1];day_block=False
        had_position=units>0
        if not units and not halted and not day_block and j>=blocked_until and j<len(d)-1 and signal[j-1]:
            entry=float(o[j]*(1+risk.slippage));stop=entry*(1-settings.initial_stop)
            per_unit_loss=entry*(1+risk.fee)-stop*(1-risk.slippage)*(1-risk.fee)
            units=min(cash*risk.max_allocation/(entry*(1+risk.fee)),cash*risk.risk_per_trade/per_unit_loss)
            spend=units*entry*(1+risk.fee);cash-=spend
            pos=dict(entry_index=j,entry_timestamp=int(ts[j]),entry_signal_timestamp=int(ts[j-1]),entry=entry,initial_stop=float(stop),stop=float(stop),target=entry*(1+settings.take_profit),highest_completed=entry,activated=False,spend=float(spend),quantity=units,entry_equity=equity[-1],planned_stop_risk=float(units*per_unit_loss))
        if units:
            exposure+=1;old_stop=pos['stop'];exit_price=None;reason=None
            if o[j]<=old_stop:exit_price=float(o[j]);reason='stop_gap'
            elif halted and had_position:exit_price=float(o[j]);reason='drawdown_halt_next_open'
            elif o[j]>=pos['target']:exit_price=float(pos['target']);reason='target_gap'
            elif l[j]<=old_stop:exit_price=float(old_stop);reason='trailing_stop' if old_stop>pos['initial_stop']+1e-12 else 'initial_stop'
            elif h[j]>=pos['target']:exit_price=float(pos['target']);reason='target'
            elif j==len(d)-1:exit_price=float(c[j]);reason='end_of_period'
            # Do not mark future intrabar lows after an opening exit.
            adverse=float(o[j]) if reason in ('stop_gap','drawdown_halt_next_open','target_gap') else float(max(l[j],old_stop))
            low_mark=cash+units*adverse*(1-risk.slippage)*(1-risk.fee)
            dd=max(dd,1-low_mark/peak)
            if exit_price is not None:
                proceeds=units*exit_price*(1-risk.slippage)*(1-risk.fee);pnl=proceeds-pos['spend'];cash+=proceeds
                trades.append(dict(**pos,exit_index=j,exit_timestamp=int(ts[j]),exit=float(exit_price*(1-risk.slippage)),reason=reason,pnl=float(pnl),**{'return':float(pnl/pos['spend'])}))
                if details:trace.append(dict(timestamp=int(ts[j]),stop_before=old_stop,stop_after=old_stop,event=reason))
                units=0.;pos=None;blocked_until=j+risk.cooldown+1
            else:
                pos['highest_completed']=max(pos['highest_completed'],float(h[j]))
                if pos['highest_completed']>=pos['entry']*(1+settings.activation_gain):pos['activated']=True
                if pos['activated']:pos['stop']=updated_stop(old_stop,pos['highest_completed'],c[j],atr[j],settings)
                if details:trace.append(dict(timestamp=int(ts[j]),stop_before=old_stop,stop_after=pos['stop'],highest_completed=pos['highest_completed'],activated=pos['activated'],event='close_update_effective_next_bar'))
        mark=cash+units*c[j]*(1-risk.slippage)*(1-risk.fee);peak=max(peak,mark);dd=max(dd,1-mark/peak);equity.append(float(mark))
        if dd>=risk.max_drawdown:halted=True
        if mark<=day_start*(1-risk.daily_loss):day_block=True
    result=metrics(trades,equity,dd,exposure,halted)
    if details:result.update(trade_log=trades,equity_curve=equity,stop_trace=trace)
    return result

