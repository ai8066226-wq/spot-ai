"""
مكتبة استراتيجيات قائمة على مؤشرات Bybit. كل استراتيجية دالة تُرجع إشارة
(+1 / -1 / 0) عند إغلاق الشمعة، مع شبكة معاملات للبحث.
"""
import numpy as np
import pandas as pd


def _cross_up(a, b):   return (a > b) & (a.shift() <= b.shift())
def _cross_dn(a, b):   return (a < b) & (a.shift() >= b.shift())


def _sig(long_cond, short_cond, side):
    if side != "long":
        raise ValueError("Spot strategies accept long only")
    s = np.zeros(len(long_cond))
    if side in ("both", "long"):
        s[long_cond.fillna(False).values] = 1
    if side in ("both", "short"):
        s[short_cond.fillna(False).values] = -1
    return s


def ema_cross_trend(d, fast=9, slow=21, trend=200, side="long"):
    f, s = d[f"ema{fast}"], d[f"ema{slow}"]
    up = d.close > d[f"ema{trend}"]
    return _sig(_cross_up(f, s) & up, _cross_dn(f, s) & ~up, side)


def rsi_reversion(d, lo=30, hi=70, trend=200, side="long"):
    """ارتداد RSI مع الاتجاه العام فقط (شراء التصحيحات في اتجاه صاعد)."""
    up = d.close > d[f"ema{trend}"]
    return _sig(_cross_up(d.rsi, pd.Series(lo, d.index)) & up,
                _cross_dn(d.rsi, pd.Series(hi, d.index)) & ~up, side)


def macd_adx(d, adx_min=25, side="long"):
    trending = d.adx > adx_min
    return _sig(_cross_up(d.macd, d.macd_sig) & trending & (d.pdi > d.mdi),
                _cross_dn(d.macd, d.macd_sig) & trending & (d.mdi > d.pdi), side)


def supertrend_flip(d, trend=50, side="long"):
    flip_up = (d.st_dir == 1) & (d.st_dir.shift() == -1)
    flip_dn = (d.st_dir == -1) & (d.st_dir.shift() == 1)
    up = d.close > d[f"ema{trend}"]
    return _sig(flip_up & up, flip_dn & ~up, side)


def bb_squeeze_breakout(d, vol_mult=1.5, side="long"):
    """انفجار تقلب: خروج من ضغط بولنجر/كيلتنر مع حجم أعلى من المتوسط."""
    sq = ((d.bb_up - d.bb_lo) < (d.kc_up - d.kc_lo)).shift().fillna(False).astype(bool)
    volok = d.volume > vol_mult * d.volume.rolling(20).mean()
    return _sig(sq & (d.close > d.bb_up) & volok, sq & (d.close < d.bb_lo) & volok, side)


def bb_mean_reversion(d, adx_max=20, side="long"):
    ranging = d.adx < adx_max
    return _sig(_cross_up(d.close, d.bb_lo) & ranging, _cross_dn(d.close, d.bb_up) & ranging, side)


def stochrsi_trend(d, lo=20, hi=80, trend=50, side="long"):
    up = (d.close > d[f"ema{trend}"]) & (d[f"ema{trend}"] > d.ema200)
    dn = (d.close < d[f"ema{trend}"]) & (d[f"ema{trend}"] < d.ema200)
    return _sig(_cross_up(d.stk, d["std"]) & (d.stk < lo) & up,
                _cross_dn(d.stk, d["std"]) & (d.stk > hi) & dn, side)


def donchian_breakout(d, adx_min=20, side="long"):
    return _sig((d.close > d.dc_up.shift()) & (d.adx > adx_min),
                (d.close < d.dc_lo.shift()) & (d.adx > adx_min), side)


def ichimoku_tk(d, side="long"):
    cloud_top = d[["span_a", "span_b"]].max(axis=1)
    cloud_bot = d[["span_a", "span_b"]].min(axis=1)
    return _sig(_cross_up(d.tenkan, d.kijun) & (d.close > cloud_top),
                _cross_dn(d.tenkan, d.kijun) & (d.close < cloud_bot), side)


def vwap_mfi(d, lo=25, hi=75, side="long"):
    return _sig(_cross_up(d.close, d.vwap) & (d.mfi.shift() < lo + 20) & (d.ema50 > d.ema200),
                _cross_dn(d.close, d.vwap) & (d.mfi.shift() > hi - 20) & (d.ema50 < d.ema200), side)


# اسم الاستراتيجية -> (الدالة، شبكة المعاملات)
STRATEGIES = {
    "ema_cross_trend":   (ema_cross_trend,   dict(fast=[9, 21], slow=[21, 50], trend=[100, 200])),
    "rsi_reversion":     (rsi_reversion,     dict(lo=[25, 30, 35], hi=[65, 70, 75], trend=[100, 200])),
    "macd_adx":          (macd_adx,          dict(adx_min=[18, 22, 25, 30])),
    "supertrend_flip":   (supertrend_flip,   dict(trend=[50, 100, 200])),
    "bb_squeeze":        (bb_squeeze_breakout, dict(vol_mult=[1.0, 1.3, 1.6])),
    "bb_mean_reversion": (bb_mean_reversion, dict(adx_max=[15, 20, 25])),
    "stochrsi_trend":    (stochrsi_trend,    dict(lo=[20, 30], hi=[70, 80], trend=[21, 50])),
    "donchian_breakout": (donchian_breakout, dict(adx_min=[15, 20, 25])),
    "ichimoku_tk":       (ichimoku_tk,       dict()),
    "vwap_mfi":          (vwap_mfi,          dict(lo=[20, 30], hi=[70, 80])),
}
