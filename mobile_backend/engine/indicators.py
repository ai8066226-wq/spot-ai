"""
المؤشرات المتوفرة في منصة Bybit (TradingView) مبرمجة بـ pandas/numpy مباشرة
لتطابق صيغ TradingView (RMA/Wilder) بدون الاعتماد على مكتبات خارجية متقلبة.
"""
import numpy as np
import pandas as pd


def ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def sma(s, n):  return s.rolling(n).mean()
def rma(s, n):  return s.ewm(alpha=1 / n, adjust=False).mean()   # Wilder


def rsi(close, n=14):
    d = close.diff()
    up, dn = rma(d.clip(lower=0), n), rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def true_range(df):
    pc = df.close.shift()
    return pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return rma(true_range(df), n)


def macd(close, fast=12, slow=26, sig=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, sig)
    return m, s, m - s


def bollinger(close, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def stoch_rsi(close, n=14, k=3, d=3):
    r = rsi(close, n)
    lo, hi = r.rolling(n).min(), r.rolling(n).max()
    st = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    kk = sma(st, k)
    return kk, sma(kk, d)


def adx(df, n=14):
    up = df.high.diff()
    dn = -df.low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(df), n)
    pdi = 100 * rma(pd.Series(plus_dm, df.index), n) / tr
    mdi = 100 * rma(pd.Series(minus_dm, df.index), n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx, n), pdi, mdi


def supertrend(df, n=10, mult=3.0):
    a = atr(df, n).values
    hl2 = ((df.high + df.low) / 2).values
    c = df.close.values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb = ub.copy(), lb.copy()
    direction = np.ones(len(df))
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if c[i] > fub[i - 1]:
            direction[i] = 1
        elif c[i] < flb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    line = np.where(direction == 1, flb, fub)
    return pd.Series(direction, df.index), pd.Series(line, df.index)


def mfi(df, n=14):
    tp = (df.high + df.low + df.close) / 3
    mf = tp * df.volume
    pos = mf.where(tp > tp.shift(), 0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(), 0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def obv(df):
    return (np.sign(df.close.diff()).fillna(0) * df.volume).cumsum()


def rolling_vwap(df, n=24):
    tp = (df.high + df.low + df.close) / 3
    return (tp * df.volume).rolling(n).sum() / df.volume.rolling(n).sum()


def donchian(df, n=20):
    return df.low.rolling(n).min(), df.high.rolling(n).max()


def keltner(df, n=20, mult=2.0):
    mid = ema(df.close, n)
    a = atr(df, n)
    return mid - mult * a, mid, mid + mult * a


def cci(df, n=20):
    tp = (df.high + df.low + df.close) / 3
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma(tp, n)) / (0.015 * md)


def ichimoku(df, t=9, k=26, s=52):
    tenkan = (df.high.rolling(t).max() + df.low.rolling(t).min()) / 2
    kijun = (df.high.rolling(k).max() + df.low.rolling(k).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(k)
    span_b = ((df.high.rolling(s).max() + df.low.rolling(s).min()) / 2).shift(k)
    return tenkan, kijun, span_a, span_b


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """يضيف كل المؤشرات كأعمدة (تُستخدم في الاستراتيجيات وكميزات للتعلم الآلي)."""
    d = df.copy()
    c = d.close
    d["atr"] = atr(d, 14)
    for n in (9, 21, 50, 100, 200):
        d[f"ema{n}"] = ema(c, n)
    d["rsi"] = rsi(c, 14)
    d["rsi7"] = rsi(c, 7)
    d["macd"], d["macd_sig"], d["macd_hist"] = macd(c)
    d["bb_lo"], d["bb_mid"], d["bb_up"] = bollinger(c)
    d["kc_lo"], _, d["kc_up"] = keltner(d)
    d["stk"], d["std"] = stoch_rsi(c)
    d["adx"], d["pdi"], d["mdi"] = adx(d)
    d["st_dir"], d["st_line"] = supertrend(d)
    d["mfi"] = mfi(d)
    d["obv"] = obv(d)
    d["vwap"] = rolling_vwap(d, 24)
    d["dc_lo"], d["dc_up"] = donchian(d, 20)
    d["cci"] = cci(d)
    d["tenkan"], d["kijun"], d["span_a"], d["span_b"] = ichimoku(d)
    return d


def ml_features(d: pd.DataFrame) -> pd.DataFrame:
    """ميزات مُطبَّعة (بلا قيم سعرية مطلقة) حتى يعمم النموذج عبر الأزمنة والأسعار."""
    c, a = d.close, d.atr
    f = pd.DataFrame(index=d.index)
    for k in (1, 3, 6, 12, 24, 48):
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    for n in (9, 21, 50, 200):
        f[f"dist_ema{n}"] = (c - d[f"ema{n}"]) / a
    f["ema_9_21"] = (d.ema9 - d.ema21) / a
    f["ema_50_200"] = (d.ema50 - d.ema200) / a
    f["atr_pct"] = a / c
    f["atr_ratio"] = a / a.rolling(100).mean()
    f["rsi"], f["rsi7"] = d.rsi, d.rsi7
    f["macd_hist"] = d.macd_hist / a
    f["macd"] = d.macd / a
    f["bb_pos"] = (c - d.bb_lo) / (d.bb_up - d.bb_lo)
    f["bb_width"] = (d.bb_up - d.bb_lo) / d.bb_mid
    f["squeeze"] = ((d.bb_up - d.bb_lo) < (d.kc_up - d.kc_lo)).astype(int)
    f["stk"], f["std"] = d.stk, d["std"]
    f["adx"], f["di_diff"] = d.adx, d.pdi - d.mdi
    f["st_dir"] = d.st_dir
    f["st_dist"] = (c - d.st_line) / a
    f["mfi"], f["cci"] = d.mfi, d.cci
    f["obv_slope"] = d.obv.diff(12) / d.volume.rolling(12).sum()
    f["vwap_dist"] = (c - d.vwap) / a
    f["dc_pos"] = (c - d.dc_lo) / (d.dc_up - d.dc_lo)
    f["vol_z"] = (d.volume - d.volume.rolling(48).mean()) / d.volume.rolling(48).std()
    f["cloud_dist"] = (c - d[["span_a", "span_b"]].max(axis=1)) / a
    f["tk_cross"] = (d.tenkan - d.kijun) / a
    f["body"] = (c - d.open) / a
    f["upper_wick"] = (d.high - d[["open", "close"]].max(axis=1)) / a
    f["lower_wick"] = (d[["open", "close"]].min(axis=1) - d.low) / a
    if isinstance(d.index, pd.DatetimeIndex):
        f["hour_sin"] = np.sin(2 * np.pi * d.index.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * d.index.hour / 24)
        f["dow"] = d.index.dayofweek
    return f.replace([np.inf, -np.inf], np.nan)
