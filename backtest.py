"""
محرك الاختبار الرجعي: دخول عند افتتاح الشمعة التالية للإشارة، هدف ربح (TP)
ووقف خسارة (SL) بمضاعفات ATR أو نسبة مئوية، مع وقف متحرك اختياري،
وخروج زمني، ورسوم Bybit وانزلاق سعري.
إذا لمست الشمعة الهدف والوقف معاً نفترض الأسوأ (الوقف) — افتراض محافظ.
"""
import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:          # يعمل بدون numba لكن أبطأ
    def njit(*a, **k):
        def deco(f): return f
        return deco(a[0]) if a and callable(a[0]) else deco

TAKER_FEE = 0.001     # Research assumption per side; verify your actual spot fee
SLIPPAGE = 0.0002


@njit(cache=True)
def _simulate(o, h, l, c, atr, sig, tp_mult, sl_mult, use_pct, max_bars,
              trail_mult, cost):
    n = len(c)
    idx = np.flatnonzero(sig)            # نقفز مباشرة بين الإشارات (أسرع بكثير)
    out = np.zeros((len(idx), 6))  # entry_i, exit_i, side, ret, reason, bars
    k = 0
    busy_until = -1
    for q in range(len(idx)):
        i = idx[q]
        if i <= busy_until or i >= n - 2:
            continue
        s = sig[i]
        if np.isnan(atr[i]):
            continue
        e = i + 1
        entry = o[e]
        unit = entry if use_pct else atr[i]
        tp_d, sl_d = tp_mult * unit, sl_mult * unit
        if s > 0:
            tp, sl = entry + tp_d, entry - sl_d
        else:
            tp, sl = entry - tp_d, entry + sl_d
        best = entry
        exit_px, reason, j = c[n - 1], 3.0, n - 1
        last = min(n - 1, e + max_bars)
        for j in range(e, last + 1):
            if s > 0:
                if l[j] <= sl:
                    exit_px, reason = min(sl, o[j]), 1.0
                    break
                if h[j] >= tp:
                    exit_px, reason = max(tp, o[j]), 0.0
                    break
                if trail_mult > 0:
                    if h[j] > best:
                        best = h[j]
                    sl = max(sl, best - trail_mult * unit)
            else:
                if h[j] >= sl:
                    exit_px, reason = max(sl, o[j]), 1.0
                    break
                if l[j] <= tp:
                    exit_px, reason = min(tp, o[j]), 0.0
                    break
                if trail_mult > 0:
                    if l[j] < best:
                        best = l[j]
                    sl = min(sl, best + trail_mult * unit)
            if j == last:
                exit_px, reason = c[j], 2.0
        ret = s * (exit_px - entry) / entry - 2 * cost
        out[k, 0], out[k, 1], out[k, 2] = e, j, s
        out[k, 3], out[k, 4], out[k, 5] = ret, reason, j - e + 1
        k += 1
        busy_until = j      # صفقة واحدة في كل مرة
    return out[:k]


def run_backtest(df, signal, tp=2.0, sl=1.0, mode="atr", max_bars=48,
                 trail=0.0, fee=TAKER_FEE, slippage=SLIPPAGE):
    """signal: مصفوفة +1 شراء / -1 بيع / 0 لا شيء (محسوبة عند إغلاق الشمعة)."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) != len(df) or not np.isin(signal, [0, 1]).all():
        raise ValueError("Spot signals must be 0 or 1 and match candle count")
    arr = _simulate(df.open.values.astype(np.float64), df.high.values.astype(np.float64),
                    df.low.values.astype(np.float64), df.close.values.astype(np.float64),
                    df.atr.values.astype(np.float64), np.asarray(signal, dtype=np.float64),
                    float(tp), float(sl), mode == "pct", int(max_bars), float(trail),
                    fee + slippage)
    t = pd.DataFrame(arr, columns=["entry_i", "exit_i", "side", "ret", "reason", "bars"])
    if len(t):
        t["entry_time"] = df.index[t.entry_i.astype(int)]
        t["exit_time"] = df.index[t.exit_i.astype(int)]
        t["reason"] = t.reason.map({0: "TP", 1: "SL", 2: "TIME", 3: "END"})
    return t


def metrics(trades, risk_frac=1.0, periods_per_year=None, years=None):
    """إحصاءات احترافية. risk_frac: نسبة رأس المال في كل صفقة (1 = رافعة 1x)."""
    if len(trades) == 0:
        return dict(trades=0, win_rate=0, profit_factor=0, expectancy=0, total_return=0,
                    max_dd=0, sharpe=0, calmar=0, avg_bars=0)
    r = trades.ret.values * risk_frac
    eq = np.cumprod(1 + r)
    dd = 1 - eq / np.maximum.accumulate(np.r_[1.0, eq])[1:]
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    pf = gains / losses if losses > 0 else np.inf
    tpy = len(r) / years if years else len(r)
    sharpe = r.mean() / r.std() * np.sqrt(tpy) if r.std() > 0 else 0
    total = eq[-1] - 1
    cagr = (eq[-1]) ** (1 / years) - 1 if years and eq[-1] > 0 else total
    return dict(
        trades=len(r),
        win_rate=float((r > 0).mean()),
        profit_factor=float(pf),
        expectancy=float(r.mean()),
        total_return=float(total),
        max_dd=float(dd.max()),
        sharpe=float(sharpe),
        calmar=float(cagr / dd.max()) if dd.max() > 0 else 0.0,
        avg_bars=float(trades.bars.mean()),
    )


def years_of(df):
    return max((df.index[-1] - df.index[0]).total_seconds() / (365.25 * 86400), 1e-6)
