"""
محرك التعلم الآلي:
- وسم ثلاثي الحواجز (Triple-Barrier, López de Prado): لكل شمعة هل يصل السعر
  للهدف TP قبل الوقف SL خلال H شمعة؟  — يُحسب لكل زوج (TP, SL) على حدة.
- نموذج LightGBM (أو HistGradientBoosting من sklearn كبديل) لكل اتجاه.
- تحقق Walk-Forward متوسع مع فجوة تطهير (purge) لمنع تسرب المعلومات.
- عتبة الاحتمال تُختار على جزء معايرة داخل التدريب فقط ثم تُطبق خارج العينة.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from backtest import run_backtest, metrics, years_of
from indicators import ml_features

try:
    import lightgbm as lgb

    def make_model():
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                  min_child_samples=100, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
    MODEL_NAME = "LightGBM"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier

    def make_model():
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                              min_samples_leaf=100, l2_regularization=5.0,
                                              early_stopping=False)
    MODEL_NAME = "HistGradientBoosting (sklearn)"


def barrier_labels(d, tp, sl, horizon):
    """يرجع (y_long, y_short): 1 إذا لُمس الهدف قبل الوقف، وNaN في آخر H شمعة."""
    n = len(d)
    o, h, l, a = d.open.values, d.high.values, d.low.values, d.atr.values
    H = horizon
    entry = o[1:n - H + 1]                     # الدخول عند افتتاح الشمعة التالية
    hw = sliding_window_view(h[1:], H)[: n - H]
    lw = sliding_window_view(l[1:], H)[: n - H]
    at = a[: n - H]

    def first(mask):
        any_ = mask.any(axis=1)
        return np.where(any_, mask.argmax(axis=1), H + 1)

    out = []
    for side in (1, -1):
        if side == 1:
            t_tp = first(hw >= (entry + tp * at)[:, None])
            t_sl = first(lw <= (entry - sl * at)[:, None])
        else:
            t_tp = first(lw <= (entry - tp * at)[:, None])
            t_sl = first(hw >= (entry + sl * at)[:, None])
        y = ((t_tp < t_sl) & (t_tp <= H)).astype(float)   # التعادل = وقف (متحفظ)
        out.append(np.r_[y, np.full(H, np.nan)])
    return out


def _pick_threshold(p, d_cal, side, tp, sl, horizon, min_trades=15):
    best_th, best_score = None, 0.0
    for th in np.arange(0.40, 0.81, 0.025):
        sig = np.where(p > th, side, 0)
        t = run_backtest(d_cal, sig, tp=tp, sl=sl, max_bars=horizon)
        if len(t) < min_trades:
            continue
        m = metrics(t)
        score = m["expectancy"] * np.sqrt(m["trades"])
        if m["profit_factor"] > 1.05 and score > best_score:
            best_th, best_score = th, score
    return best_th


def walk_forward(d, tp=2.0, sl=1.0, horizon=48, n_splits=6, min_train=0.35,
                 sides=(1,), verbose=True):
    if tuple(sides) != (1,):
        raise ValueError("Spot ML accepts long only")
    X = ml_features(d)
    y_long, y_short = barrier_labels(d, tp, sl, horizon)
    labels = {1: y_long, -1: y_short}
    valid = X.notna().all(axis=1).values
    n = len(d)
    start = int(n * min_train)
    edges = np.linspace(start, n, n_splits + 1).astype(int)
    signal = np.zeros(n)
    importances = []

    for k in range(n_splits):
        te0, te1 = edges[k], edges[k + 1]
        tr_end = te0 - horizon                       # تطهير: لا تداخل بين الوسوم والاختبار
        cal0 = int(tr_end * 0.8)
        fold_sig = np.zeros(n)
        for side in sides:
            y = labels[side]
            m_fit = valid & ~np.isnan(y)
            idx_fit = np.flatnonzero(m_fit[: cal0 - horizon])
            idx_cal = np.arange(cal0, tr_end)
            model = make_model().fit(X.values[idx_fit], y[idx_fit])
            p_cal = np.zeros(n)
            ok = valid[idx_cal]
            p_cal[idx_cal[ok]] = model.predict_proba(X.values[idx_cal[ok]])[:, 1]
            th = _pick_threshold(p_cal[cal0:tr_end], d.iloc[cal0:tr_end], side, tp, sl, horizon)
            if th is None:
                if verbose:
                    print(f"  fold {k + 1} side {side:+d}: لا توجد عتبة مربحة على المعايرة → لا تداول")
                continue
            # إعادة التدريب على كل بيانات التدريب ثم التنبؤ خارج العينة
            idx_all = np.flatnonzero(m_fit[:tr_end])
            model = make_model().fit(X.values[idx_all], y[idx_all])
            idx_te = np.arange(te0, te1)
            ok = valid[idx_te]
            p = np.zeros(len(idx_te))
            p[ok] = model.predict_proba(X.values[idx_te[ok]])[:, 1]
            take = p > th
            # إذا تعارض الاتجاهان نلغي الإشارة
            clash = take & (fold_sig[idx_te] != 0)
            fold_sig[idx_te[take & ~clash]] = side
            fold_sig[idx_te[clash]] = 0
            if hasattr(model, "feature_importances_"):
                importances.append(pd.Series(model.feature_importances_, X.columns))
            if verbose:
                print(f"  fold {k + 1} side {side:+d}: عتبة={th:.3f} إشارات={take.sum()}")
        signal[te0:te1] = fold_sig[te0:te1]

    t = run_backtest(d, signal, tp=tp, sl=sl, max_bars=horizon)
    oos = d.iloc[edges[0]:]
    m = metrics(t, years=years_of(oos))
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False) if importances else None
    return m, t, imp


def ml_search(d, grid=((1.5, 1.0), (2.0, 1.0), (3.0, 1.5), (2.0, 2.0), (1.0, 1.0)),
              horizons=(24, 48), **kw):
    print(f"[ml] النموذج: {MODEL_NAME}")
    rows, best = [], None
    for (tp, sl), H in [(g, h) for g in grid for h in horizons]:
        print(f"[ml] TP={tp} SL={sl} H={H}")
        m, t, imp = walk_forward(d, tp=tp, sl=sl, horizon=H, **kw)
        rows.append(dict(tp=tp, sl=sl, horizon=H, **m))
        if m["trades"] >= 20 and (best is None or m["sharpe"] > best[0]["sharpe"]):
            best = (m, t, imp, (tp, sl, H))
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False), best
