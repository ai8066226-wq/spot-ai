"""
البحث عن استراتيجيات مربحة بطريقة احترافية (مقاومة للـ Overfitting):

1) تقسيم البيانات: فترة تطوير (DEV) + فترة اختبار نهائي مخفية (HOLDOUT) لا تُلمس أثناء البحث.
2) لكل تركيبة (استراتيجية × معاملات × اتجاه × TP × SL × وقف متحرك × مدة قصوى):
   نقيس الأداء على K نوافذ زمنية متتالية داخل DEV (walk-forward).
3) نقبل فقط التركيبات المربحة في معظم النوافذ وبعدد صفقات كافٍ،
   ونرتّبها بمقياس متحفظ (أسوأ/وسيط النوافذ وليس المجموع).
4) أفضل المرشحين يُختبرون مرة واحدة على HOLDOUT — هذا هو الرقم الذي تثق به.
"""
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from backtest import run_backtest, metrics, years_of
from strategies import STRATEGIES

EXIT_GRID = dict(
    tp=[1.0, 1.5, 2.0, 3.0, 4.0],      # مضاعفات ATR
    sl=[0.75, 1.0, 1.5, 2.0],
    trail=[0.0, 1.5],                  # 0 = بدون وقف متحرك
    max_bars=[24, 72],
)
FAST_EXIT_GRID = dict(tp=[1.5, 2.0, 3.0], sl=[1.0, 1.5], trail=[0.0], max_bars=[48])

_D = None  # بيانات مشتركة لعمليات المعالجة المتوازية


def _init(d):
    global _D
    _D = d


def _param_sets(grid):
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, vals))
        if "fast" in p and p["fast"] >= p["slow"]:
            continue
        yield p


def _eval_job(job):
    name, params, side, exit_grid, n_folds, min_trades = job
    func = STRATEGIES[name][0]
    d = _D
    sig = func(d, side=side, **params)
    if np.count_nonzero(sig) < min_trades * n_folds:
        return []
    edges = np.linspace(0, len(d), n_folds + 1).astype(int)
    fold_years = years_of(d) / n_folds
    rows = []
    for tp, sl, trail, mb in itertools.product(exit_grid["tp"], exit_grid["sl"],
                                               exit_grid["trail"], exit_grid["max_bars"]):
        t = run_backtest(d, sig, tp=tp, sl=sl, trail=trail, max_bars=mb)
        if len(t) < min_trades * n_folds:
            continue
        fold_id = np.searchsorted(edges, t.entry_i.values, side="right") - 1
        fm = [metrics(t[fold_id == k], years=fold_years) for k in range(n_folds)]
        pfs = np.array([min(m["profit_factor"], 10) for m in fm])
        rets = np.array([m["total_return"] for m in fm])
        ntr = np.array([m["trades"] for m in fm])
        allm = metrics(t, years=years_of(d))
        rows.append(dict(
            strategy=name, params=params, side=side, tp=tp, sl=sl, trail=trail, max_bars=mb,
            trades=allm["trades"], win_rate=allm["win_rate"], pf=allm["profit_factor"],
            ret=allm["total_return"], max_dd=allm["max_dd"], sharpe=allm["sharpe"],
            fold_pf_min=pfs.min(), fold_pf_med=np.median(pfs),
            folds_profitable=(rets > 0).mean(), fold_min_trades=ntr.min(),
        ))
    return rows


def search(d, holdout_frac=0.3, n_folds=4, min_trades=8, fast=False,
           sides=("long",), workers=None, top=15):
    if tuple(sides) != ("long",):
        raise ValueError("Spot research accepts long only")
    split = int(len(d) * (1 - holdout_frac))
    dev, hold = d.iloc[:split], d.iloc[split - 250:]   # 250 شمعة إحماء للمؤشرات
    exit_grid = FAST_EXIT_GRID if fast else EXIT_GRID

    jobs = [(name, p, side, exit_grid, n_folds, min_trades)
            for name, (_, grid) in STRATEGIES.items()
            for p in _param_sets(grid) for side in sides]
    n_cfg = len(jobs) * np.prod([len(v) for v in exit_grid.values()])
    print(f"[search] {len(jobs)} مجموعة إشارات × مخارج = {n_cfg:,} تركيبة")

    rows = []
    workers = workers or min(4, max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(dev,)) as ex:
        for i, r in enumerate(ex.map(_eval_job, jobs, chunksize=4)):
            rows.extend(r)
            if i % 20 == 0:
                print(f"\r[search] {i + 1}/{len(jobs)}", end="")
    print()
    res = pd.DataFrame(rows)
    if res.empty:
        print("[search] لا توجد نتائج")
        return res, pd.DataFrame()

    # فلتر الصلابة: مربح في ≥75% من النوافذ، وPF أدنى نافذة ≥ 0.9، وصفقات كافية بكل نافذة
    robust = res[(res.folds_profitable >= 0.75) & (res.fold_pf_min >= 0.9)
                 & (res.fold_min_trades >= min_trades) & (res.pf > 1.1)].copy()
    robust["score"] = robust.fold_pf_med * np.sqrt(robust.trades) * (1 - robust.max_dd)
    robust = robust.sort_values("score", ascending=False)
    # إزالة التكرار: أفضل مخرج لكل (استراتيجية، معاملات، اتجاه)
    robust["key"] = robust.strategy + robust.params.astype(str) + robust.side
    best = robust.drop_duplicates("key").head(top)
    print(f"[search] اجتاز فلتر الصلابة: {len(robust)} من {len(res)}")

    # الاختبار النهائي على البيانات المخفية
    hy = years_of(hold.iloc[250:])
    out = []
    for _, r in best.iterrows():
        func = STRATEGIES[r.strategy][0]
        sig = func(hold, side=r.side, **r.params)
        sig[:250] = 0
        t = run_backtest(hold, sig, tp=r.tp, sl=r.sl, trail=r.trail, max_bars=r.max_bars)
        m = metrics(t, years=hy)
        out.append({**{k: r[k] for k in ["strategy", "params", "side", "tp", "sl", "trail",
                                         "max_bars", "pf", "win_rate", "trades"]},
                    **{f"OOS_{k}": v for k, v in m.items()}})
    oos = pd.DataFrame(out)
    if not oos.empty:
        oos = oos.rename(columns=dict(pf="DEV_pf", win_rate="DEV_win", trades="DEV_trades"))
        oos["passed"] = (oos.OOS_profit_factor > 1.1) & (oos.OOS_trades >= 10)
    res.attrs["n_tested"] = int(n_cfg)
    return res, oos
