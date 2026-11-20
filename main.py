"""
نقطة التشغيل.

أمثلة:
  python main.py --symbol BTCUSDT --interval 60 --days 1095            # بحث قواعد + تعلم آلي
  python main.py --symbol ETHUSDT --interval 240 --days 1460 --mode rules
  python main.py --demo --fast                                         # تجربة على بيانات اصطناعية
"""
import argparse
import os
import json
import hashlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_bybit import fetch_klines, synthetic_ohlcv
from indicators import add_all
from search import search
from ml_engine import ml_search
from strategies import STRATEGIES
from backtest import run_backtest

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format)


def plot_equity(d, trades, title, path):
    eq = (1 + trades.set_index("exit_time").ret).cumprod()
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].plot(d.index, d.close, lw=0.7, color="#888")
    ax[0].set_title(title)
    ax[1].plot(eq.index, eq.values, lw=1.4, color="#1f6feb")
    ax[1].axhline(1, color="#999", lw=0.6)
    ax[1].set_ylabel("Equity (x)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="60", help="1 3 5 15 30 60 120 240 360 720 D")
    ap.add_argument("--days", type=int, default=1095)
    ap.add_argument("--mode", choices=["rules", "ml", "all"], default="all")
    ap.add_argument("--fast", action="store_true", help="شبكة مخارج مصغرة")
    ap.add_argument("--csv", help="Local OHLCV CSV with timestamp in milliseconds; no download")
    ap.add_argument("--demo", action="store_true", help="بيانات اصطناعية بدون إنترنت")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.csv and a.demo:
        ap.error("Choose --csv or --demo")
    if a.csv:
        raw = pd.read_csv(a.csv)
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if not set(required).issubset(raw.columns):
            raise ValueError("CSV missing OHLCV columns")
        raw = raw[required].apply(pd.to_numeric, errors="raise").sort_values("timestamp").reset_index(drop=True)
        if len(raw) < 1500 or not np.isfinite(raw.to_numpy()).all():
            raise ValueError("Insufficient or nonfinite data")
        step = int(a.interval) * 60000 if a.interval != "D" else 86400000
        if raw.timestamp.duplicated().any() or not (raw.timestamp.diff().dropna() == step).all():
            raise ValueError("Duplicate timestamps, gaps or wrong interval")
        if (raw[["open", "high", "low", "close"]] <= 0).any().any() or (raw.volume < 0).any():
            raise ValueError("Invalid price or volume")
        if ((raw.high < raw[["open", "close", "low"]].max(axis=1)) | (raw.low > raw[["open", "close", "high"]].min(axis=1))).any():
            raise ValueError("Invalid candle range")
        raw = raw[raw.timestamp + step <= pd.Timestamp.now(tz="UTC").timestamp() * 1000].copy()
        raw.index = pd.to_datetime(raw.pop("timestamp"), unit="ms", utc=True)
        raw.index.name = "time"
        if len(raw) < 1500:
            raise ValueError("Insufficient closed candles")
    else:
        raw = synthetic_ohlcv() if a.demo else fetch_klines(a.symbol, a.interval, a.days)
    metadata = dict(symbol=a.symbol, interval=a.interval, source="uploaded_csv" if a.csv else ("synthetic" if a.demo else "bybit_spot"), synthetic=a.demo, rows=len(raw), start=str(raw.index[0]), end=str(raw.index[-1]), mode=a.mode, csv_sha256=hashlib.sha256(open(a.csv,"rb").read()).hexdigest() if a.csv else None, research_only=True, live_enabled=False, notes=["Historical data may have been examined in previous experiments; results are exploratory, not a fresh independent test.","ML parameter ranking reuses walk-forward results for selection; fresh future paper data required."])
    with open(f"{a.out}/run_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    d = add_all(raw).iloc[200:]
    print(f"[main] {len(d)} شمعة من {d.index[0]} إلى {d.index[-1]}")

    if a.mode in ("rules", "all"):
        res, oos = search(d, fast=a.fast)
        res.to_csv(f"{a.out}/all_configs_dev.csv", index=False)
        print("\n========== أفضل الاستراتيجيات (نتيجة البيانات المخفية OOS) ==========")
        if oos.empty:
            print("لم تجتز أي استراتيجية فلتر الصلابة — وهذا بحد ذاته نتيجة مهمة.")
        else:
            cols = ["strategy", "params", "side", "tp", "sl", "trail", "max_bars", "DEV_pf",
                    "OOS_trades", "OOS_win_rate", "OOS_profit_factor", "OOS_total_return",
                    "OOS_max_dd", "OOS_sharpe", "passed"]
            print(oos[cols].to_string(index=False))
            oos.to_csv(f"{a.out}/top_strategies_oos.csv", index=False)
            print(f"\n⚠ تم اختبار {res.attrs['n_tested']:,} تركيبة. كلما زاد العدد زاد احتمال "
                  "الربح بالصدفة؛ اعتمد فقط على ما نجح في OOS وبعدد صفقات كافٍ.")
            win = oos[oos.passed].head(1)
            if len(win):
                r = win.iloc[0]
                sig = STRATEGIES[r.strategy][0](d, side=r.side, **r.params)
                t = run_backtest(d, sig, tp=r.tp, sl=r.sl, trail=r.trail, max_bars=r.max_bars)
                t.to_csv(f"{a.out}/best_rule_trades.csv", index=False)
                plot_equity(d, t, f"{r.strategy} {r.params} {r.side} TP{r.tp} SL{r.sl}",
                            f"{a.out}/best_rule_equity.png")

    if a.mode in ("ml", "all"):
        table, best = ml_search(d)
        print("\n========== التعلم الآلي (كل النتائج خارج العينة walk-forward) ==========")
        print(table.to_string(index=False))
        table.to_csv(f"{a.out}/ml_results.csv", index=False)
        if best:
            m, t, imp, (tp, sl, H) = best
            t.to_csv(f"{a.out}/best_ml_trades.csv", index=False)
            plot_equity(d, t, f"ML TP{tp} SL{sl} H{H}", f"{a.out}/best_ml_equity.png")
            if imp is not None:
                print("\nأهم الميزات:")
                print(imp.head(12).to_string())
                imp.to_csv(f"{a.out}/feature_importance.csv")
    print(f"\n[main] النتائج في مجلد: {a.out}/")


if __name__ == "__main__":
    main()
