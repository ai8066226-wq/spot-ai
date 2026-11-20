"""
جلب بيانات الشموع التاريخية من Bybit (API v5 العام - لا يحتاج مفاتيح)
وتخزينها محلياً كـ CSV حتى لا تُعاد التنزيلات.
"""
import os
import time
import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.bybit.com/v5/market/kline"
# فترات Bybit: 1 3 5 15 30 60 120 240 360 720 D W M
INTERVAL_MS = {
    "1": 60_000, "3": 180_000, "5": 300_000, "15": 900_000, "30": 1_800_000,
    "60": 3_600_000, "120": 7_200_000, "240": 14_400_000, "360": 21_600_000,
    "720": 43_200_000, "D": 86_400_000,
}


def fetch_klines(symbol="BTCUSDT", interval="60", days=730, category="spot",
                 cache_dir="data", sleep=0.12) -> pd.DataFrame:
    """ينزّل الشموع على دفعات (1000 شمعة لكل طلب) من الأحدث للأقدم."""
    if category != "spot":
        raise ValueError("This project supports spot only")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"spot_{symbol}_{interval}_{days}d.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["time"], index_col="time")
        print(f"[data] تحميل من الذاكرة المحلية: {path} ({len(df)} شمعة)")
        return df

    end = int(time.time() * 1000) // INTERVAL_MS[interval] * INTERVAL_MS[interval] - 1
    start = end - days * 86_400_000
    rows, cursor = [], end
    session = requests.Session()
    while cursor > start:
        params = dict(category=category, symbol=symbol, interval=interval,
                      end=cursor, limit=1000)
        for attempt in range(5):
            try:
                r = session.get(BASE_URL, params=params, timeout=15)
                js = r.json()
                if js.get("retCode") == 0:
                    break
                raise RuntimeError(js.get("retMsg"))
            except Exception as e:
                print(f"[data] خطأ ({e}) محاولة {attempt + 1}")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("فشل الاتصال بـ Bybit")
        batch = js["result"]["list"]
        if not batch:
            break
        rows.extend(batch)
        oldest = int(batch[-1][0])
        if oldest <= start or len(batch) < 2:
            break
        cursor = oldest - 1
        time.sleep(sleep)
        print(f"\r[data] {len(rows)} شمعة...", end="")
    print()

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df = df.astype(float).drop_duplicates("ts").sort_values("ts")
    df = df[df.ts >= start]
    df["time"] = pd.to_datetime(df.ts, unit="ms", utc=True)
    df = df.set_index("time").drop(columns="ts")
    df.to_csv(path)
    print(f"[data] تم الحفظ: {path} ({len(df)} شمعة)")
    return df


def synthetic_ohlcv(n=20_000, seed=7, freq="1h") -> pd.DataFrame:
    """بيانات اصطناعية (مشي عشوائي مع أنظمة تقلب واتجاه) لاختبار المحرك بدون إنترنت."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1, 0, 1], size=n // 500 + 1), 500)[:n]
    vol = 0.006 * np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.3).clip(0.5, 2.5)
    ret = regime * 0.00015 + rng.standard_t(4, n) * vol * 0.6
    close = 30_000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    span = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + span * 0.5
    low = np.minimum(open_, close) - span * 0.5
    volume = rng.lognormal(10, 0.5, n) * (1 + 50 * np.abs(ret))
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(dict(open=open_, high=high, low=low, close=close,
                             volume=volume, turnover=volume * close), index=idx)
