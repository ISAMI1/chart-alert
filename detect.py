"""日足チャートパターン判定（GitHub Actions から毎営業日実行）

watchlist.csv の銘柄について、最新の日足で次の5パターンの「上抜け・発生」を判定し、
latest_result.json に保存する。
  - カップウィズハンドル / 逆三尊 / ダブルボトム / ゴールデンクロス(25日・75日) / 三角保ち合い上放れ
"""
import csv
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

JST = timezone(timedelta(hours=9))


# ---------- 共通 ----------
def pivots(arr, k, kind):
    """左右k本の中で最安値(low)/最高値(high)になる確定済みの位置"""
    idx = []
    n = len(arr)
    for i in range(k, n - k):
        w = arr[i - k:i + k + 1]
        if (kind == "low" and arr[i] == w.min()) or (kind == "high" and arr[i] == w.max()):
            if not idx or i - idx[-1] > 1:
                idx.append(i)
    return idx


def crossed_up(prev_close, close, level_prev, level_now):
    return prev_close <= level_prev and close > level_now


# ---------- 各パターン ----------
def golden_cross(df):
    c = df["Close"]
    s25, s75 = c.rolling(25).mean(), c.rolling(75).mean()
    if len(c) < 76 or np.isnan(s75.iloc[-2]):
        return None
    if s25.iloc[-2] <= s75.iloc[-2] and s25.iloc[-1] > s75.iloc[-1]:
        return f"25日線({s25.iloc[-1]:.0f})が75日線({s75.iloc[-1]:.0f})を上抜け"
    return None


def double_bottom(df, lookback=90):
    d = df.iloc[-lookback:]
    lo, hi, cl = d["Low"].values, d["High"].values, d["Close"].values
    tr = [t for t in pivots(lo, 5, "low") if t < len(d) - 2]
    for a_i in range(len(tr) - 1, 0, -1):
        t2 = tr[a_i]
        for t1 in reversed(tr[:a_i]):
            if t2 - t1 < 10:
                continue
            b = min(lo[t1], lo[t2])
            if abs(lo[t1] - lo[t2]) / b > 0.04:
                continue
            neck = hi[t1:t2 + 1].max()
            if neck < b * 1.06:
                continue
            if hi[t2 + 1:-1].size and cl[t2 + 1:-1].max() > neck:
                continue  # 既に上抜け済み
            if crossed_up(cl[-2], cl[-1], neck, neck):
                return f"安値{lo[t1]:.0f}/{lo[t2]:.0f}の二番底、ネックライン{neck:.0f}を上抜け"
    return None


def inverse_hs(df, lookback=130):
    d = df.iloc[-lookback:]
    lo, hi, cl = d["Low"].values, d["High"].values, d["Close"].values
    tr = [t for t in pivots(lo, 5, "low") if t < len(d) - 2]
    n = len(d)
    for j in range(len(tr) - 3, -1, -1):
        a, b, c = tr[j], tr[j + 1], tr[j + 2]
        la, lb, lc = lo[a], lo[b], lo[c]
        if not (lb < la * 0.97 and lb < lc * 0.97):
            continue
        if abs(la - lc) / min(la, lc) > 0.06:
            continue
        p1 = a + int(np.argmax(hi[a:b + 1]))
        p2 = b + int(np.argmax(hi[b:c + 1]))
        if p2 == p1:
            continue
        slope = (hi[p2] - hi[p1]) / (p2 - p1)
        neck_now = hi[p2] + slope * (n - 1 - p2)
        neck_prev = hi[p2] + slope * (n - 2 - p2)
        between = [hi[p2] + slope * (i - p2) for i in range(c + 1, n - 1)]
        if any(cl[i] > between[i - c - 1] for i in range(c + 1, n - 1)):
            continue
        if crossed_up(cl[-2], cl[-1], neck_prev, neck_now):
            return f"頭{lb:.0f}・肩{la:.0f}/{lc:.0f}、ネックライン{neck_now:.0f}を上抜け"
        break
    return None


def cup_with_handle(df, lookback=200):
    d = df.iloc[-lookback:]
    lo, hi, cl = d["Low"].values, d["High"].values, d["Close"].values
    n = len(d)
    peaks = pivots(hi, 5, "high")
    for r in reversed(peaks):
        handle_len = n - 1 - r
        if handle_len < 3 or handle_len > 25:
            continue
        for l in reversed([p for p in peaks if p < r - 30]):
            hl, hr = hi[l], hi[r]
            if abs(hl - hr) / hl > 0.05:
                continue
            if hi[l + 1:r].max() > max(hl, hr) * 1.01:
                continue
            bottom = lo[l:r + 1].min()
            depth = (max(hl, hr) - bottom) / max(hl, hr)
            if not 0.12 <= depth <= 0.40:
                continue
            handle_low = lo[r:n - 1].min()
            if handle_low < bottom + (hr - bottom) * 0.5 or handle_low < hr * 0.85:
                continue
            pivot_level = hi[r:n - 1].max()
            if crossed_up(cl[-2], cl[-1], pivot_level, pivot_level):
                return f"カップ深さ{depth*100:.0f}%・取っ手{handle_len}日、{pivot_level:.0f}を上抜け"
            break
    return None


def triangle_breakout(df, window=35, vol_ratio=None):
    d = df.iloc[-window - 1:-1]  # 当日を除く期間で保ち合いを判定
    lo, hi = d["Low"].values, d["High"].values
    ph, pl = pivots(hi, 3, "high"), pivots(lo, 3, "low")
    if len(ph) < 2 or len(pl) < 2:
        return None
    sh, ih = np.polyfit(ph, hi[ph], 1)
    sl, il = np.polyfit(pl, lo[pl], 1)
    price = df["Close"].iloc[-2]
    tol = price * 0.0015
    if sh > tol or sl < -tol or sh - sl > -price * 0.0005:
        return None  # 上値が切り下がり(水平)・下値が切り上がり(水平)で収束していない
    width_start = (ih) - (il)
    width_end = (ih + sh * (window - 1)) - (il + sl * (window - 1))
    if width_start <= 0 or width_end <= 0 or width_end > width_start * 0.7:
        return None
    upper_now = ih + sh * window
    upper_prev = ih + sh * (window - 1)
    cl = df["Close"].values
    if crossed_up(cl[-2], cl[-1], upper_prev, upper_now) and (vol_ratio or 0) >= 1.3:
        return f"{window}日の保ち合い上辺{upper_now:.0f}を出来高{vol_ratio:.1f}倍で上抜け"
    return None


PATTERNS = [
    ("カップウィズハンドル", cup_with_handle),
    ("逆三尊", inverse_hs),
    ("ダブルボトム", double_bottom),
    ("ゴールデンクロス", golden_cross),
    ("三角保ち合い上放れ", triangle_breakout),
]


def analyze(df):
    """df: 列 Open/High/Low/Close/Volume、日付昇順。該当パターン [(pattern, detail)] を返す"""
    v = df["Volume"].values
    vr = float(v[-1] / v[-21:-1].mean()) if len(v) > 21 and v[-21:-1].mean() > 0 else None
    out = []
    for name, fn in PATTERNS:
        try:
            r = fn(df, vol_ratio=vr) if fn is triangle_breakout else fn(df)
        except Exception:  # 1パターンの失敗で全体を止めない
            r = None
        if r:
            out.append((name, r))
    return out, vr


# ---------- 実行 ----------
def load_watchlist(path="watchlist.csv"):
    with open(path, encoding="utf-8-sig") as f:
        return [(r["code"].strip(), r["name"].strip()) for r in csv.DictReader(f) if r.get("code")]


def fetch(code):
    import yfinance as yf
    sym = code if "." in code else f"{code}.T"
    for attempt in range(3):
        try:
            df = yf.Ticker(sym).history(period="1y", interval="1d", auto_adjust=False)
            if df is not None and len(df) > 0:
                return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def main():
    wl = load_watchlist()
    signals, errors, dates = [], [], []
    for code, name in wl:
        df = fetch(code)
        if df is None or len(df) < 80:
            errors.append(f"{code} {name}: 株価データ取得失敗")
            continue
        last_date = df.index[-1].strftime("%Y-%m-%d")
        dates.append(last_date)
        hits, vr = analyze(df)
        for pattern, detail in hits:
            signals.append({
                "date": last_date, "code": code, "name": name, "pattern": pattern,
                "close": round(float(df["Close"].iloc[-1]), 1),
                "volume_ratio": round(vr, 2) if vr else None, "detail": detail,
            })
        time.sleep(0.5)
    result = {
        "run_at": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
        "data_date": max(dates) if dates else None,
        "watchlist_count": len(wl),
        "signals": signals,
        "errors": errors,
    }
    with open("latest_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
