"""インジケーターと翌日買い指値（t 日の情報のみで確定）。"""
from __future__ import annotations

import pandas as pd


def _ma_rising(close: pd.Series, period: int, days: int) -> pd.Series:
    """MA(period) が直近 days 営業日連続で上昇しているか（t 日終値時点で判定）。"""
    ma = close.rolling(period).mean()
    daily_up = ma.diff(1) > 0
    return daily_up.rolling(days).sum() == days


def _ma_perfect_order(close: pd.Series, periods: list[int]) -> pd.Series:
    """短期 MA > 次の MA > … のパーフェクトオーダー（t 日終値のみで判定）。"""
    sorted_p = sorted(int(p) for p in periods)
    mas = [close.rolling(p).mean() for p in sorted_p]
    cond = pd.Series(True, index=close.index)
    for i in range(len(mas) - 1):
        cond = cond & (mas[i] > mas[i + 1])
    return cond


def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """MA・ATR・BB と limit_next / trend_ok を付与。"""
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]

    df["MA_long"] = c.rolling(cfg["trend_ma_period"]).mean()

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(cfg["atr_period"]).mean()

    ma = c.rolling(cfg["bb_period"]).mean()
    sd = c.rolling(cfg["bb_period"]).std(ddof=0)
    df["BB_lower"] = ma - cfg["bb_sigma"] * sd

    if cfg["entry_method"] == "atr":
        df["limit_next"] = c - cfg["atr_mult"] * df["ATR"]
    elif cfg["entry_method"] == "bb":
        df["limit_next"] = df["BB_lower"]
    else:
        raise ValueError("entry_method は 'atr' か 'bb'")

    conds: list[pd.Series] = []
    if cfg.get("use_trend_filter", True):
        conds.append(c > df["MA_long"])
    if cfg.get("use_ma_rise_filter", False):
        for period in cfg.get("ma_rise_periods", [20, 60]):
            conds.append(_ma_rising(c, int(period), int(cfg.get("ma_rise_days", 5))))
    if cfg.get("use_ma_order_filter", False):
        periods = cfg.get("ma_order_periods", [5, 20, 60])
        conds.append(_ma_perfect_order(c, periods))

    if conds:
        df["trend_ok"] = pd.concat(conds, axis=1).all(axis=1)
    else:
        df["trend_ok"] = True

    return df