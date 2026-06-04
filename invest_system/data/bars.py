"""情報主導型サンプリング：tick/volume/dollar バーとドル・インバランスバー。

統合ナレッジベース §3.2 / DP3 の実装。AFML (López de Prado 2018) ch.2。
時間バーは情報到着率の不均一を無視するため採用しない。

入力 ``trades`` は pandas.DataFrame：
  index : DatetimeIndex（約定時刻、昇順・一意）
  price : float（約定価格）
  volume: float（約定数量, base）
  side  : str 'buy'/'sell'（任意。無ければ tick rule で符号付け）
出力 ``bars`` は DatetimeIndex（バー確定時刻 = ブロック内最後の約定時刻）：
  open, high, low, close, volume, dollar, vwap, n_ticks, start_time
未確定の末尾バー（しきい値未達）は出力しない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_BAR_COLUMNS = ["open", "high", "low", "close", "volume", "dollar",
                "vwap", "n_ticks", "start_time"]


def _aggregate(prices, vols, dollars, times, start, end) -> dict:
    seg_v = vols[start:end + 1]
    seg_d = dollars[start:end + 1]
    vol = float(seg_v.sum())
    dol = float(seg_d.sum())
    return {
        "open": float(prices[start]),
        "high": float(prices[start:end + 1].max()),
        "low": float(prices[start:end + 1].min()),
        "close": float(prices[end]),
        "volume": vol,
        "dollar": dol,
        "vwap": float(dol / vol) if vol > 0 else float("nan"),
        "n_ticks": int(end - start + 1),
        "start_time": times[start],
        "_close_time": times[end],
    }


def _threshold_bars(trades: pd.DataFrame, increments: np.ndarray,
                    threshold: float) -> pd.DataFrame:
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    prices = trades["price"].to_numpy(dtype=float)
    vols = trades["volume"].to_numpy(dtype=float)
    dollars = prices * vols
    times = trades.index.to_numpy()
    rows, idx = [], []
    csum = 0.0
    start = 0
    for i in range(len(trades)):
        csum += increments[i]
        if csum >= threshold:
            bar = _aggregate(prices, vols, dollars, times, start, i)
            idx.append(bar.pop("_close_time"))
            rows.append(bar)
            start = i + 1
            csum = 0.0
    return pd.DataFrame(rows, columns=_BAR_COLUMNS,
                        index=pd.DatetimeIndex(idx, name="close_time"))


def tick_bars(trades: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """一定の約定回数ごとにサンプリング。"""
    return _threshold_bars(trades, np.ones(len(trades)), threshold)


def volume_bars(trades: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """一定の出来高（base 数量）ごとにサンプリング。"""
    return _threshold_bars(trades, trades["volume"].to_numpy(dtype=float),
                           threshold)


def dollar_bars(trades: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """一定の取引代金（price×volume）ごとにサンプリング。標準推奨。"""
    inc = (trades["price"] * trades["volume"]).to_numpy(dtype=float)
    return _threshold_bars(trades, inc, threshold)


def apply_tick_rule(prices, init: int = 1) -> np.ndarray:
    """tick rule：価格変化の符号（+1/-1）。0 は直前の符号を引き継ぐ。"""
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    signs = np.empty(n)
    if n == 0:
        return signs
    last = init
    signs[0] = last
    for i in range(1, n):
        dp = prices[i] - prices[i - 1]
        if dp > 0:
            last = 1
        elif dp < 0:
            last = -1
        signs[i] = last
    return signs


def _signed_flow(trades: pd.DataFrame) -> np.ndarray:
    """各約定の符号（買い=+1/売り=-1）。side があれば使用、不明は tick rule で補完。"""
    if "side" in trades.columns and trades["side"].notna().any():
        side = trades["side"].astype(str).str.lower().to_numpy()
        b = np.where(side == "buy", 1.0, np.where(side == "sell", -1.0, 0.0))
        if (b == 0).any():  # 不明分を tick rule で補完
            tr = apply_tick_rule(trades["price"].to_numpy())
            b = np.where(b == 0, tr, b)
        return b
    return apply_tick_rule(trades["price"].to_numpy())


def dollar_imbalance_bars(trades: pd.DataFrame, threshold: float,
                          sign_method: str = "auto") -> pd.DataFrame:
    """ドル・インバランスバー（固定しきい値版）。

    符号付きドルフロー b_t·(price·volume) を累積し、|累積| が threshold に達したら
    バーを確定。インフォームド・トレーダーの非対称な執行（オーダーフロー不均衡）に
    同期してサンプリングする。AFML ch.2。

    sign_method : 'auto'（side優先, 無ければtick rule） | 'tick'（常にtick rule）
    注：AFML の EWMA 適応しきい値版は将来拡張。まず固定しきい値で堅実に。
    """
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    prices = trades["price"].to_numpy(dtype=float)
    vols = trades["volume"].to_numpy(dtype=float)
    dollars = prices * vols
    times = trades.index.to_numpy()
    b = apply_tick_rule(prices) if sign_method == "tick" else _signed_flow(trades)
    signed = b * dollars

    rows, idx = [], []
    theta = 0.0
    start = 0
    for i in range(len(trades)):
        theta += signed[i]
        if abs(theta) >= threshold:
            bar = _aggregate(prices, vols, dollars, times, start, i)
            bar["imbalance"] = float(theta)
            idx.append(bar.pop("_close_time"))
            rows.append(bar)
            start = i + 1
            theta = 0.0
    return pd.DataFrame(rows, columns=_BAR_COLUMNS + ["imbalance"],
                        index=pd.DatetimeIndex(idx, name="close_time"))
