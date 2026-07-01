"""合成日足（動作確認・テスト用）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_data(n_symbols: int = 3, n_days: int = 1500, seed: int = 42) -> dict[str, pd.DataFrame]:
    """トレンド + ノイズ + たまの急落を含む日足を銘柄別に生成。"""
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    for k in range(n_symbols):
        drift = rng.uniform(0.0002, 0.0006)
        vol = rng.uniform(0.015, 0.025)
        rets = rng.normal(drift, vol, n_days)
        shocks = rng.random(n_days) < 0.03
        rets[shocks] -= rng.uniform(0.03, 0.08, shocks.sum())
        close = 1000 * np.exp(np.cumsum(rets))
        intraday = np.abs(rng.normal(0, vol, n_days))
        high = close * (1 + intraday)
        low = close * (1 - intraday)
        open_ = np.empty(n_days)
        open_[0] = close[0]
        open_[1:] = close[:-1] * (1 + rng.normal(0, vol * 0.5, n_days - 1))
        open_ = np.clip(open_, low, high)
        vol_ = rng.integers(50_000, 500_000, n_days)
        out[f"SYNTH_{k + 1}"] = pd.DataFrame({
            "Date": dates, "Open": open_, "High": high,
            "Low": low, "Close": close, "Volume": vol_,
        })
    return out