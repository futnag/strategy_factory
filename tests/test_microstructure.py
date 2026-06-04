"""マイクロ構造・流動性特徴の検証。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.features.microstructure import (
    amihud_illiquidity,
    corwin_schultz_spread,
    garman_klass_vol,
    parkinson_vol,
    roll_spread,
    rsi,
    vpin,
)


def _ohlcv(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.abs(rng.normal(10, 3, n)) + 1.0, index=idx)
    return open_, high, low, close, volume


def test_parkinson_positive():
    _, h, l, _, _ = _ohlcv()
    assert (parkinson_vol(h, l, 20).dropna() > 0).all()


def test_garman_klass_nonnegative():
    o, h, l, c, _ = _ohlcv()
    assert (garman_klass_vol(o, h, l, c, 20).dropna() >= 0).all()


def test_amihud_window1_matches_definition():
    _, _, _, c, v = _ohlcv()
    dv = c * v
    illiq = amihud_illiquidity(c, dv, window=1).dropna()
    expected = (c.pct_change(fill_method=None).abs() / dv).reindex(illiq.index)
    assert np.allclose(illiq.to_numpy(), expected.to_numpy())


def test_roll_spread_nonnegative():
    _, _, _, c, _ = _ohlcv()
    assert (roll_spread(c, 20).dropna() >= 0).all()


def test_corwin_schultz_nonnegative():
    _, h, l, _, _ = _ohlcv()
    assert (corwin_schultz_spread(h, l).dropna() >= 0).all()


def test_vpin_bounded_and_high_on_uptrend():
    _, _, _, c, v = _ohlcv()
    vp = vpin(c, v, 30).dropna()
    assert ((vp >= 0) & (vp <= 1)).all()
    # 単調上昇 → 買い圧一辺倒 → VPIN ≈ 1
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    up = pd.Series(np.linspace(100, 200, 200), index=idx)
    one = pd.Series(np.ones(200), index=idx)
    assert vpin(up, one, 30).dropna().mean() > 0.9


def test_rsi_bounded_and_high_on_uptrend():
    _, _, _, c, _ = _ohlcv()
    r = rsi(c, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    idx = pd.date_range("2020-01-01", periods=100, freq="h")
    up = pd.Series(np.linspace(100, 200, 100), index=idx)
    assert rsi(up, 14).dropna().iloc[-1] > 99
