"""TSMOM（時系列モメンタム）部品の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.research.strategies_tsmom import (
    annualized_vol, blend_weights, tsmom_weights,
)


def _monthly():
    idx = pd.date_range("2020-01-31", periods=15, freq="ME")
    up = np.linspace(100, 160, 15)            # 一貫した上昇
    dn = np.linspace(100, 70, 15)             # 一貫した下落
    return pd.DataFrame({"UP": up, "DN": dn}, index=idx)


def test_tsmom_sign_and_vol_scaling():
    m = _monthly()
    vol = pd.DataFrame(0.20, index=m.index, columns=m.columns)
    vol["DN"] = 0.40                           # DN はボラ2倍 → ウェイト半分
    w = tsmom_weights(m, vol, lookback=12, vol_target=0.10)
    t = m.index[-1]
    assert w[t]["UP"] == pytest.approx(+0.10 / 0.20 / 2)   # +0.25
    assert w[t]["DN"] == pytest.approx(-0.10 / 0.40 / 2)   # -0.125（半分）
    # 履歴不足（lookback 未満）の月はウェイト無し
    assert m.index[5] not in w


def test_tsmom_flat_and_nan_drop():
    idx = pd.date_range("2020-01-31", periods=15, freq="ME")
    m = pd.DataFrame({"FLAT": 100.0, "OK": np.linspace(100, 130, 15)}, index=idx)
    m.loc[idx[-1], "OK"] = np.nan              # 最終月に欠損
    vol = pd.DataFrame(0.20, index=idx, columns=m.columns)
    w = tsmom_weights(m, vol, lookback=12, vol_target=0.10)
    t = idx[-2]
    assert "FLAT" not in w[t].index            # sign==0 は無ポジ
    assert w[t]["OK"] == pytest.approx(0.10 / 0.20)        # N=1（有効1資産）
    assert idx[-1] not in w                    # 全資産無効の日はエントリ無し


def test_tsmom_pit_future_change_does_not_affect_past():
    m = _monthly()
    vol = pd.DataFrame(0.20, index=m.index, columns=m.columns)
    t = m.index[12]
    w1 = tsmom_weights(m, vol, lookback=12)[t]
    m2 = m.copy()
    m2.iloc[-1] = 9999.0                       # 未来の改変
    w2 = tsmom_weights(m2, vol, lookback=12)[t]
    pd.testing.assert_series_equal(w1, w2)


def test_annualized_vol_floor_and_window():
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    rng = np.random.default_rng(0)
    quiet = 100 * np.exp(np.cumsum(rng.normal(0, 0.0001, 80)))   # 極小ボラ
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))
    v = annualized_vol(pd.DataFrame({"Q": quiet, "W": wild}, index=idx),
                       window=63, floor=0.05)
    assert v["Q"].dropna().iloc[-1] == pytest.approx(0.05)       # floor 適用
    assert v["W"].dropna().iloc[-1] > 0.20


def test_blend_weights_average():
    t = pd.Timestamp("2024-01-31")
    a = {t: pd.Series({"X": 0.4})}
    b = {t: pd.Series({"X": -0.2, "Y": 0.2})}
    w = blend_weights([a, b])[t]
    assert w["X"] == pytest.approx(0.1)        # (0.4 - 0.2) / 2
    assert w["Y"] == pytest.approx(0.1)
