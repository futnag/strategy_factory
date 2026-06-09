"""Regime-Switching の検証（柱D・KB §11）：

- timeseries.regime のラベラ（Efficiency Ratio・拡張窓三分位）の値域と **先読み不変**。
- research.RegimeGated のハードゲート/サイズ調整/レジーム未確定 flat。
- research.regime_breakdown のレジーム別分解。
ネットワーク不要・合成データ・固定乱数種で決定的。
"""
import numpy as np
import pandas as pd

from invest_system.research import AsOf, RegimeGated, Strategy, regime_breakdown
from invest_system.timeseries.regime import (
    efficiency_ratio,
    equal_weight_market,
    expanding_tertile,
    trend_regime,
    vol_regime,
)


# --- ラベラ -----------------------------------------------------------------
def test_efficiency_ratio_trend_vs_range():
    trend = pd.Series(np.arange(300, dtype=float) + 100.0)        # 純トレンド
    assert abs(float(efficiency_ratio(trend, window=60).iloc[-1]) - 1.0) < 1e-9
    osc = pd.Series(100.0 + np.tile([0.0, 1.0], 150))             # 振動＝レンジ
    assert float(efficiency_ratio(osc, window=60).iloc[-1]) < 0.1


def test_expanding_tertile_labels_and_pit():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 600))
    lab = expanding_tertile(s, min_periods=100)
    assert lab.dropna().isin([0.0, 1.0, 2.0]).all()
    inc = pd.Series(np.arange(400, dtype=float))                  # 単調増→末尾は最上位
    assert float(expanding_tertile(inc, min_periods=100).iloc[-1]) == 2.0
    k = 300                                                       # PIT：未来改変→≤k 不変
    s2 = s.copy(); s2.iloc[k + 1:] += 50.0
    pd.testing.assert_series_equal(
        expanding_tertile(s, min_periods=100).iloc[:k + 1],
        expanding_tertile(s2, min_periods=100).iloc[:k + 1])


def test_trend_and_vol_regime_pit_and_labels():
    rng = np.random.default_rng(1)
    px = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (500, 5)), axis=0)
    close = pd.DataFrame(px, columns=[f"S{i}" for i in range(5)],
                         index=pd.date_range("2021-01-04", periods=500, freq="B"))
    tr = trend_regime(close, window=60, min_periods=120)
    vr = vol_regime(close, window=60, min_periods=120)
    assert tr.dropna().isin([0.0, 1.0, 2.0]).all()
    assert vr.dropna().isin([0.0, 1.0, 2.0]).all()
    assert equal_weight_market(close).iloc[-1] > 0                # 水準は正
    k = 300                                                       # PIT：未来改変→≤k 不変
    close2 = close.copy(); close2.iloc[k + 1:] *= 1.5
    pd.testing.assert_series_equal(
        trend_regime(close2, 60, 120).iloc[:k + 1], tr.iloc[:k + 1])
    pd.testing.assert_series_equal(
        vol_regime(close2, 60, 120).iloc[:k + 1], vr.iloc[:k + 1])


# --- RegimeGated ------------------------------------------------------------
class _Const(Strategy):
    """テスト用：asof を無視して固定ウェイトを返す base 戦略。"""

    def __init__(self, w):
        self._w = w
        self.name = "const"
        self.params = {"k": "v"}

    def target_weights(self, asof):
        return self._w.copy()


def _asof(date):
    d = pd.Timestamp(date)
    return AsOf({"close": pd.DataFrame({"A": [1.0]}, index=[d])}, d)


def test_regime_gated_hard_gate():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    regime = pd.Series([0, 0, 1, 1, 2, 2, 0, 1, 2, 0], index=idx, dtype="float64")
    base = _Const(pd.Series({"A": 0.5, "B": -0.5}))
    g = RegimeGated(base, regime, allowed={0})
    assert g.target_weights(_asof(idx[0]))["A"] == 0.5            # レジーム0＝建玉
    assert g.target_weights(_asof(idx[4])).empty                 # レジーム2＝flat
    assert g.params["regime"].startswith("allow")                # base と別試行タグ
    assert g.params["k"] == "v"                                  # base params 継承


def test_regime_gated_sizing():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    regime = pd.Series([0, 1, 2, 0, 1, 2], index=idx, dtype="float64")
    g = RegimeGated(_Const(pd.Series({"A": 1.0})), regime,
                    sizing={0: 1.0, 1: 0.5, 2: 0.0})
    assert g.target_weights(_asof(idx[0]))["A"] == 1.0
    assert abs(g.target_weights(_asof(idx[1]))["A"] - 0.5) < 1e-12
    assert g.target_weights(_asof(idx[2])).empty                 # 係数0＝flat


def test_regime_gated_unknown_and_empty_base():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    regime = pd.Series([np.nan, np.nan, 0, 0, 0, 0], index=idx, dtype="float64")
    g = RegimeGated(_Const(pd.Series({"A": 1.0})), regime, allowed={0})
    assert g.target_weights(_asof(idx[0])).empty                 # ≤t が NaN＝flat
    assert not g.target_weights(_asof(idx[3])).empty
    g2 = RegimeGated(_Const(pd.Series(dtype="float64")), regime, allowed={0})
    assert g2.target_weights(_asof(idx[3])).empty                # base 空はそのまま


# --- regime_breakdown -------------------------------------------------------
def test_regime_breakdown_separates():
    idx = pd.date_range("2022-01-03", periods=200, freq="B")
    rng = np.random.default_rng(3)
    regime = pd.Series(rng.integers(0, 3, 200), index=idx, dtype="float64")
    ret = pd.Series(np.where(regime.values == 0, 0.01, 0.0)
                    + rng.normal(0, 0.001, 200), index=idx)      # 0だけ正
    bd = regime_breakdown(ret, regime, ann=252.0)
    assert set(bd["regime"]) == {0.0, 1.0, 2.0}
    s0 = float(bd[bd["regime"] == 0.0]["sharpe_ann"].iloc[0])
    s2 = float(bd[bd["regime"] == 2.0]["sharpe_ann"].iloc[0])
    assert s0 > s2                                               # レジーム0が突出
