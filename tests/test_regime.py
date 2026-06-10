"""Regime-Switching の検証（柱D・KB §11）：

- timeseries.regime のラベラ（Efficiency Ratio・拡張窓三分位）の値域と **先読み不変**。
- research.RegimeGated のハードゲート/サイズ調整/レジーム未確定 flat。
- research.regime_breakdown のレジーム別分解。
ネットワーク不要・合成データ・固定乱数種で決定的。
"""
import numpy as np
import pandas as pd

from invest_system.research import (
    AsOf, RegimeGated, RegimeSwitch, Strategy, regime_breakdown,
    walk_forward_regime_assignment,
)
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


def test_regime_switch_routes_by_regime():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    regime = pd.Series([0, 0, 2, 2, 1, 2], index=idx, dtype="float64")
    a = _Const(pd.Series({"A": 1.0}))
    b = _Const(pd.Series({"B": -1.0}))
    sw = RegimeSwitch(regime, {0: a, 2: b})                      # regime1 は未マップ
    assert sw.target_weights(_asof(idx[0]))["A"] == 1.0          # regime0→a
    assert sw.target_weights(_asof(idx[2]))["B"] == -1.0         # regime2→b
    assert sw.target_weights(_asof(idx[4])).empty               # regime1 未マップ→flat
    assert sw.params["switch"] == {0: "const", 2: "const"}


def test_regime_switch_unknown_is_flat():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    regime = pd.Series([np.nan, 0, 0, 0], index=idx, dtype="float64")
    sw = RegimeSwitch(regime, {0: _Const(pd.Series({"A": 1.0}))})
    assert sw.target_weights(_asof(idx[0])).empty               # ≤t NaN→flat
    assert not sw.target_weights(_asof(idx[2])).empty


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


# --- walk_forward_regime_assignment -----------------------------------------
def test_walk_forward_assignment_picks_past_best_and_is_pit():
    idx = pd.date_range("2020-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)
    regime = pd.Series(np.tile([0.0, 1.0], 30), index=idx)       # 交互に 0,1
    a = pd.Series(np.where(regime.values == 0, 0.02, -0.02)      # A: r0で+ r1で−
                  + rng.normal(0, 1e-4, 60), index=idx)
    b = pd.Series(np.where(regime.values == 1, 0.02, -0.02)      # B: 逆
                  + rng.normal(0, 1e-4, 60), index=idx)
    asg = walk_forward_regime_assignment({"A": a, "B": b}, regime, min_obs=4, warmup=10)
    late, r_late = asg.iloc[10:], regime.iloc[10:]
    assert (late[r_late == 0] == "A").all()                      # 過去ベスト＝r0→A
    assert (late[r_late == 1] == "B").all()                      # r1→B
    assert asg.iloc[:10].isna().all()                            # warmup は現金
    a2 = a.copy(); a2.iloc[41:] *= -5.0                          # 未来(s>40)を改変
    asg2 = walk_forward_regime_assignment({"A": a2, "B": b}, regime, min_obs=4, warmup=10)
    pd.testing.assert_series_equal(asg.iloc[:41], asg2.iloc[:41])  # ≤40 の割当は不変=PIT


def test_walk_forward_cash_when_all_nonpositive():
    idx = pd.date_range("2020-01-31", periods=40, freq="ME")
    regime = pd.Series(np.zeros(40), index=idx)                  # 単一レジーム
    a = pd.Series(-0.01, index=idx)                             # 常に負
    asg = walk_forward_regime_assignment({"A": a}, regime, min_obs=4, warmup=8)
    assert asg.iloc[8:].isna().all()                            # 正の sleeve 無し→現金
