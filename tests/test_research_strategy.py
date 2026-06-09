"""検証ファクトリ Phase1：データビュー＋戦略契約の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.research.data_view import AsOfView
from invest_system.research.strategy import (
    CalendarStrategy, CompositeStrategy, CrossSectionalStrategy, EarningsRunup,
    GapReversal, PairsStrategy, SignalTimingStrategy,
)
from invest_system.research.strategies_meanrev import (
    CointegratedPairs, JohansenBasket, LinearMeanReversion,
)


def _ohlc():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    op = pd.DataFrame({"A": [100., 100, 88], "B": [100., 100, 100]}, index=idx)
    cl = pd.DataFrame({"A": [100., 100, 90], "B": [100., 100, 101]}, index=idx)
    return idx, {"open": op, "close": cl}


# --- AsOfView：未来不可視（リーク遮断）-------------------------------------
def test_asofview_hides_future():
    idx, panels = _ohlc()
    view = AsOfView(panels)
    a = view.asof(idx[1])                       # 2日目時点
    assert a.frame("close").index.max() == idx[1]   # 未来(idx[2])は含まない
    assert a.n_bars() == 2
    assert a.last("close")["A"] == 100.0
    assert a.lag("close", 1)["A"] == 100.0      # 前日


def test_asofview_requires_close():
    with pytest.raises(ValueError):
        AsOfView({"open": pd.DataFrame()})


def test_asof_unknown_field_raises():
    idx, panels = _ohlc()
    a = AsOfView(panels).asof(idx[2])
    with pytest.raises(KeyError):
        a.frame("volume")


# --- GapReversal -----------------------------------------------------------
def test_gap_reversal_triggers_on_gap_down():
    idx, panels = _ohlc()
    a = AsOfView(panels).asof(idx[2])           # 3日目：A は -12% ギャップダウン
    w = GapReversal(threshold=0.10, hold=1, side=1).target_weights(a)
    assert list(w.index) == ["A"]               # A のみ建玉
    assert w["A"] == pytest.approx(1.0)
    assert "B" not in w.index                   # ギャップ無しは除外


def test_gap_reversal_no_trigger_returns_empty():
    idx, panels = _ohlc()
    a = AsOfView(panels).asof(idx[1])           # 2日目：まだギャップ無し
    w = GapReversal(threshold=0.10).target_weights(a)
    assert w.empty


def test_gap_reversal_short_side():
    idx, panels = _ohlc()
    a = AsOfView(panels).asof(idx[2])
    w = GapReversal(threshold=0.10, side=-1).target_weights(a)
    assert w["A"] == pytest.approx(-1.0)


# --- SignalTimingStrategy --------------------------------------------------
def test_signal_timing_long_when_positive_else_flat():
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    close = pd.DataFrame({"0000": [100., 101, 102]}, index=idx)
    signal = pd.Series([0.5, -0.2, 0.3], index=idx)     # index=利用可能日
    view = AsOfView({"close": close})
    strat = SignalTimingStrategy(signal, code="0000", threshold=0.0, side=1)
    assert strat.target_weights(view.asof(idx[0]))["0000"] == 1.0   # 0.5>0 → ロング
    assert strat.target_weights(view.asof(idx[1])).empty            # 直近-0.2 → 現金


def test_signal_timing_no_future_signal():
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    close = pd.DataFrame({"0000": [100., 101]}, index=idx)
    # シグナルは将来日付のみ → asof 時点では参照不可 → 現金
    signal = pd.Series([0.9], index=[idx[1]])
    strat = SignalTimingStrategy(signal, code="0000")
    assert strat.target_weights(AsOfView({"close": close}).asof(idx[0])).empty


# --- CalendarStrategy / PairsStrategy --------------------------------------
def test_calendar_strategy_turn_of_month():
    idx = pd.date_range("2024-01-01", periods=31, freq="D")
    view = AsOfView({"close": pd.DataFrame({"X": [100.] * 31}, index=idx)})
    s = CalendarStrategy("X", dom_start=28, dom_end=2, side=1)
    assert s.target_weights(view.asof(pd.Timestamp("2024-01-28")))["X"] == 1.0
    assert s.target_weights(view.asof(pd.Timestamp("2024-01-02")))["X"] == 1.0
    assert s.target_weights(view.asof(pd.Timestamp("2024-01-15"))).empty


def test_earnings_runup_long_window_short_rest():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    close = pd.DataFrame({"A": [100.] * 2, "B": [100.] * 2, "C": [100.] * 2}, index=idx)
    days = pd.DataFrame({"A": [10., 10], "B": [50., 50], "C": [1., 1]}, index=idx)
    view = AsOfView({"close": close})
    w = EarningsRunup(days, pre=20, lag=2).target_weights(view.asof(idx[0]))
    assert w["A"] == pytest.approx(1.0)         # 2<10<=20 → run-up窓ロング
    assert w["B"] == pytest.approx(-0.5)        # 50>20 → 窓外ショート
    assert w["C"] == pytest.approx(-0.5)        # 1<=2 → 窓外ショート(直前)


def test_pairs_strategy_mean_reversion():
    idx = pd.date_range("2024-01-01", periods=70, freq="D")
    rng = np.random.default_rng(0)
    b = 100 * np.cumprod(1 + rng.normal(0, 0.01, 70))
    a = b.copy()
    a[-1] = a[-1] * 1.2                              # A が最後に急騰→割高
    view = AsOfView({"close": pd.DataFrame({"A": a, "B": b}, index=idx)})
    w = PairsStrategy("A", "B", lookback=60, entry=1.5).target_weights(view.asof(idx[-1]))
    assert w["A"] == pytest.approx(-0.5)            # A売り
    assert w["B"] == pytest.approx(0.5)             # B買い


# --- CrossSectionalStrategy ------------------------------------------------
def test_cross_sectional_long_short():
    idx = pd.date_range("2024-01-31", periods=1, freq="ME")
    close = pd.DataFrame({c: [1.0] for c in ["A", "B", "C", "D", "E"]}, index=idx)
    factor = pd.DataFrame({"A": [1.], "B": [2], "C": [3], "D": [4], "E": [5]}, index=idx)
    a = AsOfView({"close": close}).asof(idx[0])
    w = CrossSectionalStrategy(factor, quantile=0.2).target_weights(a)
    assert w["E"] == pytest.approx(1.0)         # 上位ロング
    assert w["A"] == pytest.approx(-1.0)        # 下位ショート
    assert set(w.index) == {"A", "E"}


def test_cross_sectional_long_only_market_hedged():
    idx = pd.date_range("2024-01-31", periods=1, freq="ME")
    close = pd.DataFrame({c: [1.0] for c in ["A", "B", "C", "D", "E"]}, index=idx)
    factor = pd.DataFrame({"A": [1.], "B": [2], "C": [3], "D": [4], "E": [5]}, index=idx)
    a = AsOfView({"close": close}).asof(idx[0])
    w = CrossSectionalStrategy(factor, 0.2, long_only=True).target_weights(a)
    assert w["E"] == pytest.approx(0.8)         # +1/1 − 1/5（上位ロング−市場ショート）
    assert w["A"] == pytest.approx(-0.2)        # −1/5（市場ショートのみ）
    assert abs(w.sum()) < 1e-9                  # ダラーニュートラル


def test_composite_strategy_sums_weights():
    idx = pd.date_range("2024-01-31", periods=1, freq="ME")
    close = pd.DataFrame({c: [1.0] for c in ["A", "B", "C", "D", "E"]}, index=idx)
    f1 = pd.DataFrame({"A": [1.], "B": [2], "C": [3], "D": [4], "E": [5]}, index=idx)
    a = AsOfView({"close": close}).asof(idx[0])
    s1 = CrossSectionalStrategy(f1, 0.2, name="s1")
    combo = CompositeStrategy([s1, s1], [0.5, 0.5])     # 同一戦略×2×0.5 = s1
    w = combo.target_weights(a)
    assert w["E"] == pytest.approx(1.0) and w["A"] == pytest.approx(-1.0)


# --- 柱D: 時系列・統計的裁定（CointegratedPairs / Johansen / LinearMeanReversion） ---
def _coint_close(n=130, beta=2.0, seed=0, bump=3.0, bump_at=-1):
    """a≈beta·b の共和分ペア。bump_at で a を bump 押し上げ（割高→z>0）。"""
    rng = np.random.default_rng(seed)
    b = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    a = beta * b + rng.normal(0, 1.0, n)
    a[bump_at] = a[bump_at] + bump
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return idx, pd.DataFrame({"A": a, "B": b}, index=idx)


def test_coint_pairs_builds_beta_hedged_position():
    idx, close = _coint_close()
    s = CointegratedPairs("A", "B", lookback=120, entry=1.5)
    w = s.target_weights(AsOfView({"close": close}).asof(idx[-1]))
    assert w["A"] < 0 and w["B"] > 0              # 割高Aを売り・Bを買い
    assert abs(w.abs().sum() - 1.0) < 1e-9        # グロス1
    assert abs(w["A"] + 0.5) < 0.1                # a≈2b → ほぼ ±0.5（β加重）
    assert abs(w.sum()) < 0.1                     # a≈β·b ＝ ほぼダラー中立


def test_coint_pairs_gate_blocks_non_cointegrated():
    rng = np.random.default_rng(1)
    n = 130
    a = 100 + np.cumsum(rng.normal(0, 1, n))      # 独立RW
    b = 100 + np.cumsum(rng.normal(0, 1, n))      # 独立RW（共和分でない）
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    view = AsOfView({"close": pd.DataFrame({"A": a, "B": b}, index=idx)})
    w = CointegratedPairs("A", "B", lookback=120,
                          coint_gate=True).target_weights(view.asof(idx[-1]))
    assert w.empty                                # ゲートで建玉せず


def test_coint_pairs_no_lookahead():
    idx, close = _coint_close(seed=3, bump=4.0, bump_at=-5)  # idx[-5] で建玉発生
    s = CointegratedPairs("A", "B", lookback=120, entry=1.5)
    w_t = s.target_weights(AsOfView({"close": close}).asof(idx[-5]))
    future = close.copy()
    future.iloc[-4:] *= 1.5                        # idx[-5] より後を改変
    w_t2 = s.target_weights(AsOfView({"close": future}).asof(idx[-5]))
    assert not w_t.empty                           # idx[-5] で建玉している
    pd.testing.assert_series_equal(w_t, w_t2)      # ≤t は未来改変に不変


def test_coint_pairs_kalman_method_also_trades():
    idx, close = _coint_close(seed=2)
    w = CointegratedPairs("A", "B", lookback=120, entry=1.5,
                          method="kalman").target_weights(
        AsOfView({"close": close}).asof(idx[-1]))
    assert not w.empty and w["A"] < 0 < w["B"]


def test_linear_mean_reversion_sign():
    idx = pd.date_range("2024-01-01", periods=70, freq="D")
    base = np.full(70, 100.0)
    base[-1] = 110.0                               # 直近だけ割高 → z>0 → ショート
    view = AsOfView({"close": pd.DataFrame({"X": base}, index=idx)})
    w = LinearMeanReversion("X", lookback=60, scale=2.0).target_weights(
        view.asof(idx[-1]))
    assert w["X"] < 0                              # 割高→ショート（在庫∝ −z）


def test_johansen_basket_gross_one_or_empty():
    rng = np.random.default_rng(5)
    n = 200
    f = np.cumsum(rng.normal(0, 1, n))
    close = pd.DataFrame({"A": f + rng.normal(0, .5, n) + 50,
                          "B": f + rng.normal(0, .5, n) + 30,
                          "C": f + rng.normal(0, .5, n) + 10},
                         index=pd.date_range("2024-01-01", periods=n, freq="D"))
    close.iloc[-1, 0] += 5.0                        # 末尾で乖離
    w = JohansenBasket(["A", "B", "C"], lookback=150, entry=1.0).target_weights(
        AsOfView({"close": close}).asof(close.index[-1]))
    if not w.empty:
        assert abs(w.abs().sum() - 1.0) < 1e-9 and set(w.index) <= {"A", "B", "C"}


def test_johansen_basket_needs_two_codes():
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    close = pd.DataFrame({"A": np.arange(200.0)}, index=idx)
    w = JohansenBasket(["A", "Z"], lookback=150).target_weights(
        AsOfView({"close": close}).asof(idx[-1]))
    assert w.empty                                 # 有効銘柄<2 → 空
