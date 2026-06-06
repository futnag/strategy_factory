"""検証ファクトリ Phase1：データビュー＋戦略契約の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.research.data_view import AsOfView
from invest_system.research.strategy import (
    CalendarStrategy, CrossSectionalStrategy, GapReversal, PairsStrategy,
    SignalTimingStrategy,
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
