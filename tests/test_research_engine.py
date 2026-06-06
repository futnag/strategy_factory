"""検証ファクトリ Phase1：バックテスト・エンジンの検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.research.data_view import AsOfView
from invest_system.research.engine import backtest, _ann_factor
from invest_system.research.strategy import (
    CrossSectionalStrategy, GapReversal, SignalTimingStrategy,
)


def test_gap_engine_realizes_next_period_return():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    op = pd.DataFrame({"A": [100., 100, 88, 99], "B": [100., 100, 100, 101]}, index=idx)
    cl = pd.DataFrame({"A": [100., 100, 90, 99], "B": [100., 100, 101, 101]}, index=idx)
    view = AsOfView({"open": op, "close": cl})
    res = backtest(GapReversal(threshold=0.10, hold=1), view, costs_bps=15.0)
    # d2 で A 建玉 → d2→d3 リターン 99/90-1=0.10、回転1で 15bps 控除
    assert res.gross.loc[idx[2]] == pytest.approx(0.10)
    assert res.returns.loc[idx[2]] == pytest.approx(0.10 - 0.0015)
    assert res.returns.loc[idx[0]] == 0.0 and res.returns.loc[idx[1]] == 0.0
    assert res.n_positions.loc[idx[2]] == 1


def test_cross_sectional_engine_long_short_return():
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    cl = pd.DataFrame({"A": [100., 95], "B": [100., 100], "C": [100., 100],
                       "D": [100., 100], "E": [100., 110]}, index=idx)
    factor = pd.DataFrame({"A": [1., 1], "B": [2., 2], "C": [3., 3],
                           "D": [4., 4], "E": [5., 5]}, index=idx)
    view = AsOfView({"close": cl})
    res = backtest(CrossSectionalStrategy(factor, quantile=0.2), view, costs_bps=0.0)
    # t0: ロングE(+10%) − ショートA(−5%) = 0.10 − (−0.05) = 0.15
    assert res.gross.loc[idx[0]] == pytest.approx(0.15)


def test_costs_reduce_returns():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    op = pd.DataFrame({"A": [100., 100, 88, 99]}, index=idx)
    cl = pd.DataFrame({"A": [100., 100, 90, 99]}, index=idx)
    view = AsOfView({"open": op, "close": cl})
    free = backtest(GapReversal(0.10), view, costs_bps=0.0).returns.loc[idx[2]]
    costed = backtest(GapReversal(0.10), view, costs_bps=50.0).returns.loc[idx[2]]
    assert costed < free


def test_execution_lag_shifts_realization():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    close = pd.DataFrame({"A": [100., 110, 99, 108.9]}, index=idx)  # +10/-10/+10%
    sig = pd.Series(1.0, index=idx)                 # 常時ロング
    strat = SignalTimingStrategy(sig, "A", threshold=0.0)
    view = AsOfView({"close": close})
    r0 = backtest(strat, view, costs_bps=0.0, execution_lag=0).returns
    assert r0.loc[idx[0]] == pytest.approx(0.10)    # 決定足終値で執行: t0→t1
    r1 = backtest(strat, view, costs_bps=0.0, execution_lag=1).returns
    assert r1.loc[idx[0]] == pytest.approx(-0.10)   # 翌足で執行: t1→t2
    assert idx[2] not in r1.index                   # 末尾2本は実現不可で除外


def _xs_setup():
    # CrossSectionalStrategy は最低5銘柄必要。6銘柄, q=0.2 → long F / short A。
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    cols = ["A", "B", "C", "D", "E", "F"]
    close = pd.DataFrame({c: [100., 101] for c in cols}, index=idx)
    factor = pd.DataFrame({c: [i + 1.0, i + 1.0] for i, c in enumerate(cols)},
                          index=idx)
    adv = pd.DataFrame({c: [2e8, 2e8] for c in cols}, index=idx)
    adv["A"] = [1e8, 1e8]                            # A（ショート側）が最も薄い
    return idx, close, factor, adv


def test_capacity_from_adv():
    idx, close, factor, adv = _xs_setup()
    res = backtest(CrossSectionalStrategy(factor, quantile=0.2),
                   AsOfView({"close": close}), costs_bps=0.0, adv=adv,
                   participation=0.1)
    # long F / short A, |w|=1ずつ → 容量 = min(0.1×1e8/1, 0.1×2e8/1) = 1e7（A律速）
    assert res.capacity_jpy == pytest.approx(1e7)


def test_capacity_nan_without_adv():
    idx, close, factor, adv = _xs_setup()
    res = backtest(CrossSectionalStrategy(factor, quantile=0.2),
                   AsOfView({"close": close}), costs_bps=0.0)
    assert np.isnan(res.capacity_jpy)


def test_ann_factor_inference():
    daily = pd.date_range("2020-01-01", periods=252 * 3, freq="B")
    assert 230 < _ann_factor(daily) < 270          # 日次 ≈ 252
    monthly = pd.date_range("2016-01-31", periods=120, freq="ME")
    assert 11.5 < _ann_factor(monthly) < 12.5       # 月次 ≈ 12
