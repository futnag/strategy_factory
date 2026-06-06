"""検証ファクトリ Phase1：バックテスト・エンジンの検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.research.data_view import AsOfView
from invest_system.research.engine import backtest, _ann_factor
from invest_system.research.strategy import CrossSectionalStrategy, GapReversal


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


def test_ann_factor_inference():
    daily = pd.date_range("2020-01-01", periods=252 * 3, freq="B")
    assert 230 < _ann_factor(daily) < 270          # 日次 ≈ 252
    monthly = pd.date_range("2016-01-31", periods=120, freq="ME")
    assert 11.5 < _ann_factor(monthly) < 12.5       # 月次 ≈ 12
