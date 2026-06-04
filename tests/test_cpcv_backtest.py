"""CPCV バックテストのパス復元・被覆・Sharpe を検証（スタブモデルで決定的に）。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.backtest.cpcv_backtest import cpcv_backtest
from invest_system.validation.cpcv import CombinatorialPurgedKFold


class _Const:
    """常に一定値を予測。"""

    def __init__(self, v=1.0):
        self.v = v

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.v)


class _Oracle:
    """特徴量 'sig'（=ラベル）をそのまま予測する完全予測器。"""

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return X["sig"].to_numpy()


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    y = rng.choice([-1.0, 1.0], size=n)
    ret = pd.Series(y * rng.uniform(0.001, 0.02, n), index=idx)   # sign(ret)=y
    X = pd.DataFrame({"sig": y, "f": rng.normal(0, 1, n)}, index=idx)
    t1 = pd.Series(idx, index=idx)
    return X, pd.Series(y, index=idx), ret, t1


def test_n_paths_and_full_coverage():
    X, y, ret, t1 = _data()
    cv = CombinatorialPurgedKFold(6, 2, 0.0)
    res = cpcv_backtest(X, y, ret, t1, cv, lambda: _Const(1.0))
    assert res.n_paths == cv.get_n_paths() == 5
    assert res.paths.shape == (5, len(X))
    assert not np.isnan(res.paths).any()        # 各パスが全観測を一度ずつ被覆


def test_oracle_all_paths_positive():
    X, y, ret, t1 = _data()
    cv = CombinatorialPurgedKFold(6, 2, 0.0)
    res = cpcv_backtest(X, y, ret, t1, cv, lambda: _Oracle())
    assert len(res.path_sharpes) == 5
    assert (res.path_sharpes > 0).all()         # 完全予測 → 全パス正のSharpe
    assert res.mean_sharpe > 0
    assert res.frac_negative == 0.0


def test_index_mismatch_raises():
    X, y, ret, t1 = _data()
    bad_t1 = pd.Series(
        t1.to_numpy(),
        index=pd.date_range("2031-01-01", periods=len(t1), freq="D"))
    cv = CombinatorialPurgedKFold(6, 2, 0.0)
    with pytest.raises(ValueError):
        cpcv_backtest(X, y, ret, bad_t1, cv, lambda: _Const(1.0))


def test_sample_weight_threads_through():
    X, y, ret, t1 = _data()
    w = pd.Series(np.ones(len(X)), index=X.index)
    cv = CombinatorialPurgedKFold(6, 2, 0.0)
    res = cpcv_backtest(X, y, ret, t1, cv, lambda: _Const(1.0), sample_weight=w)
    assert res.n_paths == 5
    assert not np.isnan(res.paths).any()
