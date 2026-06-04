"""purged_cv_predict（リーク無し OOS 予測）の検証。"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from invest_system.backtest.cv_score import purged_cv_predict


def test_oos_predictions_cover_all_and_recover_learnable_target():
    rng = np.random.default_rng(0)
    n = 150
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x = rng.normal(0, 1, n)
    y = 2.0 * x + 0.01 * rng.normal(0, 1, n)        # 学習可能な線形関係
    X = pd.DataFrame({"x": x}, index=idx)
    t1 = pd.Series(idx, index=idx)                  # 瞬間ラベル
    oos = purged_cv_predict(X, y, t1, LinearRegression)
    assert not oos.isna().any()                     # 全観測が OOS 予測される
    assert r2_score(y, oos.to_numpy()) > 0.95       # 学習可能 → 高 R²


def test_noise_target_gives_low_r2():
    rng = np.random.default_rng(1)
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    X = pd.DataFrame({"x": rng.normal(0, 1, n)}, index=idx)
    y = rng.normal(0, 1, n)                          # x と無関係
    t1 = pd.Series(idx, index=idx)
    oos = purged_cv_predict(X, y, t1, LinearRegression)
    assert r2_score(y, oos.to_numpy()) < 0.2        # 予測不能 → 低 R²


def test_index_mismatch_raises():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    X = pd.DataFrame({"x": range(10)}, index=idx)
    bad_t1 = pd.Series(idx, index=pd.date_range("2021-01-01", periods=10, freq="D"))
    with pytest.raises(ValueError):
        purged_cv_predict(X, np.arange(10), bad_t1, LinearRegression)
