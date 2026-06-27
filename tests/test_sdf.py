"""条件付き SDF コア（候補② Stage1）の純関数を検証。"""
import numpy as np
import pandas as pd

from invest_system.portfolio.sdf import (
    augment_factors,
    managed_portfolios,
    ridge_tangency,
    walk_forward_weights,
)


def test_managed_portfolios_sign():
    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    codes = list("ABCD")
    rng = np.random.default_rng(0)
    z = pd.DataFrame(rng.standard_normal((8, 4)), index=dates, columns=codes)
    fwd = z * 0.1 + 0.001                         # fwd は z に正比例
    F = managed_portfolios({"f": z}, fwd)
    assert list(F.columns) == ["f"]
    assert (F["f"] > 0).mean() >= 0.8             # 予測的 z → 正のファクターリターン


def test_ridge_tangency_basic():
    rng = np.random.default_rng(1)
    F = pd.DataFrame(rng.standard_normal((60, 3)) * 0.02 + 0.01,
                     index=pd.date_range("2020-01-31", periods=60, freq="ME"),
                     columns=list("abc"))
    b = ridge_tangency(F)
    assert b is not None and len(b) == 3 and np.all(np.isfinite(b))
    assert ridge_tangency(F.iloc[:3]) is None     # 行不足は None


def test_augment_factors_shapes():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    F = pd.DataFrame({"x": range(5), "y": range(5)}, index=dates, dtype=float)
    S = pd.DataFrame({"s1": range(5), "s2": range(5)}, index=dates, dtype=float)
    Faug, terms = augment_factors(F, S)
    assert Faug.shape[1] == 2 + 2 * 2             # K + K*M
    assert ("x", None) in terms and ("y", "s2") in terms
    Fu, tu = augment_factors(F, None)
    assert Fu.shape[1] == 2 and tu == [("x", None), ("y", None)]


def test_walk_forward_weights_dollar_neutral():
    dates = pd.date_range("2019-01-31", periods=60, freq="ME")
    codes = [f"S{i}" for i in range(10)]
    rng = np.random.default_rng(2)
    z = pd.DataFrame(rng.standard_normal((60, 10)), index=dates, columns=codes)
    fwd = z * 0.05 + rng.standard_normal((60, 10)) * 0.01      # z が次期を予測
    zdict = {"f": z}
    F = managed_portfolios(zdict, fwd)
    W = walk_forward_weights(F, zdict, min_train=36)
    assert not W.empty
    s = W.dropna(how="all").sum(axis=1).abs()
    assert (s < 1e-9).mean() > 0.9                # 各行ダラーニュートラル
    g = W.dropna(how="all").abs().sum(axis=1)
    assert (np.abs(g - 1.0) < 1e-9).mean() > 0.9  # グロス1
