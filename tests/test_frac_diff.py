"""分数階差分（FFD）の正しさとメモリ／定常性のトレードオフを検証。"""
import numpy as np
import pandas as pd

from invest_system.features.frac_diff import (
    find_min_d,
    frac_diff_ffd,
    get_weights_ffd,
)


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"),
                     name="x")


def test_weights_d0_is_identity():
    assert np.allclose(get_weights_ffd(0.0), [1.0])


def test_weights_d1_is_first_difference():
    assert np.allclose(get_weights_ffd(1.0), [-1.0, 1.0])


def test_ffd_d0_returns_original():
    s = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = frac_diff_ffd(s, 0.0)
    assert np.allclose(out.to_numpy(), s.to_numpy())


def test_ffd_d1_equals_first_difference():
    s = _series([1.0, 3.0, 6.0, 10.0, 15.0])
    out = frac_diff_ffd(s, 1.0)
    assert np.allclose(out.to_numpy(), s.diff().dropna().to_numpy())


def test_memory_decreases_with_d():
    rng = np.random.default_rng(5)
    rw = _series(np.cumsum(rng.normal(0, 1, 1000)))

    # thresh をやや緩めて窓幅を抑える（小さい d ほど FFD の窓が広がるため）
    def corr_at(d):
        f = frac_diff_ffd(rw, d, thresh=1e-3).dropna()
        return float(np.corrcoef(rw.loc[f.index].to_numpy(), f.to_numpy())[0, 1])

    assert corr_at(0.2) > corr_at(0.8)


def test_find_min_d_on_random_walk():
    rng = np.random.default_rng(3)
    rw = _series(np.cumsum(rng.normal(0, 1, 600)))
    min_d, corr, table = find_min_d(rw, d_grid=np.round(np.arange(0.0, 1.01, 0.1), 2))
    assert min_d is not None
    assert 0.0 < min_d <= 1.0
    assert not bool(table.loc[0.0, "stationary"])     # 原系列（ランダムウォーク）は非定常
    assert bool(table.loc[min_d, "stationary"])       # 最小 d で定常化
    assert corr > 0.4                                  # メモリが保たれている
