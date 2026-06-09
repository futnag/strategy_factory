"""柱D 時系列ツールキット（mean_reversion / cointegration / kalman）の検証。

KB §11 / DP12-14。ネットワーク不要・合成データ・固定乱数種で決定的。
"""
import numpy as np
import pandas as pd

from invest_system.timeseries import (
    KalmanHedge,
    adf_pvalue,
    cadf,
    half_life,
    hedge_ratio_ols,
    hedge_ratio_tls,
    hurst_exponent,
    johansen,
    ou_params,
    spread_series,
    variance_ratio,
)


def _ar1(n, phi, seed, sigma=1.0):
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0, sigma)
    return s


def _rw(n, seed, drift=0.0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(drift, 1.0, n))


# --- mean_reversion ---------------------------------------------------------
def test_half_life_meanreverting_vs_trend():
    hl = half_life(_ar1(3000, 0.95, 1))
    assert 8.0 < hl < 22.0                       # 理論 -ln2/ln(0.95)≈13.5
    assert np.isinf(half_life(np.arange(500.0)))  # 完全トレンド→回帰せず


def test_hurst_regimes():
    assert hurst_exponent(_ar1(3000, 0.9, 2)) < 0.45     # 平均回帰
    assert 0.4 < hurst_exponent(_rw(3000, 3)) < 0.6      # ランダムウォーク


def test_variance_ratio_regimes():
    assert variance_ratio(_ar1(4000, 0.9, 4))[16] < 0.9  # 平均回帰
    assert 0.8 < variance_ratio(_rw(4000, 5))[2] < 1.2   # RW≈1


def test_adf_pvalue_stationary_vs_unitroot():
    assert adf_pvalue(_ar1(2000, 0.9, 6)) < 0.05
    # ドリフト付き RW＝明確に非定常（トレンド）。定数項回帰の ADF は棄却しない。
    assert adf_pvalue(_rw(2000, 7, drift=0.3)) > 0.05


def test_ou_params_sign():
    theta, mu, sigma = ou_params(_ar1(3000, 0.92, 8))
    assert theta > 0 and np.isfinite(mu) and sigma > 0


# --- cointegration ----------------------------------------------------------
def _coint_pair(n, beta, seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n)) + 100.0
    y = beta * x + rng.normal(0, 1, n)
    return pd.Series(y), pd.Series(x)


def test_hedge_ratio_ols_tls():
    y, x = _coint_pair(3000, 2.0, 9)
    assert abs(hedge_ratio_ols(y, x) - 2.0) < 0.1
    assert abs(hedge_ratio_tls(y, x) - 2.0) < 0.1


def test_cadf_cointegrated_vs_independent():
    y, x = _coint_pair(3000, 2.0, 10)
    beta, p = cadf(y, x)
    assert abs(beta - 2.0) < 0.1 and p < 0.01
    xi, yi = pd.Series(_rw(3000, 11)), pd.Series(_rw(3000, 12))
    _, p_indep = cadf(yi, xi)
    assert p_indep > 0.05                        # 独立RWは共和分でない


def test_spread_series_value():
    y, x = pd.Series([10.0, 12.0, 14.0]), pd.Series([2.0, 3.0, 4.0])
    assert np.allclose(spread_series(y, x, 2.0).to_numpy(), [6.0, 6.0, 6.0])


def test_johansen_cointegrated_basket():
    rng = np.random.default_rng(13)
    n = 3000
    f = np.cumsum(rng.normal(0, 1, n))
    P = pd.DataFrame({"a": f + rng.normal(0, 1, n),
                      "b": 2 * f + rng.normal(0, 1, n),
                      "c": -f + rng.normal(0, 1, n)})
    jr = johansen(P)
    assert jr.n_relations >= 1
    assert jr.strongest.shape == (3,)


# --- kalman -----------------------------------------------------------------
def test_kalman_converges_to_constant_beta():
    rng = np.random.default_rng(14)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 2000)) + 50.0)
    y = 2.0 * x + rng.normal(0, 0.1, 2000)
    res = KalmanHedge().filter(x, y)
    assert abs(float(res["beta"].iloc[-1]) - 2.0) < 0.3


def test_kalman_is_lookahead_free():
    rng = np.random.default_rng(15)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 1200)) + 50.0)
    y = 1.5 * x + rng.normal(0, 0.2, 1200)
    full = KalmanHedge().filter(x, y)
    t = 900
    trunc = KalmanHedge().filter(x.iloc[:t + 1], y.iloc[:t + 1])
    # ≤t の出力は未来データに不変（再帰式＝オンライン＝先読み不能）
    assert np.abs(full["beta"].iloc[:t + 1].to_numpy()
                  - trunc["beta"].to_numpy()).max() < 1e-9
