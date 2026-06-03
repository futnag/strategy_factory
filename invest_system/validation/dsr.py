"""Deflated Sharpe Ratio と多重検定統計量。

統合ナレッジベース §5.3-5.4 / DP6-DP7 の実装。
References: Bailey & López de Prado (2012, 2014); López de Prado & Lewis (2018).

すべての Sharpe 量は「リターンと同じ頻度（per-period, 非年率）」で扱う。
年率 SR が必要な場合のみ別途 sqrt(freq) で換算し、頻度を混在させないこと。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, skew as _skew, kurtosis as _kurtosis

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns) -> float:
    """per-period Sharpe ratio（mean / 標本std, ddof=1）。非年率。"""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        raise ValueError("need >= 2 returns")
    sd = r.std(ddof=1)
    if sd == 0:
        raise ValueError("zero volatility")
    return float(r.mean() / sd)


def _moments(returns):
    r = np.asarray(returns, dtype=float)
    sr = sharpe_ratio(r)
    sk = float(_skew(r, bias=False))
    ku = float(_kurtosis(r, fisher=False, bias=False))  # 非超過尖度（正規=3）
    return sr, sk, ku, int(r.size)


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int,
                               skew: float, kurt: float) -> float:
    """PSR: P(真の SR > sr_benchmark)。Bailey & López de Prado (2012)。

    kurt は非超過尖度（正規分布 = 3）。sr, sr_benchmark は per-period。
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        raise ValueError("non-positive variance term in PSR (check skew/kurt/sr)")
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR]：真 SR=0 の N 個の独立試行から得られる最大 Sharpe の期待値。

    「False Strategy 定理」(López de Prado & Lewis 2018)。DSR の基準 SR* に使う。
    sr_variance は試行間の Sharpe 推定値の分散 V[{SR_n}]。
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sr_variance < 0:
        raise ValueError("sr_variance must be >= 0")
    if n_trials == 1:
        return 0.0
    sigma = np.sqrt(sr_variance)
    g = EULER_MASCHERONI
    maxz = ((1.0 - g) * norm.ppf(1.0 - 1.0 / n_trials)
            + g * norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    return float(sigma * maxz)


def deflated_sharpe_ratio(sr: float, sr_variance: float, n_trials: int,
                          n_obs: int, skew: float, kurt: float) -> float:
    """DSR：多重検定・非正規性・標本長を補正した「真 SR>0」確率。

    Bailey & López de Prado (2014). = PSR(sr*), sr* = E[max SR]（多重検定の膨張分）。
    すべての Sharpe は per-period。
    """
    sr_star = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(sr, sr_star, n_obs, skew, kurt)


def deflated_sharpe_ratio_from_returns(returns, sr_variance: float,
                                       n_trials: int) -> float:
    """便宜関数：リターン系列から sr/skew/kurt/n を推定して DSR を返す。"""
    sr, sk, ku, n = _moments(returns)
    return deflated_sharpe_ratio(sr, sr_variance, n_trials, n, sk, ku)


def min_track_record_length(sr: float, sr_benchmark: float, skew: float,
                            kurt: float, prob: float = 0.95) -> float:
    """minTRL：PSR(sr_benchmark) >= prob を満たす最小観測数。"""
    if not (0.0 < prob < 1.0):
        raise ValueError("prob must be in (0,1)")
    if sr <= sr_benchmark:
        raise ValueError("sr must exceed sr_benchmark for finite minTRL")
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    return float(1.0 + denom * (norm.ppf(prob) / (sr - sr_benchmark)) ** 2)
