"""ノイズ除去とロバスト配分（最小分散/HRP/NCO）を検証。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.portfolio.allocation import (
    hrp_weights,
    min_variance_weights,
    nco_weights,
)
from invest_system.portfolio.denoise import cov_to_corr, denoise_covariance


def _labels(n):
    return [f"a{i:02d}" for i in range(n)]


def _noisy_cov(n=30, k=3, t=60, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0, 1, (n, k))
    factors = rng.normal(0, 1, (t, k))
    idio = rng.normal(0, 0.5, (t, n))
    returns = factors @ loadings.T + idio
    cov = np.cov(returns, rowvar=False)
    return pd.DataFrame(cov, index=_labels(n), columns=_labels(n)), t / n


# --- ノイズ除去 -----------------------------------------------------------
def test_denoise_preserves_variances_and_trace():
    cov, q = _noisy_cov()
    den = denoise_covariance(cov, q)
    assert np.allclose(np.diag(den.values), np.diag(cov.values))   # 分散は不変
    corr_den = cov_to_corr(den.values)
    assert np.isclose(np.trace(corr_den), cov.shape[0])            # 相関のトレース = N
    assert (np.linalg.eigvalsh(corr_den) > -1e-8).all()           # 半正定値


def test_denoise_reduces_conditioning():
    cov, q = _noisy_cov()
    cond0 = np.linalg.cond(cov_to_corr(cov.values))
    cond1 = np.linalg.cond(cov_to_corr(denoise_covariance(cov, q).values))
    assert cond1 < cond0                                          # 悪条件を緩和


# --- 最小分散 -------------------------------------------------------------
def test_min_variance_two_asset_analytic():
    cov = pd.DataFrame([[1.0, 0.0], [0.0, 4.0]], index=["x", "y"], columns=["x", "y"])
    w = min_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    assert w["x"] == pytest.approx(0.8)      # 逆分散 ∝ [1, 1/4] → [0.8, 0.2]
    assert w["y"] == pytest.approx(0.2)


# --- HRP ------------------------------------------------------------------
def test_hrp_equal_weights_on_identity():
    cov = pd.DataFrame(np.eye(4), index=_labels(4), columns=_labels(4))
    w = hrp_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w.values, 0.25)


def test_hrp_balances_correlated_blocks():
    corr = np.array([[1.0, 0.9, 0.1, 0.1],
                     [0.9, 1.0, 0.1, 0.1],
                     [0.1, 0.1, 1.0, 0.9],
                     [0.1, 0.1, 0.9, 1.0]])
    cov = pd.DataFrame(corr, index=_labels(4), columns=_labels(4))
    w = hrp_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()
    # 2つの相関ブロックにほぼ均等配分（各ブロック ≈ 0.5）
    assert abs((w.iloc[0] + w.iloc[1]) - 0.5) < 0.1


# --- NCO ------------------------------------------------------------------
def test_nco_weights_sum_to_one_on_block_structure():
    rng = np.random.default_rng(1)
    n = 12
    blocks = np.repeat([0, 1, 2], n // 3)
    corr = np.full((n, n), 0.05)
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        for i in idx:
            for j in idx:
                corr[i, j] = 1.0 if i == j else 0.6
    np.fill_diagonal(corr, 1.0)
    cov = pd.DataFrame(corr, index=_labels(n), columns=_labels(n))
    w = nco_weights(cov, max_k=6)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(w.values).all()
