"""共分散のノイズ除去（L9）。

統合ナレッジベース §6 / DP9。AFML ch.2。
ランダム行列理論（Marčenko-Pastur）により、ノイズに由来する固有値を平均値で
置き換える（定数残差固有値法）。これで Markowitz の信号/ノイズ誘導不安定性を緩和。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """共分散行列を相関行列に変換。"""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1.0, 1.0)


def marcenko_pastur_pdf(var: float, q: float, pts: int = 1000) -> pd.Series:
    """Marčenko-Pastur 分布の密度。q = T/N（観測数/変数数）, var = ノイズ分散。"""
    lam_min = var * (1.0 - np.sqrt(1.0 / q)) ** 2
    lam_max = var * (1.0 + np.sqrt(1.0 / q)) ** 2
    lam = np.linspace(lam_min, lam_max, pts)
    pdf = q / (2.0 * np.pi * var * lam) * np.sqrt((lam_max - lam) * (lam - lam_min))
    return pd.Series(pdf, index=lam)


def _fit_kde(obs, x, bwidth: float = 0.25) -> np.ndarray:
    from sklearn.neighbors import KernelDensity

    obs = np.asarray(obs, dtype=float).reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=bwidth).fit(obs)
    return np.exp(kde.score_samples(np.asarray(x, dtype=float).reshape(-1, 1)))


def _err_pdf(var, eigenvalues, q, bwidth, pts=1000) -> float:
    v = float(var[0]) if np.ndim(var) else float(var)
    theo = marcenko_pastur_pdf(v, q, pts)
    emp = _fit_kde(eigenvalues, theo.index.values, bwidth)
    return float(np.sum((emp - theo.values) ** 2))


def find_max_eval(eigenvalues: np.ndarray, q: float, bwidth: float = 0.25):
    """ノイズ分散 var をフィットし、MP 上限固有値 λ+ を返す。AFML snippet 2.4。"""
    from scipy.optimize import minimize

    res = minimize(_err_pdf, x0=np.array([0.5]),
                   args=(eigenvalues, q, bwidth), bounds=[(1e-5, 1.0 - 1e-5)])
    var = float(res.x[0]) if res.success else 1.0
    lam_max = var * (1.0 + np.sqrt(1.0 / q)) ** 2
    return lam_max, var


def denoise_covariance(cov, q: float, bwidth: float = 0.25):
    """RMT による共分散ノイズ除去（定数残差固有値法）。AFML snippet 2.5。

    cov : pd.DataFrame | np.ndarray   共分散行列。
    q : float                          T/N（観測数/変数数）。
    Returns 同形・同ラベルの除去後共分散。
    """
    is_df = isinstance(cov, pd.DataFrame)
    labels = cov.index if is_df else None
    cov_v = cov.values if is_df else np.asarray(cov, dtype=float)

    corr = cov_to_corr(cov_v)
    eVal, eVec = np.linalg.eigh(corr)
    order = eVal.argsort()[::-1]
    eVal, eVec = eVal[order], eVec[:, order]

    lam_max, _ = find_max_eval(eVal, q, bwidth)
    n_facts = int((eVal > lam_max).sum())

    eVal_ = eVal.copy()
    if n_facts < len(eVal_):                        # ノイズ固有値を平均で置換
        eVal_[n_facts:] = eVal_[n_facts:].mean()

    corr_ = eVec @ np.diag(eVal_) @ eVec.T
    corr_ = cov_to_corr(corr_)                      # 単位対角に再スケール
    std = np.sqrt(np.diag(cov_v))
    cov_ = corr_ * np.outer(std, std)
    return pd.DataFrame(cov_, index=labels, columns=labels) if is_df else cov_
