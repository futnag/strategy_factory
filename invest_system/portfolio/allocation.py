"""ロバストなポートフォリオ配分（L9）：最小分散・HRP・NCO。

統合ナレッジベース §6 / DP9。AFML ch.16（HRP）, NCO。
逆行列を直接使う Markowitz は信号/ノイズ誘導不安定性に脆弱。HRP は逆行列を使わず
樹状構造で配分し、NCO は相関ブロックに分割して不安定性をブロック内に封じ込める。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .denoise import cov_to_corr


def min_variance_weights(cov, mu: Optional[pd.Series] = None):
    """Markowitz 最適化（mu=None で大域最小分散）。逆行列直接法（比較用ベースライン）。"""
    is_df = isinstance(cov, pd.DataFrame)
    c = cov.values if is_df else np.asarray(cov, dtype=float)
    inv = np.linalg.inv(c)
    if mu is None:
        rhs = np.ones((c.shape[0], 1))
    else:
        rhs = (mu.values if isinstance(mu, pd.Series) else np.asarray(mu, dtype=float)).reshape(-1, 1)
    w = inv @ rhs
    w = (w / (np.ones((1, c.shape[0])) @ w)).flatten()
    return pd.Series(w, index=cov.index) if is_df else w


# --- HRP（AFML ch.16） ----------------------------------------------------
def _ivp(cov: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _cluster_var(cov: pd.DataFrame, items) -> float:
    sub = cov.loc[items, items].values
    w = _ivp(sub)
    return float(w @ sub @ w)


def _quasi_diag(link: np.ndarray) -> list:
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def hrp_weights(cov):
    """Hierarchical Risk Parity。逆行列不要のロバスト配分。AFML ch.16。"""
    cov = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(cov)
    corr = pd.DataFrame(cov_to_corr(cov.values), index=cov.index, columns=cov.columns)
    dist = np.sqrt(np.clip((1.0 - corr.values) / 2.0, 0.0, None))
    link = linkage(squareform(dist, checks=False), method="single")
    sort_ix = [cov.index[i] for i in _quasi_diag(link)]

    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [c[k:l] for c in clusters
                    for k, l in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w.reindex(cov.index)


# --- NCO --------------------------------------------------------------------
def _cluster_kmeans(corr: np.ndarray, max_k: int = 10, n_init: int = 10,
                    random_state: int = 0) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples

    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    max_k = min(max_k, corr.shape[0] - 1)
    best_q, best_labels = -np.inf, None
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=n_init,
                        random_state=random_state).fit_predict(dist)
        silh = silhouette_samples(dist, labels)
        quality = silh.mean() / silh.std() if silh.std() > 0 else 0.0
        if quality > best_q:
            best_q, best_labels = quality, labels
    clusters: dict = {}
    for i, lab in enumerate(best_labels):
        clusters.setdefault(int(lab), []).append(i)
    return clusters


def nco_weights(cov, mu: Optional[pd.Series] = None, max_k: int = 10):
    """Nested Clustered Optimization。相関ブロックに分割して最適化し不安定性を封じる。"""
    cov = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(cov)
    labels = list(cov.index)
    clusters = _cluster_kmeans(cov_to_corr(cov.values), max_k)

    # クラスタ内最適化
    w_intra = pd.DataFrame(0.0, index=cov.index, columns=sorted(clusters))
    for cl, idx in clusters.items():
        items = [labels[i] for i in idx]
        mu_ = None if mu is None else mu.loc[items]
        w_intra.loc[items, cl] = min_variance_weights(cov.loc[items, items], mu_).values

    # クラスタ間最適化（縮約共分散）
    W = w_intra.values
    cov_inter = pd.DataFrame(W.T @ cov.values @ W,
                             index=w_intra.columns, columns=w_intra.columns)
    mu_inter = None if mu is None else pd.Series(W.T @ mu.values, index=w_intra.columns)
    w_inter = min_variance_weights(cov_inter, mu_inter)

    # 合成
    return w_intra.mul(w_inter, axis=1).sum(axis=1).reindex(cov.index)
