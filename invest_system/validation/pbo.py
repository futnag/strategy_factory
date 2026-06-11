"""PBO（バックテスト過学習確率）— CSCV 法。

Bailey, Borwein, López de Prado, Zhu (2016) "The Probability of Backtest
Overfitting", Journal of Computational Finance 20(4)。docs/02 §13 / L8 で
主指標として設計済みだったが未実装だったギャップの充足（2026-06）。

PBO = 「IS（インサンプル）で最良だった構成が、OOS（アウトオブサンプル）で
中央値**以下**になる確率」。CSCV（組合せ対称クロスバリデーション）：
リターン行列（T×N, 列=グリッド構成）を時間軸で S 個のブロックに等分し、
S/2 個を IS・残りを OOS とする全 C(S, S/2) 通りで
  1. IS の Sharpe 最大の構成 n* を選ぶ
  2. OOS の Sharpe で n* の相対順位 ω = rank/(N+1) を測る
を繰り返し、PBO = P(ω ≤ 0.5)。ノイズだけのグリッドなら ω は一様分布
＝PBO≈0.5。真のエッジがあれば PBO→0、過学習グリッドなら PBO→1。

判定には使わない（**表示専用**・DP18。判定は DSR≥0.95 のみ）。決定論的
（乱数不使用）・純関数・ネットワーク不要。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class PBOResult:
    pbo: float                 # P(IS最良がOOSで中央値以下)
    n_combinations: int        # C(S, S/2)
    n_strategies: int          # N（グリッド構成数）
    n_obs: int                 # 共通標本長 T
    n_splits: int              # S
    logits: np.ndarray         # 各組合せの λ = ln(ω/(1−ω))


def _sharpe_cols(m: np.ndarray) -> np.ndarray:
    """列ごとの per-period Sharpe（mean/std, ddof=1）。std=0 は -inf（選ばれない）。"""
    mu = m.mean(axis=0)
    sd = m.std(axis=0, ddof=1)
    out = np.full(m.shape[1], -np.inf)
    ok = sd > 0
    out[ok] = mu[ok] / sd[ok]
    return out


def pbo_cscv(returns: pd.DataFrame, n_splits: int = 8) -> PBOResult:
    """CSCV による PBO。returns: 行=時点・列=グリッド構成のリターン行列。

    行は全列が揃う共通標本に落とす（dropna）。S は偶数（既定 8 → C(8,4)=70
    組合せ。月次 ~120 観測ならブロック ~15 行）。N<2 または標本不足は
    PBO=NaN（判定不能を明示）。
    """
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError("n_splits must be an even integer >= 2")
    df = returns.dropna(how="any")
    n_obs, n_strat = df.shape
    if n_strat < 2 or n_obs < 2 * n_splits:
        return PBOResult(float("nan"), 0, n_strat, n_obs, n_splits,
                         np.empty(0))
    m = df.to_numpy(dtype=float)
    blocks = np.array_split(np.arange(n_obs), n_splits)
    half = n_splits // 2
    logits = []
    below = 0
    for combo in combinations(range(n_splits), half):
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate([blocks[b] for b in range(n_splits)
                                   if b not in combo])
        n_star = int(np.argmax(_sharpe_cols(m[is_rows])))
        oos_sr = _sharpe_cols(m[oos_rows])
        # n* の OOS 相対順位 ω（1=最下位 … N=最上位）/(N+1)
        rank = 1 + int((oos_sr < oos_sr[n_star]).sum())
        omega = rank / (n_strat + 1.0)
        logits.append(float(np.log(omega / (1.0 - omega))))
        if omega <= 0.5:
            below += 1
    lam = np.asarray(logits)
    return PBOResult(below / len(lam), len(lam), n_strat, n_obs, n_splits, lam)
