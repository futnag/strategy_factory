"""オンライン動的ヘッジ（柱D・KB §11.4）。

Kalman フィルタで時変ヘッジ比 β_t と切片 α_t を逐次推定。各 t の出力は ≤t の観測のみ
で決まる＝本質的にオンライン＝先読み不能（DP12 を最も自然に満たす）。自前 numpy 実装
（新規重依存なし）。教科書の静的 OLS ヘッジを置換する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _align_idx(x, y):
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        df = pd.concat([x, y], axis=1).dropna()
        return (df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float),
                df.index)
    xx = np.asarray(x, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    return xx, yy, pd.RangeIndex(len(xx))


class KalmanHedge:
    """状態 [β, α] を観測 y_t = β_t·x_t + α_t + ε で逐次更新する動的回帰。

    delta: 状態遷移ノイズ比（大きいほど β が速く動く）。r: 観測ノイズ分散。
    予測誤差 e_t（＝スプレッド）と sqrt(q_t)（＝誤差分散）を返し、z=e/sqrt(q) を信号に。
    e_t は「事前状態（≤t-1）＋既知の x_t」による一段先予測＝先読みなし。
    """

    def __init__(self, delta: float = 1e-4, r: float = 1e-3):
        self.delta = float(delta)
        self.r = float(r)

    def filter(self, x, y) -> pd.DataFrame:
        """x, y（同 index の Series か配列）を逐次フィルタ。

        Returns
        -------
        DataFrame(index=共通日付, columns=[beta, alpha, e, sqrt_q, z])
            各行 t は ≤t の観測のみで決まる（再帰式＝オンライン＝先読み不能）。
        """
        xs, ys, idx = _align_idx(x, y)
        n = xs.size
        beta = np.zeros((n, 2))
        e = np.full(n, np.nan)
        sqrt_q = np.full(n, np.nan)
        b = np.zeros(2)                       # 初期状態 [β0, α0] = 0
        P = np.eye(2)                         # 初期不確実性（拡散的）
        Vw = self.delta / (1.0 - self.delta) * np.eye(2)
        for t in range(n):
            F = np.array([xs[t], 1.0])        # 観測行列 [x_t, 1]
            if t > 0:
                P = P + Vw                     # ランダムウォーク状態の予測ステップ
            et = ys[t] - F @ b                 # 一段先予測誤差（＝スプレッド）
            q = float(F @ P @ F) + self.r      # 予測誤差分散
            K = (P @ F) / q                    # カルマンゲイン
            b = b + K * et                     # 状態更新
            P = P - np.outer(K, F) @ P
            beta[t] = b
            e[t] = et
            sqrt_q[t] = np.sqrt(q)
        z = e / sqrt_q
        return pd.DataFrame(
            {"beta": beta[:, 0], "alpha": beta[:, 1], "e": e,
             "sqrt_q": sqrt_q, "z": z}, index=idx)
