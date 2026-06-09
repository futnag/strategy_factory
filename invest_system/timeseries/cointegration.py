"""共和分とヘッジ比（柱D・KB §11.3）— スプレッドの作り方。

個々が I(1) でも線形結合が I(0) になる関係＝共和分。結合係数＝ヘッジ比。全関数は
「渡された ≤t 系列」で推定＝AsOf（全期間一括は先読み＝DP12 違反）。statsmodels lazy import。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _align(y, x) -> tuple[np.ndarray, np.ndarray]:
    """y, x を共通 index（Series 時）で内部結合し NaN 除去した配列対を返す。"""
    if isinstance(y, pd.Series) and isinstance(x, pd.Series):
        df = pd.concat([y, x], axis=1).dropna()
        return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
    yy = np.asarray(y, dtype=float).ravel()
    xx = np.asarray(x, dtype=float).ravel()
    m = ~(np.isnan(yy) | np.isnan(xx))
    return yy[m], xx[m]


def hedge_ratio_ols(y, x) -> float:
    """OLS 回帰 y = α + β·x の β（ヘッジ比）。説明変数選択に非対称。"""
    yy, xx = _align(y, x)
    if yy.size < 2:
        return float("nan")
    X = np.column_stack([np.ones_like(xx), xx])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return float(coef[1])


def hedge_ratio_tls(y, x) -> float:
    """直交（TLS/PCA）回帰のヘッジ比。説明変数の選び方に対し対称。"""
    yy, xx = _align(y, x)
    if yy.size < 2:
        return float("nan")
    data = np.column_stack([xx - xx.mean(), yy - yy.mean()])
    _, _, vt = np.linalg.svd(data, full_matrices=False)
    vx, vy = vt[-1]                       # 最小特異値方向
    return float(-vx / vy) if vy != 0 else float("nan")


def cadf(y, x, regression: str = "c") -> tuple[float, float]:
    """Engle-Granger CADF：β=OLSヘッジ比、残差 y-α-β·x を ADF。返り値 (β, adf_pval)。

    adf_pval が小さいほど残差が定常＝共和分（取引可能なスプレッド）。KB §11.3。
    """
    from statsmodels.tsa.stattools import adfuller

    yy, xx = _align(y, x)
    if yy.size < 12:
        return (float("nan"), float("nan"))
    X = np.column_stack([np.ones_like(xx), xx])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    beta = float(coef[1])
    resid = yy - X @ coef
    if np.std(resid) == 0:
        return (beta, float("nan"))
    pval = float(adfuller(resid, regression=regression, autolag="AIC")[1])
    return (beta, pval)


def spread_series(y, x, beta: float) -> pd.Series:
    """スプレッド y - β·x（Series なら共通 index で整列）。"""
    if isinstance(y, pd.Series) and isinstance(x, pd.Series):
        df = pd.concat([y, x], axis=1).dropna()
        return df.iloc[:, 0] - beta * df.iloc[:, 1]
    yy = np.asarray(y, dtype=float).ravel()
    xx = np.asarray(x, dtype=float).ravel()
    return pd.Series(yy - beta * xx)


@dataclass
class JohansenResult:
    """Johansen 検定結果。eigvecs の先頭列が最強の平均回帰結合。"""

    eigvecs: np.ndarray          # 列が固有ベクトル（最強回帰＝先頭列）
    eig: np.ndarray
    trace_stat: np.ndarray
    crit_95: np.ndarray
    n_relations: int             # trace>crit95 を満たす共和分関係の数

    @property
    def strongest(self) -> np.ndarray:
        """最強の平均回帰結合（先頭固有ベクトル）。"""
        return self.eigvecs[:, 0]


def johansen(prices: pd.DataFrame, det_order: int = 0,
             k_ar_diff: int = 1) -> JohansenResult:
    """Johansen 共和分検定（3資産以上のバスケット）。statsmodels lazy import。

    trace 統計量と 95% 臨界値、固有ベクトル（最強回帰＝先頭列）を返す。KB §11.3。
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    p = prices.dropna()
    res = coint_johansen(np.asarray(p, dtype=float), det_order, k_ar_diff)
    trace = np.asarray(res.lr1, dtype=float)
    crit95 = np.asarray(res.cvt, dtype=float)[:, 1]      # [90%,95%,99%] の 95%
    n_rel = int(np.sum(trace > crit95))
    return JohansenResult(np.asarray(res.evec, dtype=float),
                          np.asarray(res.eig, dtype=float), trace, crit95, n_rel)
