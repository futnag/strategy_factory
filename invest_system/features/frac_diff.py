"""分数階差分（FFD）：定常性とメモリの両立。

統合ナレッジベース §3.1 / DP2 の実装。AFML (López de Prado 2018) ch.5。
d=1 の整数差分（リターン化）はメモリを抹消するため使わない。ADF で定常化する
最小の d を選ぶことで、定常性を確保しつつ元系列との高い相関（記憶）を保つ。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def get_weights_ffd(d: float, thresh: float = 1e-5) -> np.ndarray:
    """固定幅窓 FFD の重み（古い→新しい の順、末尾が最新=1.0）。"""
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff_ffd(series: pd.Series, d: float,
                  thresh: float = 1e-5) -> pd.Series:
    """固定幅窓の分数階差分系列。先頭 width 件は窓不足で除外して返す。"""
    w = get_weights_ffd(d, thresh)
    width = len(w) - 1
    x = series.to_numpy(dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(width, n):
        out[i] = float(np.dot(w, x[i - width:i + 1]))
    res = pd.Series(out, index=series.index, name=series.name)
    return res.iloc[width:] if width > 0 else res


def frac_diff_d_table(series: pd.Series, d_grid: Optional[np.ndarray] = None,
                      thresh: float = 1e-5, signif: str = "5%") -> pd.DataFrame:
    """各 d について ADF統計量・p値・臨界値・元系列との相関・定常性を集計。

    KB §3.1 の「d vs ADF vs メモリ保存」表を再現する。index=d。
    注：FFD の窓幅は d→0 で広がるため、小さな d の評価には十分に長い系列が必要。
    窓が系列長を超える d は観測数不足（<10）として非定常扱いにする。
    """
    from statsmodels.tsa.stattools import adfuller

    if d_grid is None:
        d_grid = np.round(np.linspace(0.0, 1.0, 11), 2)
    rows = []
    for d in d_grid:
        ffd = frac_diff_ffd(series, float(d), thresh).dropna()
        if ffd.shape[0] < 10 or float(ffd.std()) == 0.0:
            rows.append((float(d), np.nan, np.nan, np.nan, np.nan, False))
            continue
        adf_stat, pval, _, _, crit, _ = adfuller(
            ffd.to_numpy(), regression="c", autolag="AIC")
        corr = float(np.corrcoef(series.loc[ffd.index].to_numpy(),
                                 ffd.to_numpy())[0, 1])
        rows.append((float(d), float(adf_stat), float(pval),
                     float(crit[signif]), corr, bool(adf_stat < crit[signif])))
    table = pd.DataFrame(
        rows, columns=["d", "adf_stat", "pval", f"crit_{signif}",
                       "corr", "stationary"])
    return table.set_index("d")


def find_min_d(series: pd.Series, d_grid: Optional[np.ndarray] = None,
               thresh: float = 1e-5, signif: str = "5%"):
    """定常性を満たす最小の d と、その相関（メモリ保存度）を返す。

    Returns
    -------
    (min_d, corr_at_min_d, table)
        定常化しなければ (None, None, table)。
    """
    table = frac_diff_d_table(series, d_grid, thresh, signif)
    stationary = table[table["stationary"]]
    if stationary.empty:
        return None, None, table
    min_d = float(stationary.index.min())
    return min_d, float(stationary.loc[min_d, "corr"]), table
