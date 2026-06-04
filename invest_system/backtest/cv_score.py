"""Purged 交差検証スコアリング（リーク無しの OOS 評価）。

統合ナレッジベース §5.1-5.2 / DP5-DP6。CombinatorialPurgedKFold(k=1) =
purged k-fold で各観測をちょうど一度ずつ test にし、out-of-fold 予測を返す。
回帰（ボラ予測等）・分類のどちらにも使える汎用評価。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..validation.cpcv import CombinatorialPurgedKFold


def purged_cv_predict(X: pd.DataFrame, y, t1: pd.Series,
                      model_factory: Callable[[], object],
                      n_splits: int = 6, embargo_pct: float = 0.01) -> pd.Series:
    """purged k-fold の out-of-fold 予測（各観測を一度ずつ test）。

    Parameters
    ----------
    X : DataFrame        特徴量（index = イベント時刻）。
    y : array-like       目的変数（X と同順・同長）。
    t1 : Series          各観測のラベル終了時刻（パージング用）。X と同 index。
    model_factory : () -> estimator   fit/predict を持つ推定器を新規生成。

    Returns
    -------
    pd.Series   X.index に整合した out-of-fold 予測。
    """
    if not X.index.equals(t1.index):
        raise ValueError("X and t1 must share the same index")
    cv = CombinatorialPurgedKFold(n_splits, 1, embargo_pct)   # k=1 → 各obs1回test
    y_arr = np.asarray(y, dtype=float)
    oos = np.full(len(X), np.nan)
    for train_idx, test_idx in cv.split(X, t1):
        model = model_factory()
        model.fit(X.iloc[train_idx], y_arr[train_idx])
        oos[test_idx] = model.predict(X.iloc[test_idx])
    return pd.Series(oos, index=X.index)
