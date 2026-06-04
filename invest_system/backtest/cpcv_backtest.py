"""CPCV バックテスト：purged CPCV で一次モデルを学習・予測し、φ 本のパスを復元。

統合ナレッジベース §5.2 / DP6 / DP10。AFML (López de Prado 2018) ch.12。
各分割で purged-train に学習し test を予測。各グループはちょうど φ=C(N-1,k-1) 個の
分割でテストされるので、p 番目の分割を path p に割り当てることで、全観測を一度ずつ
被覆する φ 本の完全なバックテストパスを復元する。点推定でなく Sharpe の「分布」を返す。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..validation.cpcv import CombinatorialPurgedKFold


@dataclass
class CPCVBacktestResult:
    """CPCV バックテストの結果。"""

    path_sharpes: np.ndarray   # 各パスの per-period Sharpe
    paths: np.ndarray          # φ×n の予測ポジション（通常は全被覆＝NaN無し）
    n_paths: int

    @property
    def mean_sharpe(self) -> float:
        return float(np.mean(self.path_sharpes)) if self.path_sharpes.size else float("nan")

    @property
    def std_sharpe(self) -> float:
        return float(np.std(self.path_sharpes, ddof=1)) if self.path_sharpes.size > 1 else 0.0

    @property
    def min_sharpe(self) -> float:
        return float(np.min(self.path_sharpes)) if self.path_sharpes.size else float("nan")

    @property
    def max_sharpe(self) -> float:
        return float(np.max(self.path_sharpes)) if self.path_sharpes.size else float("nan")

    @property
    def frac_negative(self) -> float:
        return float(np.mean(self.path_sharpes < 0)) if self.path_sharpes.size else float("nan")


def cpcv_backtest(X: pd.DataFrame, y: pd.Series, ret: pd.Series, t1: pd.Series,
                  cv: CombinatorialPurgedKFold,
                  model_factory: Callable[[], object],
                  sample_weight: Optional[pd.Series] = None) -> CPCVBacktestResult:
    """purged CPCV で一次モデルを評価し、φ 本のパスの Sharpe 分布を返す。

    Parameters
    ----------
    X : DataFrame      特徴量（index = イベント時刻）。
    y : Series         ラベル（方向, 例 {-1,+1}）。X と同 index。
    ret : Series       各イベントの実現リターン（トリプルバリア）。X と同 index。
    t1 : Series        各イベントのラベル終了時刻（パージング用）。X と同 index。
    cv : CombinatorialPurgedKFold
    model_factory : () -> estimator   fit/predict を持つ推定器を新規生成。
    sample_weight : Series | None     非IID独自性重み等（L6）。任意。

    Returns
    -------
    CPCVBacktestResult
    """
    if not (X.index.equals(y.index) and X.index.equals(ret.index)
            and X.index.equals(t1.index)):
        raise ValueError("X, y, ret, t1 must share the same index")

    n = X.shape[0]
    N = cv.n_splits
    groups = np.array_split(np.arange(n), N)
    group_of_pos = np.empty(n, dtype=int)
    for gi, arr in enumerate(groups):
        group_of_pos[arr] = gi

    y_arr = np.asarray(y)
    w_arr = None if sample_weight is None else np.asarray(sample_weight)

    # 各分割で学習・予測
    splits = []  # (test_idx, preds)
    for train_idx, test_idx in cv.split(X, t1):
        model = model_factory()
        x_tr = X.iloc[train_idx]
        y_tr = y_arr[train_idx]
        if w_arr is not None:
            model.fit(x_tr, y_tr, sample_weight=w_arr[train_idx])
        else:
            model.fit(x_tr, y_tr)
        preds = np.asarray(model.predict(X.iloc[test_idx]), dtype=float)
        splits.append((test_idx, preds))

    # 各グループをテストした分割インデックス（出現順）
    group_splits: dict[int, list[int]] = {g: [] for g in range(N)}
    for si, (test_idx, _) in enumerate(splits):
        for g in np.unique(group_of_pos[test_idx]):
            group_splits[int(g)].append(si)

    # φ 本のパスを復元（path p ← 各グループの p 番目のテスト分割）
    phi = cv.get_n_paths()
    ret_arr = ret.to_numpy(dtype=float)
    paths = np.full((phi, n), np.nan)
    for p in range(phi):
        for g in range(N):
            si = group_splits[g][p]
            test_idx, preds = splits[si]
            gmask = group_of_pos[test_idx] == g
            paths[p, test_idx[gmask]] = preds[gmask]

    # 各パスの戦略リターン（ポジション×実現リターン）→ Sharpe
    path_sharpes = []
    for p in range(phi):
        strat = paths[p] * ret_arr
        strat = strat[~np.isnan(strat)]
        if strat.size > 1 and strat.std(ddof=1) > 0:
            path_sharpes.append(strat.mean() / strat.std(ddof=1))
    return CPCVBacktestResult(np.array(path_sharpes), paths, phi)
