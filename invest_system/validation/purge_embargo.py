"""Purging & Embargo：リーク防止のための外科的処置。

統合ナレッジベース §5.1 / DP5 の実装。AFML (López de Prado 2018) ch.7。
イベントは pandas.Series で表現：index = ラベル開始 t0, value = ラベル終了 t1。
index は一意（重複なし）かつ単調増加を仮定する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    """テスト期間とラベル区間が重なる訓練観測をパージ。AFML snippet 7.1。

    Parameters
    ----------
    t1 : pd.Series
        訓練候補。index = 観測開始 t0、value = 観測終了 t1。
    test_times : pd.Series
        テスト区間。index = テスト開始 t0、value = テスト終了 t1。

    Returns
    -------
    pd.Series
        パージ後の訓練 t1 Series。
    """
    trn = t1.copy(deep=True)
    for start, end in test_times.items():
        df0 = trn[(start <= trn.index) & (trn.index <= end)].index   # 訓練開始がテスト内
        df1 = trn[(start <= trn) & (trn <= end)].index               # 訓練終了がテスト内
        df2 = trn[(trn.index <= start) & (end <= trn)].index          # 訓練がテストを内包
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


def embargo_after(index: pd.Index, test_idx: np.ndarray, embargo: int) -> np.ndarray:
    """各テストブロック直後の embargo 件の位置インデックスを返す（前方エンバーゴ）。

    テスト集合の「直後」の訓練データは系列相関でテスト情報を含みうるため遮断する。
    AFML snippet 7.2 の考え方を位置インデックスで実装。

    Parameters
    ----------
    index : pd.Index
        全観測の index（長さ n の判定にのみ使用）。
    test_idx : np.ndarray
        テスト観測の位置インデックス（整数）。
    embargo : int
        各テストブロック直後に遮断する観測数。

    Returns
    -------
    np.ndarray
        遮断すべき位置インデックス（昇順・一意）。
    """
    n = len(index)
    if embargo <= 0 or test_idx.size == 0:
        return np.empty(0, dtype=int)
    test_sorted = np.unique(test_idx)
    banned: set[int] = set()
    block_end = int(test_sorted[0])
    for i in range(1, test_sorted.size):
        cur = int(test_sorted[i])
        if cur == block_end + 1:
            block_end = cur
            continue
        banned.update(range(block_end + 1, min(block_end + 1 + embargo, n)))
        block_end = cur
    banned.update(range(block_end + 1, min(block_end + 1 + embargo, n)))
    banned.difference_update(test_sorted.tolist())  # テスト自身は対象外
    return np.array(sorted(banned), dtype=int)
