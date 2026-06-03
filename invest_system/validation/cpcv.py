"""Combinatorial Purged Cross-Validation (CPCV)。

統合ナレッジベース §5.2 / DP6 の実装。AFML (López de Prado 2018) ch.12。
単一パス検証を排し、複数のバックテストパスを生成して Sharpe を「分布」で評価する。
n_test_splits=1 のときは Purged K-Fold（パス数 1）に縮退する。

前提：X.index は一意・単調増加（典型的なバー系列）。
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pandas as pd

from .purge_embargo import get_train_times, embargo_after


class CombinatorialPurgedKFold:
    """N 群を作り、k 群をテストとする全 C(N,k) 通りで (train, test) を生成。

    各分割でテストとラベル期間が重なる訓練観測をパージし、テスト直後に
    前方エンバーゴを適用する。
    """

    def __init__(self, n_splits: int = 6, n_test_splits: int = 2,
                 embargo_pct: float = 0.0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not (1 <= n_test_splits < n_splits):
            raise ValueError("require 1 <= n_test_splits < n_splits")
        if not (0.0 <= embargo_pct < 1.0):
            raise ValueError("embargo_pct must be in [0,1)")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

    def get_n_splits(self) -> int:
        """テスト分割の組合せ数 = C(N, k)。"""
        return comb(self.n_splits, self.n_test_splits)

    def get_n_paths(self) -> int:
        """生成されるバックテストパス数 φ = C(N-1, k-1)。"""
        return comb(self.n_splits - 1, self.n_test_splits - 1)

    def split(self, X, t1: pd.Series):
        """(train_idx, test_idx) を位置インデックス(np.ndarray)で yield する。

        Parameters
        ----------
        X : pandas object
            index = 観測開始 t0。
        t1 : pd.Series
            X と同じ index を持ち、value = ラベル終了 t1。
        """
        if not X.index.equals(t1.index):
            raise ValueError("X and t1 must share the same index")
        n = X.shape[0]
        positions = np.arange(n)
        groups = np.array_split(positions, self.n_splits)
        embargo = int(n * self.embargo_pct)

        for test_groups in itertools.combinations(range(self.n_splits),
                                                  self.n_test_splits):
            test_pos = np.sort(np.concatenate([groups[g] for g in test_groups]))
            test_times = self._block_spans(X.index, t1, test_groups, groups)
            train_t1 = get_train_times(t1, test_times)
            train_pos = X.index.get_indexer(train_t1.index)
            train_pos = train_pos[train_pos >= 0]
            if embargo > 0:
                banned = embargo_after(X.index, test_pos, embargo)
                if banned.size:
                    train_pos = np.setdiff1d(train_pos, banned)
            yield np.sort(train_pos), test_pos

    @staticmethod
    def _block_spans(index: pd.Index, t1: pd.Series, test_groups, groups) -> pd.Series:
        """選択されたテスト群を連続ブロックにまとめ、各ブロックの [t0_first, max t1] を返す。"""
        starts, ends = [], []
        for block in _contiguous(sorted(test_groups)):
            block_pos = np.concatenate([groups[g] for g in block])
            b0, b1 = int(block_pos.min()), int(block_pos.max())
            starts.append(index[b0])
            ends.append(t1.iloc[b0:b1 + 1].max())
        return pd.Series(ends, index=starts)


def _contiguous(sorted_ids):
    """連続整数を塊に分割：[0,1,3] -> [[0,1],[3]]。"""
    blocks, cur = [], [sorted_ids[0]]
    for x in sorted_ids[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            blocks.append(cur)
            cur = [x]
    blocks.append(cur)
    return blocks
