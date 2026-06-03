"""パージング＆エンバーゴの正しさを検証。"""
import numpy as np
import pandas as pd

from invest_system.validation.purge_embargo import get_train_times, embargo_after


def _index(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_get_train_times_purges_overlapping_labels():
    idx = _index(10)
    # 各ラベルは2日間（t1 = t0 + 2日）
    t1 = pd.Series(idx + pd.Timedelta(days=2), index=idx)
    # テスト区間 [day3, day5]
    test_times = pd.Series([idx[5]], index=[idx[3]])
    trn = get_train_times(t1, test_times)
    remaining = set(idx.get_indexer(trn.index))
    assert remaining == {0, 6, 7, 8, 9}


def test_get_train_times_no_overlap_keeps_all():
    idx = _index(6)
    t1 = pd.Series(idx, index=idx)  # 瞬間的ラベル
    test_times = pd.Series([idx[5]], index=[idx[5]])  # 末尾のみテスト
    trn = get_train_times(t1, test_times)
    # 末尾(5)のみが（df1で）除去され、残りは保持
    assert set(idx.get_indexer(trn.index)) == {0, 1, 2, 3, 4}


def test_embargo_single_block():
    idx = _index(10)
    banned = embargo_after(idx, np.array([3, 4, 5]), embargo=2)
    assert set(banned) == {6, 7}


def test_embargo_two_blocks():
    idx = _index(10)
    banned = embargo_after(idx, np.array([1, 2, 7]), embargo=1)
    assert set(banned) == {3, 8}


def test_embargo_zero_returns_empty():
    idx = _index(10)
    banned = embargo_after(idx, np.array([3, 4]), embargo=0)
    assert banned.size == 0


def test_embargo_at_tail_truncates():
    idx = _index(10)
    banned = embargo_after(idx, np.array([8, 9]), embargo=3)
    assert banned.size == 0  # 末尾以降に観測が無い
