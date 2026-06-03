"""CPCV の組合せ・カバレッジ・リーク遮断を検証。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.validation.cpcv import CombinatorialPurgedKFold


def _frame(n):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"f": np.arange(n)}, index=idx), idx


def test_path_and_split_counts():
    cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)
    assert cv.get_n_splits() == 15      # C(6,2)
    assert cv.get_n_paths() == 5        # C(5,1)


def test_invalid_params():
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(n_splits=4, n_test_splits=4)
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(n_splits=1, n_test_splits=1)
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo_pct=1.0)


def test_coverage_each_obs_in_test_exactly_n_paths():
    X, idx = _frame(60)
    t1 = pd.Series(idx, index=idx)  # 瞬間ラベル（パージ無し）
    cv = CombinatorialPurgedKFold(6, 2, 0.0)
    counts = np.zeros(60, dtype=int)
    n_splits = 0
    for train_idx, test_idx in cv.split(X, t1):
        n_splits += 1
        assert np.intersect1d(train_idx, test_idx).size == 0
        assert test_idx.size == 20
        counts[test_idx] += 1
    assert n_splits == cv.get_n_splits() == 15
    # 各観測はちょうど get_n_paths() 回テストに現れる
    assert (counts == cv.get_n_paths()).all()


def test_purged_kfold_reduction_k1():
    X, idx = _frame(50)
    t1 = pd.Series(idx, index=idx)
    cv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=1)
    assert cv.get_n_paths() == 1
    counts = np.zeros(50, dtype=int)
    n_splits = 0
    for train_idx, test_idx in cv.split(X, t1):
        n_splits += 1
        assert np.intersect1d(train_idx, test_idx).size == 0
        counts[test_idx] += 1
    assert n_splits == 5
    assert (counts == 1).all()  # 各観測はちょうど一度テストに入る


def test_no_label_leakage_with_purge_and_embargo():
    X, idx = _frame(60)
    span = pd.Timedelta(days=2)
    t1 = pd.Series(idx + span, index=idx)
    cv = CombinatorialPurgedKFold(6, 2, embargo_pct=0.1)  # embargo = 6
    t0v = idx.values
    t1v = t1.values
    for train_idx, test_idx in cv.split(X, t1):
        assert np.intersect1d(train_idx, test_idx).size == 0
        # どの訓練観測のラベル区間も、どのテスト観測のラベル区間とも重ならない
        for i in train_idx:
            for j in test_idx:
                overlap = (t0v[i] <= t1v[j]) and (t1v[i] >= t0v[j])
                assert not overlap


def test_split_requires_aligned_index():
    X, idx = _frame(30)
    bad_t1 = pd.Series(idx, index=pd.date_range("2021-01-01", periods=30, freq="D"))
    cv = CombinatorialPurgedKFold(6, 2)
    with pytest.raises(ValueError):
        list(cv.split(X, bad_t1))
