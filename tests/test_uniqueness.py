"""サンプル独自性・重み・逐次ブートストラップを検証。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.sampling.uniqueness import (
    average_uniqueness,
    average_uniqueness_from_indicator,
    get_indicator_matrix,
    num_concurrent_events,
    sample_weights_by_return,
    sequential_bootstrap,
    time_decay,
)


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_concurrency_overlap():
    idx = _idx(10)
    # A: [idx0, idx3], B: [idx2, idx5]
    t1 = pd.Series([idx[3], idx[5]], index=[idx[0], idx[2]])
    conc = num_concurrent_events(idx, t1)
    assert [conc.loc[idx[i]] for i in range(6)] == [1, 1, 2, 2, 1, 1]


def test_average_uniqueness_overlap():
    idx = _idx(10)
    t1 = pd.Series([idx[3], idx[5]], index=[idx[0], idx[2]])
    u = average_uniqueness(idx, t1)
    # A の独自性 = mean(1,1,1/2,1/2) = 0.75（B も対称で 0.75）
    assert u.loc[idx[0]] == pytest.approx(0.75)
    assert u.loc[idx[2]] == pytest.approx(0.75)


def test_unique_event_uniqueness_is_one():
    idx = _idx(6)
    t1 = pd.Series([idx[2]], index=[idx[0]])      # 重なり無し
    u = average_uniqueness(idx, t1)
    assert u.loc[idx[0]] == pytest.approx(1.0)


def test_sample_weights_sum_to_n_and_positive():
    idx = _idx(20)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 20))), index=idx)
    t1 = pd.Series([idx[3], idx[7], idx[11]], index=[idx[0], idx[4], idx[8]])
    w = sample_weights_by_return(idx, t1, close)
    assert (w > 0).all()
    assert w.sum() == pytest.approx(len(w))


def test_time_decay_no_decay():
    u = pd.Series([0.5, 0.7, 0.9], index=_idx(3))
    d = time_decay(u, last_weight=1.0)
    assert np.allclose(d.to_numpy(), 1.0)


def test_time_decay_old_observations_downweighted():
    u = pd.Series([0.5, 0.7, 0.9], index=_idx(3))
    d = time_decay(u, last_weight=0.0)
    assert d.iloc[-1] == pytest.approx(1.0)
    assert d.iloc[0] < d.iloc[-1]
    assert (d >= 0).all()


def test_indicator_matrix_and_uniqueness_consistency():
    idx = _idx(10)
    t1 = pd.Series([idx[3], idx[5]], index=[idx[0], idx[2]])
    ind = get_indicator_matrix(idx, t1)
    assert ind.shape[1] == 2
    assert ind.iloc[:, 0].sum() == 4          # A: bars 0..3
    assert ind.iloc[:, 1].sum() == 4          # B: bars 2..5
    au = average_uniqueness_from_indicator(ind)
    assert au.iloc[0] == pytest.approx(0.75)
    assert au.iloc[1] == pytest.approx(0.75)


def test_sequential_bootstrap_mechanics_and_reproducible():
    idx = _idx(12)
    t1 = pd.Series([idx[2], idx[5], idx[8], idx[11]],
                   index=[idx[0], idx[3], idx[6], idx[9]])   # 重なり無し
    ind = get_indicator_matrix(idx, t1)
    draw = sequential_bootstrap(ind, size=4, random_state=0)
    assert len(draw) == 4
    assert all(0 <= d < 4 for d in draw)
    assert draw == sequential_bootstrap(ind, size=4, random_state=0)
