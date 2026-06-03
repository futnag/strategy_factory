"""DSR / PSR / E[max SR] / minTRL の解析的性質を検証。"""
import numpy as np
import pytest

from invest_system.validation import dsr


def test_sharpe_ratio_basic():
    r = np.array([0.01, -0.01, 0.02, 0.0, 0.01])
    assert dsr.sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1))


def test_psr_half_when_sr_equals_benchmark():
    # sr == benchmark → z=0 → PSR=0.5
    assert dsr.probabilistic_sharpe_ratio(0.1, 0.1, 100, 0.0, 3.0) == pytest.approx(0.5)


def test_psr_increases_with_sample_length():
    short = dsr.probabilistic_sharpe_ratio(0.1, 0.0, 50, 0.0, 3.0)
    long = dsr.probabilistic_sharpe_ratio(0.1, 0.0, 500, 0.0, 3.0)
    assert 0.5 < short < long < 1.0


def test_psr_rejects_nonpositive_variance_term():
    # 大きな負の歪度と高い SR で分散項が非正にならないことを担保（ここは正常系）
    val = dsr.probabilistic_sharpe_ratio(0.1, 0.0, 100, -0.5, 5.0)
    assert 0.0 < val < 1.0


def test_expected_max_sharpe_zero_for_single_trial():
    assert dsr.expected_max_sharpe(1, 1.0) == 0.0


def test_expected_max_sharpe_scales_with_sigma():
    # sqrt(var) に比例：var を4倍 → 値は2倍
    base = dsr.expected_max_sharpe(20, 1.0)
    assert dsr.expected_max_sharpe(20, 4.0) == pytest.approx(2.0 * base)


def test_expected_max_sharpe_increases_with_trials():
    assert dsr.expected_max_sharpe(1000, 1.0) > dsr.expected_max_sharpe(10, 1.0)


def test_expected_max_sharpe_known_value():
    # 手計算の参照値：N=10, var=1 → ≈1.5745
    assert dsr.expected_max_sharpe(10, 1.0) == pytest.approx(1.5745, abs=0.02)


def test_dsr_not_greater_than_psr_against_zero():
    sr, var, n_obs = 0.12, 0.0009, 250
    d = dsr.deflated_sharpe_ratio(sr, var, 10, n_obs, 0.0, 3.0)
    psr0 = dsr.probabilistic_sharpe_ratio(sr, 0.0, n_obs, 0.0, 3.0)
    assert 0.0 <= d <= psr0 <= 1.0


def test_dsr_decreases_with_more_trials():
    sr, var, n_obs = 0.12, 0.0009, 250
    few = dsr.deflated_sharpe_ratio(sr, var, 2, n_obs, 0.0, 3.0)
    many = dsr.deflated_sharpe_ratio(sr, var, 1000, n_obs, 0.0, 3.0)
    assert few > many


def test_dsr_from_returns_matches_explicit():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, size=500)
    sr, sk, ku, n = dsr._moments(r)
    explicit = dsr.deflated_sharpe_ratio(sr, 0.0004, 16, n, sk, ku)
    conv = dsr.deflated_sharpe_ratio_from_returns(r, 0.0004, 16)
    assert conv == pytest.approx(explicit)


def test_min_track_record_length_roundtrip():
    sr, bench, sk, ku, prob = 0.1, 0.0, 0.0, 3.0, 0.95
    n = dsr.min_track_record_length(sr, bench, sk, ku, prob)
    # minTRL で評価した PSR は目標確率に一致する
    assert dsr.probabilistic_sharpe_ratio(sr, bench, n, sk, ku) == pytest.approx(prob)


def test_min_track_record_length_requires_sr_above_benchmark():
    with pytest.raises(ValueError):
        dsr.min_track_record_length(0.05, 0.05, 0.0, 3.0)
