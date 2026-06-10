"""ケリー基準（連続版・フラクショナル）の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.portfolio import kelly_fraction, kelly_weights


def _monthly_returns():
    idx = pd.date_range("2016-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.01, 0.04, len(idx)), index=idx)


def test_kelly_fraction_closed_form():
    r = _monthly_returns()
    res = kelly_fraction(r, fraction=0.5)
    mu, var = float(r.mean()), float(r.var(ddof=1))
    assert res.f_full == pytest.approx(mu / var)
    assert res.f == pytest.approx(0.5 * mu / var)
    assert 11.5 < res.ann_factor < 12.5            # 月次を自動推定
    # 期待対数成長率 g(f)=fμ−f²σ²/2（年率換算）。満額は g=μ²/2σ²。
    assert res.growth_ann_full == pytest.approx(
        (mu * mu / (2 * var)) * res.ann_factor)
    g_half = res.f * mu - 0.5 * res.f ** 2 * var
    assert res.growth_ann_frac == pytest.approx(g_half * res.ann_factor)
    # ハーフケリーの成長率は満額の 3/4（連続版の恒等式）
    assert res.growth_ann_frac == pytest.approx(0.75 * res.growth_ann_full)


def test_kelly_fraction_negative_mean_says_do_not_bet():
    r = -_monthly_returns().abs()
    res = kelly_fraction(r)
    assert res.f_full < 0 and res.f < 0


def test_kelly_fraction_requires_ann_factor_without_dates():
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.04, 40))
    with pytest.raises(ValueError):
        kelly_fraction(r)
    res = kelly_fraction(r, ann_factor=12.0)
    assert np.isfinite(res.f)


def test_kelly_fraction_rejects_short_series():
    idx = pd.date_range("2024-01-31", periods=5, freq="ME")
    with pytest.raises(ValueError):
        kelly_fraction(pd.Series(0.01, index=idx))


def test_kelly_weights_diagonal_matches_univariate():
    mu = pd.Series({"s1": 0.01, "s2": 0.02})
    cov = pd.DataFrame(np.diag([0.04 ** 2, 0.08 ** 2]),
                       index=mu.index, columns=mu.index)
    f = kelly_weights(mu, cov, fraction=1.0)
    assert f["s1"] == pytest.approx(0.01 / 0.04 ** 2)
    assert f["s2"] == pytest.approx(0.02 / 0.08 ** 2)
    half = kelly_weights(mu, cov, fraction=0.5)
    assert half["s1"] == pytest.approx(0.5 * f["s1"])
    assert list(half.index) == ["s1", "s2"]        # ラベル保持
