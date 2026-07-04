"""KWZ スパニング実装の検証（閉形式 vs シミュレーション・論文アンカー値）。"""
import numpy as np
import pytest

from invest_system.validation.spanning import (
    breakeven_theta, expected_oos_sharpe, grs_shifted_test, simulate_sharpes,
    spanning_report,
)


def test_closed_form_matches_simulation():
    """E[θ̃] の閉形式（eq.12）と Prop.1 シミュレーションの一致（調査時の数値検証の再現）。"""
    rng = np.random.default_rng(7)
    for n, t, theta in [(2, 120, 0.30), (2, 120, 0.15), (6, 120, 0.30)]:
        _, oos = simulate_sharpes(n, t, theta, n_sims=400_000, rng=rng)
        assert expected_oos_sharpe(n, t, theta) == pytest.approx(oos.mean(), abs=2e-3)


def test_survey_anchor_values():
    """調査レポート（2026-07-04 §A T1）の検証済みアンカー値。"""
    assert expected_oos_sharpe(2, 120, 0.30) == pytest.approx(0.284, abs=0.005)
    assert expected_oos_sharpe(2, 120, 0.15) == pytest.approx(0.116, abs=0.005)
    assert expected_oos_sharpe(6, 120, 0.30) == pytest.approx(0.239, abs=0.005)


def test_is_biased_up_oos_below_theta():
    """E[θ̂_IS] > θ（過大）・E[θ̃_OOS] < θ（推定リスクの haircut）。"""
    theta = 0.20
    is_, oos = simulate_sharpes(2, 120, theta, n_sims=200_000,
                                rng=np.random.default_rng(1))
    assert is_.mean() > theta
    assert oos.mean() < theta


def test_breakeven_table4_anchor():
    """KWZ Table 4 基準ケース: θ₁=0.10, T=120, N=2, c=0.5 → θ_b≈0.162（+62%）。"""
    tb = breakeven_theta(0.10, 2, 120, c=0.5, n_sims=150_000)
    assert tb == pytest.approx(0.162, abs=0.01)


def test_breakeven_monotone_in_T():
    """T が長いほど推定リスクが減り break-even は下がる。"""
    tb_short = breakeven_theta(0.10, 2, 90, n_sims=80_000)
    tb_long = breakeven_theta(0.10, 2, 240, n_sims=80_000)
    assert tb_long < tb_short


def test_grs_shifted_basic():
    """増分ゼロなら p≈1 側・大きな増分なら p が小さくなる方向性。"""
    _, p_null = grs_shifted_test(0.10 ** 2, 0.10 ** 2, 2, 120, 0.01)
    _, p_big = grs_shifted_test(0.40 ** 2, 0.10 ** 2, 2, 120, 0.01)
    assert p_null > 0.5
    assert p_big < p_null


def test_report_smoke():
    rep = spanning_report(0.10, 0.18, 0.11, 120)
    assert rep["theta_breakeven"] > 0.10
    assert 0 < rep["grs_p"] <= 1
    assert "表示・提案専用" in rep["note"]
