"""PBO（CSCV・Bailey et al. 2016）と MinBTL の検証。ネット不要・決定論的。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.validation import min_backtest_length, pbo_cscv


def _grid(n_obs, n_strat, seed=0, edge_col=None, edge_mu=0.05):
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0, 0.1, (n_obs, n_strat))
    if edge_col is not None:
        m[:, edge_col] += edge_mu
    idx = pd.date_range("2016-01-31", periods=n_obs, freq="ME")
    return pd.DataFrame(m, index=idx, columns=[f"s{i}" for i in range(n_strat)])


def test_pbo_noise_grid_is_near_half():
    # 純ノイズのグリッド：IS最良はランダム → OOS順位は一様 → PBO ≈ 0.5
    r = pbo_cscv(_grid(240, 10, seed=1), n_splits=8)
    assert r.n_combinations == 70                  # C(8,4)
    assert 0.25 <= r.pbo <= 0.75
    assert len(r.logits) == 70


def test_pbo_true_edge_is_low():
    # 1構成だけ真のエッジ（μ=0.05, σ=0.1 → SR≈0.5/月）→ IS最良がOOSでも上位
    r = pbo_cscv(_grid(240, 10, seed=2, edge_col=3), n_splits=8)
    assert r.pbo < 0.2


def test_pbo_deterministic_and_guards():
    g = _grid(240, 5, seed=3)
    a, b = pbo_cscv(g), pbo_cscv(g)
    assert a.pbo == b.pbo and np.array_equal(a.logits, b.logits)  # 乱数不使用
    assert np.isnan(pbo_cscv(g[["s0"]]).pbo)       # N<2 は判定不能＝NaN
    assert np.isnan(pbo_cscv(g.iloc[:10]).pbo)     # 標本不足も NaN
    with pytest.raises(ValueError):
        pbo_cscv(g, n_splits=7)                    # S は偶数のみ


def test_min_backtest_length_grows_with_trials():
    assert min_backtest_length(1) == 0.0           # 1試行は選択バイアスなし
    b10, b100 = min_backtest_length(10), min_backtest_length(100)
    assert 0 < b10 < b100                          # K が増えるほど必要標本が伸びる
    # target を半分にすると必要年数は4倍（2乗則）
    assert min_backtest_length(10, 0.5) == pytest.approx(4 * b10)
    with pytest.raises(ValueError):
        min_backtest_length(10, 0.0)


def test_judge_report_shows_pbo_and_minbtl():
    from invest_system.research.data_view import AsOfView
    from invest_system.research.judge import judge_grid
    from invest_system.research.strategy import CrossSectionalStrategy
    from invest_system.validation.registry import TrialRegistry

    rng = np.random.default_rng(4)
    idx = pd.date_range("2016-01-31", periods=120, freq="ME")
    codes = [f"S{i}" for i in range(30)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.05, (120, 30)), axis=0),
        index=idx, columns=codes)
    strategies = [
        CrossSectionalStrategy(
            pd.DataFrame(rng.normal(0, 1, (120, 30)), index=idx, columns=codes),
            quantile=0.2, name=f"noise{i}") for i in range(4)]
    with TrialRegistry(":memory:") as reg:
        v = judge_grid(strategies, AsOfView({"close": close}), scope="unit_pbo",
                       hypothesis="pure noise, no a priori edge",
                       economic_rationale="none; randomly generated factors",
                       registry=reg, costs_bps=0.0)
    assert "PBO(CSCV)" in v.report_md and "MinBTL" in v.report_md
    assert np.isnan(v.pbo) or 0.0 <= v.pbo <= 1.0
    assert v.min_btl_years > 0                     # K=4 > 1
    assert v.passed is False                       # 判定は従来どおり DSR のみ
