"""検証ファクトリ Phase1：判定器の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd

from invest_system.research.data_view import AsOfView
from invest_system.research.judge import judge_grid
from invest_system.research.strategy import CrossSectionalStrategy
from invest_system.validation.registry import TrialRegistry


def _noise_setup(k_strategies, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-31", periods=120, freq="ME")
    codes = [f"S{i}" for i in range(30)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.05, (120, 30)), axis=0),
        index=idx, columns=codes)
    view = AsOfView({"close": close})
    strategies = [
        CrossSectionalStrategy(
            pd.DataFrame(rng.normal(0, 1, (120, 30)), index=idx, columns=codes),
            quantile=0.2, name=f"noise{i}")
        for i in range(k_strategies)]
    return view, strategies


def test_judge_counts_trials_and_fails_on_noise():
    view, strategies = _noise_setup(8)
    with TrialRegistry(":memory:") as reg:
        v = judge_grid(strategies, view, scope="unit_noise",
                       hypothesis="pure noise, no a priori edge",
                       economic_rationale="none; randomly generated factors",
                       registry=reg, costs_bps=0.0)
    assert v.k == 8                       # 全試行が scope に計上
    assert len(v.results) == 8
    assert v.passed is False              # ノイズは多重検定後に通らない
    assert all(0.0 <= r.dsr <= 1.0 for r in v.results)
    assert "判定レポート" in v.report_md and "FAIL" in v.report_md


def test_judge_requires_a_priori_theory():
    # 仮説・経済的合理性が空（短すぎ）なら事前登録ゲートで弾かれる
    view, strategies = _noise_setup(2)
    import pytest
    with TrialRegistry(":memory:") as reg:
        with pytest.raises(ValueError):
            judge_grid(strategies, view, scope="x", hypothesis="x",
                       economic_rationale="y", registry=reg)


def test_more_trials_raise_the_bar():
    # 同一scopeで試行を増やすと E[maxSR] が上がり、最良DSRは上がりにくい（p-hack不能）
    v2, s2 = _noise_setup(3, seed=1)
    v20, s20 = _noise_setup(30, seed=1)
    with TrialRegistry(":memory:") as r2, TrialRegistry(":memory:") as r20:
        a = judge_grid(s2, v2, scope="s", hypothesis="noise test grid",
                       economic_rationale="random factors only", registry=r2,
                       costs_bps=0.0)
        b = judge_grid(s20, v20, scope="s", hypothesis="noise test grid",
                       economic_rationale="random factors only", registry=r20,
                       costs_bps=0.0)
    assert b.k == 30 and a.k == 3
    # 試行30の最良DSR は 試行3 の最良DSR を上回らない（デフレートが強まる）
    assert b.best.dsr <= a.best.dsr + 0.05
