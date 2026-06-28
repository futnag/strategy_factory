"""セクターレジーム比較パイプラインのテスト。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from invest_system.research.sector_regime.data_loader import (
    generate_synthetic_sector_daily,
    list_sectors,
)
from invest_system.research.sector_regime.weekly import build_weekly_features, feature_matrix
from invest_system.research.sector_regime.detectors import (
    walk_forward_hmm,
    walk_forward_bocpd,
    all_method_specs,
)
from invest_system.research.sector_regime.evaluation import evaluate_method, composite_score
from invest_system.research.sector_regime.config import PipelineConfig
from invest_system.research.sector_regime.pipeline import run_pipeline


@pytest.fixture
def synthetic_daily():
    return generate_synthetic_sector_daily(
        sectors=["3300", "5250"],
        n_days=1500,
        seed=42,
    )


def test_weekly_no_future_leak(synthetic_daily):
    """週次リターンが当該週の close のみで決まること。"""
    weekly = build_weekly_features(synthetic_daily)
    sub = weekly[weekly["sector"] == "3300"].set_index("date").sort_index()
    manual = np.log(sub["close"] / sub["close"].shift(1))
    np.testing.assert_allclose(
        sub["log_return_w"].dropna().values,
        manual.dropna().values,
        rtol=1e-10,
    )


def test_weekly_features_columns(synthetic_daily):
    weekly = build_weekly_features(synthetic_daily)
    assert "log_return_w" in weekly.columns
    assert "realized_vol_w" in weekly.columns
    assert "mom_20w" in weekly.columns
    assert "rel_strength_w" in weekly.columns


def test_hmm_walk_forward(synthetic_daily):
    weekly = build_weekly_features(synthetic_daily)
    feat = feature_matrix(weekly, "3300")
    out = walk_forward_hmm(feat, sector="3300", n_states=2, warmup=52, refit_every=4)
    valid = out.regime.dropna()
    assert len(valid) > 50
    assert valid.isin([0, 1]).all()


def test_bocpd_walk_forward(synthetic_daily):
    weekly = build_weekly_features(synthetic_daily)
    feat = feature_matrix(weekly, "3300")
    out = walk_forward_bocpd(feat, sector="3300", warmup=52)
    valid = out.regime.dropna()
    assert len(valid) > 50


def test_evaluation_scores(synthetic_daily):
    weekly = build_weekly_features(synthetic_daily)
    feat = feature_matrix(weekly, "3300")
    out = walk_forward_hmm(feat, sector="3300", n_states=2, warmup=52)
    met = evaluate_method(out, weekly, feat, skip_stability=True)
    assert 0.0 <= met["composite_score"] <= 1.0
    assert "pred_score" in met
    assert "quality_score" in met


def test_all_method_specs_count():
    specs = all_method_specs()
    assert len(specs) >= 10  # HMM×3 + CPD×3 + GMM×3 + Markov×2 + BOCPD


def test_pipeline_synthetic_runs():
    config = PipelineConfig(warmup_weeks=52, min_weeks=100)
    result = run_pipeline(
        "synthetic",
        sectors=["3300", "5250"],
        config=config,
        output_dir=None,
        show_progress=False,
    )
    assert not result.ranking.empty
    assert not result.best.empty
    assert len(result.best) == 2


def test_list_sectors(synthetic_daily):
    assert set(list_sectors(synthetic_daily)) == {"3300", "5250"}