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
    run_detector,
)
from invest_system.research.sector_regime.evaluation import evaluate_method, composite_score
from invest_system.research.sector_regime.config import PipelineConfig
from invest_system.research.sector_regime.pipeline import (
    _run_sector_methods,
    _spec_label,
    run_pipeline,
)


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
    assert len(specs) == 12  # HMM×3 + CPD×3 + GMM×3 + Markov×2 + BOCPD


def test_run_detector_all_specs(synthetic_daily):
    """パイプラインと同じ kwargs 渡しで全12手法が動作すること。"""
    weekly = build_weekly_features(synthetic_daily)
    feat = feature_matrix(weekly, "3300")
    cfg = PipelineConfig(warmup_weeks=52, refit_every=4)
    base_kw = {
        "warmup": cfg.warmup_weeks,
        "refit_every": cfg.refit_every,
        "window": cfg.rolling_window,
        "random_state": cfg.random_state,
    }
    for spec in all_method_specs():
        kw = {**base_kw, **spec.get("kwargs", {})}
        out = run_detector(spec["fn"], feat, sector="3300", config_kwargs=kw)
        valid = out.regime.dropna()
        assert len(valid) > 20, f"{out.method} produced insufficient output"


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


def test_spec_label():
    specs = all_method_specs()
    labels = {_spec_label(s) for s in specs}
    assert "hmm_2" in labels
    assert "cpd_pelt" in labels
    assert "gmm_3" in labels
    assert "markov_2" in labels
    assert "bocpd" in labels


def test_run_sector_methods_logs_failure(synthetic_daily, caplog):
    weekly = build_weekly_features(synthetic_daily)
    cfg = PipelineConfig(warmup_weeks=52, min_weeks=50)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("detector exploded")

    bad_spec = {"fn": _boom, "kwargs": {}}
    with caplog.at_level("WARNING", logger="invest_system.research.sector_regime.pipeline"):
        result = _run_sector_methods(
            "3300",
            weekly,
            cfg,
            [bad_spec],
            show_method_progress=True,
            global_task_offset=0,
            global_task_total=1,
        )

    assert result.results == []
    assert len(result.failures) == 1
    assert "detector exploded" in result.failures[0][1]
    assert any(r.levelname == "WARNING" for r in caplog.records)
    assert any("sector=3300" in r.message for r in caplog.records)


def test_pipeline_reports_method_counts():
    config = PipelineConfig(warmup_weeks=52, min_weeks=100)
    result = run_pipeline(
        "synthetic",
        sectors=["3300", "5250"],
        config=config,
        output_dir=None,
        show_progress=False,
    )
    assert result.method_successes > 0
    assert result.method_failures == 0
    assert "手法実行" in result.summary_text          # 失敗数が成果物（summary.md）に永続化される