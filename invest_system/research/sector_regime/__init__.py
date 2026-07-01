"""セクター週足レジーム検知の比較・スコアリングパイプライン。"""
from .config import PipelineConfig, SCORE_WEIGHTS
from .data_loader import (
    generate_synthetic_sector_daily,
    get_data_root,
    load_sector_daily,
    list_sectors,
    set_data_root,
)
from .pipeline import PipelineResult, configure_logging, run_pipeline
from .weekly import build_weekly_features

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "SCORE_WEIGHTS",
    "build_weekly_features",
    "generate_synthetic_sector_daily",
    "get_data_root",
    "list_sectors",
    "load_sector_daily",
    "set_data_root",
    "configure_logging",
    "run_pipeline",
]