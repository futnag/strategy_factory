"""33業種セクター別の因果的レジーム検知（López de Prado 原則）。

- PCMCI+（tigramite）で因果グラフを学習し collider を同定
- 因果エッジ強度の structural break を監視（パフォーマンスブレイクより優先）
- DoWhy + EconML で backdoor adjustment / CATE 推定
- 既存 value+PEAD メタラベルへの因果特徴量統合
"""
from .config import SECTOR_PROFILES, SectorProfile, collider_candidates
from .panels import build_sector_daily_panel, sector_data_quality_report
from .graph import learn_pcmci_graph, identify_colliders, graph_summary
from .regime import detect_structural_breaks, compare_detection_methods
from .effects import estimate_causal_effect, run_refutation_tests
from .hybrid_gate import adverse_causal_regime, fit_hybrid_causal_gate
from .meta import (
    build_causal_meta_features, build_causal_meta_features_all33,
    feature_columns_all33, feature_columns_rep6,
)
from .validate import causal_regime_cv_report
from .monitor import (
    build_monitor_snapshot, format_monitor_md, write_monitor_report,
)

__all__ = [
    "SECTOR_PROFILES",
    "SectorProfile",
    "collider_candidates",
    "build_sector_daily_panel",
    "sector_data_quality_report",
    "learn_pcmci_graph",
    "identify_colliders",
    "graph_summary",
    "detect_structural_breaks",
    "compare_detection_methods",
    "estimate_causal_effect",
    "run_refutation_tests",
    "build_causal_meta_features",
    "build_causal_meta_features_all33",
    "feature_columns_all33",
    "feature_columns_rep6",
    "adverse_causal_regime",
    "fit_hybrid_causal_gate",
    "causal_regime_cv_report",
    "build_monitor_snapshot",
    "format_monitor_md",
    "write_monitor_report",
]