"""セクターレジーム比較パイプラインの設定定数。"""
from __future__ import annotations

from dataclasses import dataclass

# 再現性
RANDOM_STATE = 42

# 週足集計（日本市場：金曜締め）
WEEKLY_FREQ = "W-FRI"

# ウォークフォワード既定
DEFAULT_WARMUP_WEEKS = 104   # 約2年
DEFAULT_REFIT_EVERY = 4      # 4週ごとに再学習
DEFAULT_ROLLING_WINDOW = 156 # 約3年（ローリング窓用）

# HMM 状態数候補
HMM_N_STATES = (2, 3, 4)

# 変化点検知アルゴリズム
CPD_METHODS = ("pelt", "binseg", "bottomup")

# GMM クラスタ数候補
GMM_N_COMPONENTS = (2, 3, 4)

# 複合スコアの重み（合計=1.0）
# 根拠:
#   - 予測力(0.40): 本分析の主目的は「翌週リターン/ボラの予測」
#   - レジーム質(0.25): 状態が経済的に識別可能でなければ予測も不安定
#   - 安定性(0.20): ウォークフォワード一貫性は実運用の信頼性に直結
#   - 経済的意味(0.15): 統計的有意差は解釈可能性の担保
SCORE_WEIGHTS = {
    "prediction": 0.40,
    "regime_quality": 0.25,
    "stability": 0.20,
    "economic": 0.15,
}

# セクター特性分類（考察用）
DEFENSIVE_SECTORS = frozenset({
    "3050", "3250", "4050", "7050", "7150", "8050",
})
CYCLICAL_SECTORS = frozenset({
    "3300", "3400", "3450", "3500", "3600", "3700", "5100",
})


@dataclass(frozen=True)
class PipelineConfig:
    """パイプライン実行パラメータ。"""
    warmup_weeks: int = DEFAULT_WARMUP_WEEKS
    refit_every: int = DEFAULT_REFIT_EVERY
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    random_state: int = RANDOM_STATE
    n_jobs: int = -1
    min_weeks: int = 260  # 最低約5年