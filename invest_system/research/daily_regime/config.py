"""daily_regime: 設定定数（step 0–1 範囲）。

接続設計 regime_detection_connection_v2.1.md の確定アーキを反映：
  - 流動性ティア：Amihud 主・ADV 副ゲート（二段）。ティア数・床は**固定・非最適化**。
  - 方向性アンカー：業種指数（S33 代理）。フォールバックB の N も**固定・非最適化**。
"""
from __future__ import annotations

from dataclasses import dataclass

RANDOM_STATE = 42

# --- 流動性ティア（Amihud 主・ADV 副ゲート） ---
N_TIERS = 3                      # Amihud 横断分位ティア数（T1=最流動 … Tn=最非流動）。固定。
ADV_FLOOR_JPY = 5e7              # ADV ゲート床（¥50M。universe "production" と整合）。固定。
LIQ_WINDOW = 60                  # Amihud/ADV の trailing 窓（営業日。price_factors §4 と整合）。

# 帰属は「当日生Amihud」でなく**平滑化した流動性水準**で行う（統計的交絡の回避）。
# 生Amihudはストレス時に跳ねるため、生値で括ると平常流動な銘柄がストレス当日に非流動
# ティアへ誤再ランクされ、(i) プールが構造的流動性ピアでなくなる、(ii) ティア日替わりで
# filtered確率が跳ね sticky/min-dwell を損なう、(iii) 帰属が検知対象ストレスと共動する。
# → トレーリング中央値で水準化し、refit 時のみ再計算して refit 間は固定（sticky pool）。
TIER_SMOOTH_WINDOW = 60          # 流動性水準の平滑窓（トレーリング中央値）。固定・非最適化。
TIER_REFIT_EVERY = 20            # ティア再計算間隔（営業日）。refit 間は固定＝sticky。固定。
TIER_LEVEL_KIND = "amihud_median"  # "amihud_median"（Amihud主）| "turnover_median"

# --- 週足方向性アンカー（業種指数・フォールバックB） ---
WEEKLY_FREQ = "W-FRI"            # 日本市場：金曜締め
ANCHOR_MOM_WEEKS = 20           # 方向性モメンタム窓（週）。⚠スタブ整列キー（本HMMは step 5）。
ANCHOR_NEUTRAL_BAND = 0.02      # |mom| < band → neutral
FALLBACK_MIN_CONSTITUENTS = 10  # フォールバックB：構成銘柄 < N で広域（等加重マーケット）代理へ。固定・非最適化。
PUBLISH_LAG_BDAYS = 1           # 週足アンカー公表ラグ（§1-4）＝完了週末の翌営業日から可用。


@dataclass(frozen=True)
class DailyRegimeConfig:
    """step 0–1 のシーム配線パラメータ（単一 config）。"""
    n_tiers: int = N_TIERS
    adv_floor_jpy: float = ADV_FLOOR_JPY
    liq_window: int = LIQ_WINDOW
    tier_smooth_window: int = TIER_SMOOTH_WINDOW
    tier_refit_every: int = TIER_REFIT_EVERY
    tier_level_kind: str = TIER_LEVEL_KIND
    weekly_freq: str = WEEKLY_FREQ
    anchor_mom_weeks: int = ANCHOR_MOM_WEEKS
    anchor_neutral_band: float = ANCHOR_NEUTRAL_BAND
    fallback_min_constituents: int = FALLBACK_MIN_CONSTITUENTS
    publish_lag_bdays: int = PUBLISH_LAG_BDAYS
    random_state: int = RANDOM_STATE
