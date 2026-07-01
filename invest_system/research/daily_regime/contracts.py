"""daily_regime: シーム・データ契約（PIT 時系列スキーマ）。

接続設計 §2 の2契約をコード化する。契約は「コメントでなく**実行時に検証され、破れば
落ちる**」もの（釘②）：
  - WeeklyDirectionalAnchor … 週足filtered→日足事前注入の結合点。§1-4 ラグ（available_from）が宿る。
  - TierMembership          … 流動性ティアの PIT 所属（assigned_through ≤ date）。

検証失敗の例外を2種に分ける：
  - ContractViolation … スキーマ不充足（列・値域・Σ=1 等）。
  - LookAheadError    … 先読み不変条件の違反（橋渡し層・pooling index が落とす）。
"""
from __future__ import annotations

import pandas as pd


class ContractViolation(ValueError):
    """データ契約違反（スキーマ不充足）。"""


class LookAheadError(AssertionError):
    """先読み不変条件の違反。"""


# === 契約1: WeeklyDirectionalAnchor ======================================
ANCHOR_COLUMNS = (
    "week_end_date", "available_from", "anchor_id",
    "p_bear", "p_neutral", "p_bull", "regime",
    "anchor_source", "n_constituents",
)
ANCHOR_REGIMES = ("bear", "neutral", "bull")
ANCHOR_SOURCES = ("sector", "market")   # フォールバックB の切替フラグ（監査可能）


def validate_anchor(df: pd.DataFrame) -> pd.DataFrame:
    """WeeklyDirectionalAnchor のスキーマ＋ラグ規約を検証（OK ならそのまま返す）。"""
    missing = [c for c in ANCHOR_COLUMNS if c not in df.columns]
    if missing:
        raise ContractViolation(f"WeeklyDirectionalAnchor 列不足: {missing}")
    if df.empty:
        return df
    probs = df[["p_bear", "p_neutral", "p_bull"]].astype(float)
    if (probs.to_numpy() < -1e-9).any() or (probs.to_numpy() > 1 + 1e-9).any():
        raise ContractViolation("確率が [0,1] 外")
    if not ((probs.sum(axis=1) - 1.0).abs() < 1e-6).all():
        raise ContractViolation("確率ベクトル Σ≠1")
    if not df["regime"].isin(ANCHOR_REGIMES).all():
        raise ContractViolation(f"regime ラベル不正（許容: {ANCHOR_REGIMES}）")
    if not df["anchor_source"].isin(ANCHOR_SOURCES).all():
        raise ContractViolation(f"anchor_source 不正（許容: {ANCHOR_SOURCES}）")
    we = pd.to_datetime(df["week_end_date"])
    af = pd.to_datetime(df["available_from"])
    if not (af > we).all():
        raise ContractViolation(
            "available_from は week_end_date より後（公表ラグ＝§1-4）でなければならない"
        )
    return df


# === 契約2: TierMembership ===============================================
TIER_COLUMNS = ("date", "code", "tier", "assigned_through")
GATE_TIER = "gate_illiquid"


def tier_labels(n_tiers: int) -> tuple[str, ...]:
    """許容ティアラベル：gate_illiquid + T1..Tn。"""
    return (GATE_TIER,) + tuple(f"T{i}" for i in range(1, n_tiers + 1))


def validate_tiers(df: pd.DataFrame, *, n_tiers: int) -> pd.DataFrame:
    """TierMembership のスキーマ＋PIT 規約（assigned_through ≤ date）を検証。"""
    missing = [c for c in TIER_COLUMNS if c not in df.columns]
    if missing:
        raise ContractViolation(f"TierMembership 列不足: {missing}")
    if df.empty:
        return df
    valid = set(tier_labels(n_tiers))
    bad = set(pd.unique(df["tier"].dropna())) - valid
    if bad:
        raise ContractViolation(f"tier ラベル不正: {bad}（許容: {sorted(valid)}）")
    d = pd.to_datetime(df["date"])
    a = pd.to_datetime(df["assigned_through"])
    if not (a <= d).all():
        raise ContractViolation("assigned_through ≤ date を満たさない（PIT 違反）")
    return df
