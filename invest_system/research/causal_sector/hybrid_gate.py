"""OOS特化ハイブリッド因果メタゲート（docs/39）。

一次サイズは docs/25 の base_meta（凍結5特徴 walk-forward）をそのまま使い、
**不利な因果レジームが複数条件で同時に成立した月のみ** downscale する。
因果特徴量で ML サイズを全面置換しない＝IS期間の過剰縮小を防ぐ（docs/34/38 の教訓）。

設計原則（LdP）:
- underfit 許容：ルールベースの縮小は3条件 AND のみ（単一シグナルでは動かない）
- collider 回避：因果特徴量はサイズの「削減トリガー」にのみ使い、学習特徴量には入れない
- PIT：当月の因果特徴量のみ参照（≤t）
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..meta_gate import fit_meta_gate

# docs/39 凍結パラメータ
DEFAULT_SHRINK_MULT = 0.5
DEFAULT_UNSTABLE_THRESH = 5


def adverse_causal_regime(causal_feat: pd.DataFrame, *,
                          unstable_thresh: int = DEFAULT_UNSTABLE_THRESH
                          ) -> pd.Series:
    """不利因果レジームフラグ（bool Series, PIT）。

    3条件 AND（docs/39 §2.2 凍結）:
      1. causal_edge < 0        — 横断33業種で VALUE→RET エッジが負
      2. edge_financial < 0     — 金融セクターでもバリュー逆風（docs/38 分離最強）
      3. n_sectors_unstable >= 5 — 不安定業種が閾値以上（全33の~15%、事前固定）
    いずれか欠損の月は False（縮小しない＝保守的）。
    """
    cf = causal_feat.reindex(causal_feat.index)
    required = ["causal_edge", "edge_financial", "n_sectors_unstable"]
    if not all(c in cf.columns for c in required):
        return pd.Series(False, index=cf.index)
    ok = cf[required].notna().all(axis=1)
    adverse = (
        ok
        & (cf["causal_edge"] < 0)
        & (cf["edge_financial"] < 0)
        & (cf["n_sectors_unstable"] >= unstable_thresh)
    )
    return adverse.astype(bool)


def fit_hybrid_causal_gate(
    primary_net: pd.Series,
    base_features: pd.DataFrame,
    causal_features: pd.DataFrame, *,
    shrink_mult: float = DEFAULT_SHRINK_MULT,
    unstable_thresh: int = DEFAULT_UNSTABLE_THRESH,
    warmup: int = 36,
    embargo: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """base_meta サイズに因果不利レジーム時のみ縮小を乗じる。

    Returns
    -------
    hybrid_size : pd.Series  — MetaGatedStrategy に渡すスケール ∈ [0,1]
    adverse_flag : pd.Series   — 当月が不利因果レジームか（診断用）
    """
    if not (0.0 < shrink_mult <= 1.0):
        raise ValueError("shrink_mult must be in (0, 1]")
    base_size = fit_meta_gate(
        primary_net, base_features, warmup=warmup, embargo=embargo,
    )
    adverse = adverse_causal_regime(
        causal_features.reindex(base_size.index), unstable_thresh=unstable_thresh,
    )
    hybrid = base_size.copy()
    mask = adverse & hybrid.notna()
    if mask.any():
        hybrid.loc[mask] = np.clip(hybrid.loc[mask] * shrink_mult, 0.0, 1.0)
    return hybrid, adverse