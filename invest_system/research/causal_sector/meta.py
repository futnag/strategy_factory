"""因果特徴量を value+PEAD メタラベルへ統合。

代表6業種版と全33業種版の2系統を提供。全33業種は kind 別エッジ3本を追加（docs/37）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    ALL_SECTORS, KIND_EDGE_FEATURES, REPRESENTATIVE_SECTORS, SECTOR_PROFILES,
)
from .panels import build_sector_daily_panel
from .regime import page_cusum, rolling_value_edge

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
CACHE_PATH = DATA_ROOT / "reports" / "causal_sector" / "meta_features_all33.parquet"

# 横断集約5本（docs/34/37 共通）
AGG_FEATURES = (
    "causal_edge", "edge_volatility", "edge_chg_3m",
    "months_since_break", "n_sectors_unstable",
)
# 全33業種版の追加3本（docs/37 凍結）
KIND_FEATURE_NAMES = tuple(f"edge_{k}" for k in KIND_EDGE_FEATURES)


def _adjustment_vars(panel: pd.DataFrame, s33: str) -> list[str]:
    """collider 除外の調整変数（最大4列）。"""
    prof = SECTOR_PROFILES.get(s33)
    coll = set(prof.collider_watch) if prof else set()
    return [c for c in panel.columns
            if c not in ("VALUE", "RET", *coll)][:4]


def _months_since_alarms(monthly_index: pd.DatetimeIndex,
                         alarms: list[pd.Timestamp]) -> pd.Series:
    """直近エッジ構造アラームからの経過月数。"""
    if not alarms:
        return pd.Series(np.nan, index=monthly_index)
    sorted_a = sorted(alarms)

    def _ms(t):
        past = [a for a in sorted_a if a <= t]
        if not past:
            return np.nan
        return (t - past[-1]).days / 30.0

    return pd.Series([_ms(t) for t in monthly_index], index=monthly_index)


def _monthly_causal_features_fast(s33: str, panel: pd.DataFrame) -> pd.DataFrame:
    """1業種の月次因果特徴（高速：全 detect_structural_breaks は使わない）。"""
    prof = SECTOR_PROFILES[s33]
    adj = _adjustment_vars(panel, s33)
    edge = rolling_value_edge(panel, window=180, step=42, adjustment_vars=adj)

    monthly_idx = panel.resample("ME").last().index
    monthly = pd.DataFrame(index=monthly_idx)
    monthly["causal_edge"] = edge.resample("ME").last()
    monthly["edge_volatility"] = edge.rolling(60).std().resample("ME").last()
    monthly["edge_chg_3m"] = monthly["causal_edge"].diff(3)

    alarms = page_cusum(edge.diff().dropna(), min_periods=30, h_sd=5.0)
    monthly["months_since_break"] = _months_since_alarms(monthly_idx, alarms)
    monthly["s33"] = s33
    monthly["kind"] = prof.kind
    return monthly


def build_sector_monthly_panel(s33: str) -> Optional[pd.DataFrame]:
    """1業種の月次因果パネル。失敗時 None。"""
    try:
        panel = build_sector_daily_panel(s33)
        if panel.dropna(subset=["VALUE", "RET"]).shape[0] < 200:
            return None
        return _monthly_causal_features_fast(s33, panel)
    except Exception:  # noqa: BLE001
        return None


def build_all_sector_monthly(*, sectors: Optional[list[str]] = None,
                             verbose: bool = False) -> pd.DataFrame:
    """全業種の月次パネルを long 形式で返す（s33, kind 列付き）。"""
    sectors = sectors or list(ALL_SECTORS)
    parts = []
    for i, s33 in enumerate(sectors):
        m = build_sector_monthly_panel(s33)
        if m is not None:
            parts.append(m)
        if verbose and (i + 1) % 8 == 0:
            print(f"  … {i + 1}/{len(sectors)} 業種処理済み")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts)


def _aggregate_monthly(long_m: pd.DataFrame, aggregate: str = "mean",
                       *, include_kind: bool = False) -> pd.DataFrame:
    """long 月次パネル → 横断集約5本（+ オプションで kind別3本）。"""
    feat_cols = ["causal_edge", "edge_volatility", "edge_chg_3m", "months_since_break"]
    agg = long_m.groupby(long_m.index)[feat_cols].agg(aggregate)

    med_vol = long_m.groupby(long_m.index)["edge_volatility"].median()
    n_unstable = []
    for dt, g in long_m.groupby(long_m.index):
        mv = med_vol.get(dt, np.nan)
        n_unstable.append((dt, int((g["edge_volatility"] > mv).sum())))
    agg["n_sectors_unstable"] = pd.Series(dict(n_unstable)).reindex(agg.index)

    if include_kind:
        kind_mean = (
            long_m.groupby([long_m.index, "kind"])["causal_edge"].mean().unstack("kind")
        )
        for kind in KIND_EDGE_FEATURES:
            col = f"edge_{kind}"
            agg[col] = kind_mean[kind] if kind in kind_mean.columns else np.nan
    return agg


def build_causal_meta_features(rebal_dates: pd.DatetimeIndex, *,
                               sectors: Optional[list[str]] = None,
                               aggregate: str = "mean",
                               use_cache: bool = False) -> pd.DataFrame:
    """因果メタ特徴量（横断集約）。

    sectors=None かつ use_cache → 全33業種キャッシュを利用。
    sectors=REPRESENTATIVE → 代表6業種（docs/34）。
    sectors=ALL_SECTORS → 全33業種（docs/37）。
    """
    if sectors is None:
        sectors = list(REPRESENTATIVE_SECTORS)

    if set(sectors) == set(ALL_SECTORS) and use_cache and CACHE_PATH.exists():
        cached = pd.read_parquet(CACHE_PATH)
        return cached.reindex(rebal_dates, method="ffill")

    long_m = build_all_sector_monthly(sectors=sectors)
    if long_m.empty:
        return pd.DataFrame(index=rebal_dates)

    include_kind = set(sectors) == set(ALL_SECTORS)
    out = _aggregate_monthly(long_m, aggregate, include_kind=include_kind)
    if include_kind:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(CACHE_PATH)
    return out.reindex(rebal_dates, method="ffill")


def build_causal_meta_features_all33(rebal_dates: pd.DatetimeIndex, *,
                                     verbose: bool = True,
                                     use_cache: bool = True) -> pd.DataFrame:
    """全33業種版因果メタ特徴量（8本：集約5 + kind3）。

    docs/37 凍結仕様。列:
      causal_edge, edge_volatility, edge_chg_3m, months_since_break,
      n_sectors_unstable, edge_heavy_industry, edge_it_comm, edge_financial
    """
    if use_cache and CACHE_PATH.exists():
        if verbose:
            print(f"  キャッシュ読込: {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH).reindex(rebal_dates, method="ffill")

    if verbose:
        print(f"  全{len(ALL_SECTORS)}業種の因果特徴量を構築中…")
    long_m = build_all_sector_monthly(sectors=list(ALL_SECTORS), verbose=verbose)
    if long_m.empty:
        return pd.DataFrame(index=rebal_dates)

    out = _aggregate_monthly(long_m, include_kind=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE_PATH)
    if verbose:
        print(f"  完了: {out.shape[1]}列 × {len(out)}月 → {CACHE_PATH}")
    return out.reindex(rebal_dates, method="ffill")


def feature_columns_rep6() -> list[str]:
    """代表6業種版の因果列名（5本）。"""
    return list(AGG_FEATURES)


def feature_columns_all33() -> list[str]:
    """全33業種版の因果列名（8本）。"""
    return list(AGG_FEATURES) + list(KIND_FEATURE_NAMES)