"""日足 → 週足特徴量（リーケージなし）。

集計規則（すべて ≤ 当該週末時点の情報のみ）:
  - close_w: 週内最終営業日の終値
  - log_return_w: log(close_w / close_{w-1})
  - realized_vol_w: 週内日次対数リターンの標準偏差（年率化 √252）
  - parkinson_vol_w: 週内 Parkinson 推定量（high/low がある場合）
  - mom_20w: 20週前比リターン
  - rel_strength_w: セクターリターン − 全市場等加重リターン
  - volume_change_w: 週次出来高の前週比変化率
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import WEEKLY_FREQ


def _daily_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _parkinson_weekly(high: pd.Series, low: pd.Series) -> pd.Series:
    """週次 Parkinson ボラ（年率化）。"""
    ratio = np.log(high / low.replace(0, np.nan))
    daily_pk = ratio ** 2 / (4.0 * np.log(2.0))
    return np.sqrt(daily_pk.rolling(5, min_periods=1).mean() * 252.0)


def build_weekly_features(
    daily: pd.DataFrame,
    *,
    freq: str = WEEKLY_FREQ,
) -> pd.DataFrame:
    """全セクターの週足特徴量パネルを構築。

    Returns
    -------
    DataFrame
        MultiIndex (sector, date) または sector 列付きロング形式。
        本関数は sector 列付きロング形式を返す。
    """
    required = {"date", "sector", "close"}
    if not required.issubset(daily.columns):
        raise ValueError(f"必須列不足: {required - set(daily.columns)}")

    parts = []
    for sector, grp in daily.groupby("sector", sort=False):
        wk = _sector_daily_to_weekly(grp, freq=freq)
        wk["sector"] = sector
        parts.append(wk)
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["sector", "date"]).reset_index(drop=True)

    # 全市場等加重（相対強度用）— 各週・各セクターのリターンから算出（先読みなし）
    mkt = (
        out.groupby("date")["log_return_w"]
        .mean()
        .rename("mkt_log_return_w")
    )
    out = out.merge(mkt.reset_index(), on="date", how="left")
    out["rel_strength_w"] = out["log_return_w"] - out["mkt_log_return_w"]
    return out


def _sector_daily_to_weekly(grp: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    """1セクターの日足 → 週足。"""
    g = grp.sort_values("date").set_index("date")
    close = g["close"].astype(float)

    # 週次 OHLCV（週内最終営業日基準）
    w = pd.DataFrame({"close": close.resample(freq).last()})
    if "volume" in g.columns:
        w["volume"] = g["volume"].astype(float).resample(freq).sum()
    if "high" in g.columns:
        w["high"] = g["high"].astype(float).resample(freq).max()
    if "low" in g.columns:
        w["low"] = g["low"].astype(float).resample(freq).min()
    w = w.dropna(subset=["close"])

    # 日次リターン → 週内実現ボラ
    daily_lr = _daily_log_returns(close)
    rv = daily_lr.resample(freq).std() * np.sqrt(252.0)
    w["realized_vol_w"] = rv.reindex(w.index)

    if "high" in w.columns and "low" in w.columns:
        daily_h = g["high"].astype(float) if "high" in g.columns else close
        daily_l = g["low"].astype(float) if "low" in g.columns else close
        pk_daily = _parkinson_weekly(daily_h.reindex(close.index), daily_l.reindex(close.index))
        w["parkinson_vol_w"] = pk_daily.resample(freq).last()

    w["log_return_w"] = np.log(w["close"] / w["close"].shift(1))
    w["mom_20w"] = w["close"] / w["close"].shift(20) - 1.0

    if "volume" in w.columns:
        w["volume_change_w"] = w["volume"] / w["volume"].shift(1) - 1.0

    w = w.reset_index().rename(columns={"index": "date"})
    if "date" not in w.columns:
        w = w.rename(columns={w.columns[0]: "date"})
    return w


def feature_matrix(weekly: pd.DataFrame, sector: str) -> pd.DataFrame:
    """1セクターの検知用特徴量行列（NaN 除去済み）。"""
    sub = weekly[weekly["sector"] == sector].set_index("date").sort_index()
    cols = ["log_return_w", "realized_vol_w", "mom_20w", "rel_strength_w"]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].dropna(how="any")


def target_series(weekly: pd.DataFrame, sector: str) -> pd.DataFrame:
    """評価用ターゲット（翌週リターン・ボラ）。"""
    sub = weekly[weekly["sector"] == sector].set_index("date").sort_index()
    out = pd.DataFrame(index=sub.index)
    out["next_log_return_w"] = sub["log_return_w"].shift(-1)
    out["next_realized_vol_w"] = sub["realized_vol_w"].shift(-1)
    if "log_return_w" in sub.columns:
        out["next_return_sign"] = np.sign(out["next_log_return_w"])
    return out