"""信用・空売りデータのロードと派生ファクター化（分析層）。

data/jquants/{margin_weekly, short_ratio, short_positions, margin_alert}/ の
Parquetキャッシュを読み、point_in_time で整合可能な long 形式や派生ファクターを作る。
派生関数は純関数（渡したDataFrameに作用）でネットワーク不要・テスト可能。

派生ファクター（符号仮説は中立。研究側のサブ期間/DSRで検証する）：
  margin_imbalance   = (信用買残 − 信用売残)/(買残 + 売残)  … 信用需給の買い優勢度
  short_to_long      = 信用売残 / 信用買残
  short_interest     = 対発行株数の空売り残高比率（報告者合算, CalcDate基準）
  sector_short_ratio = 業種の空売り金額 / 総売り金額
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..data.sources import jquants as jq


def _concat_dir(name: str, base: Optional[str] = None) -> pd.DataFrame:
    """キャッシュ部分dirの全Parquetを連結（空マーカーはスキップ）。"""
    root = Path(base) if base is not None else jq._CACHE
    d = root / name
    frames = []
    if d.exists():
        for p in sorted(d.glob("*.parquet")):
            df = pd.read_parquet(p)
            if df.empty or "_empty" in df.columns:
                continue
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_weekly_margin(base: Optional[str] = None) -> pd.DataFrame:
    return _concat_dir("margin_weekly", base)


def load_short_ratio(base: Optional[str] = None) -> pd.DataFrame:
    return _concat_dir("short_ratio", base)


def load_short_positions(base: Optional[str] = None) -> pd.DataFrame:
    return _concat_dir("short_positions", base)


def load_margin_alert(base: Optional[str] = None) -> pd.DataFrame:
    return _concat_dir("margin_alert", base)


def margin_imbalance(weekly: pd.DataFrame) -> pd.DataFrame:
    """週次信用残高 → [Date, Code, margin_imbalance, short_to_long]。"""
    cols = ["Date", "Code", "margin_imbalance", "short_to_long"]
    if weekly.empty:
        return pd.DataFrame(columns=cols)
    df = weekly.copy()
    tot = (df["LongVol"] + df["ShrtVol"]).replace(0, np.nan)
    df["margin_imbalance"] = (df["LongVol"] - df["ShrtVol"]) / tot
    df["short_to_long"] = df["ShrtVol"] / df["LongVol"].replace(0, np.nan)
    return df[cols]


def short_interest(positions: pd.DataFrame) -> pd.DataFrame:
    """空売り残高報告 → [Date, Code, short_interest]（CalcDate・銘柄別に対SO比率を合算）。"""
    cols = ["Date", "Code", "short_interest"]
    if positions.empty:
        return pd.DataFrame(columns=cols)
    df = positions.copy()
    df["Date"] = df["CalcDate"]
    g = (df.groupby(["Date", "Code"], as_index=False)["ShrtPosToSO"].sum()
         .rename(columns={"ShrtPosToSO": "short_interest"}))
    return g[cols]


def sector_short_ratio(ratio: pd.DataFrame) -> pd.DataFrame:
    """業種別空売り比率 → [Date, S33, sector_short_ratio]（空売り金額/総売り金額）。"""
    cols = ["Date", "S33", "sector_short_ratio"]
    if ratio.empty:
        return pd.DataFrame(columns=cols)
    df = ratio.copy()
    shrt = df["ShrtWithResVa"].fillna(0.0) + df["ShrtNoResVa"].fillna(0.0)
    total = (df["SellExShortVa"].fillna(0.0) + shrt).replace(0, np.nan)
    df["sector_short_ratio"] = shrt / total
    return df[cols]
