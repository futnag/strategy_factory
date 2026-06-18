"""信用/空売りの公表日アンカー月次PIT特徴量（GKX 特徴量拡充 A-1・EDINET非依存）。

`margin.py` の派生（margin_imbalance/short_to_long/short_interest/sector_short_ratio）を
**公表日アンカー**で月次PITに乗せるための long ビルダー。基準日でなく「公開され利用可能に
なった日」で ≤t 採用する（EDINET 提出日アンカーと同一思想・handoff §3-2）：

- 週次信用残高：基準日(金曜)の**翌週第2営業日**に公表（JPX）＝基準日 **+2 取引営業日**。
- 大量空売り残高(short_positions)：**DiscDate（開示日）**が公表日そのもの（ラグ不要）。
- 業種別空売り比率：日次・翌営業日に利用可能＝**+1 取引営業日**（保守）。

取引カレンダーは Silver 日次（実取引日）から取る（暦日 +4 だと祝日週でズレるため）。材化
（月末 as-of・wide・float32・3 ビュー）は feature_store.build_holdings_features が
`fundamentals.point_in_time(date_col="available_date", lag_days=0)` を再利用して行う。

符号（handoff §3-3）：margin_imbalance/short_to_long/sector_short_ratio は **raw 維持**（符号
仮説は中立）、short_interest のみ空売りアノマリー（高 SI→低リターン）に沿って材化時に**負号**。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import margin as mg

WEEKLY_MARGIN_LAG = 2     # 基準金曜 → 翌週第2営業日に公表（JPX・事前固定）
SECTOR_SHORT_LAG = 1      # 業種別空売り比率は翌営業日に利用可能（保守）


def trading_days(base: Optional[str] = None) -> pd.DatetimeIndex:
    """Silver 日次（adj_close）の実取引日インデックス＝取引カレンダーの真値。"""
    from ..data.store import load_wide
    px = load_wide("adj_close", base=base or "data")
    return pd.DatetimeIndex(px.index).normalize() if not px.empty else pd.DatetimeIndex([])


def shift_business_days(dates, n: int, cal: pd.DatetimeIndex) -> pd.Series:
    """各日付の **n 取引営業日後**（厳密に後）の日付（公表ラグの付与）。cal 範囲外は NaT。

    例：基準金曜 + 2 → 翌週火曜（月=+1, 火=+2）。連休が挟まっても取引カレンダー追随で正しい。
    """
    d = pd.to_datetime(pd.Series(list(dates))).dt.normalize()
    arr = np.asarray(cal.values, dtype="datetime64[ns]")
    if len(arr) == 0:
        return pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    pos = np.searchsorted(arr, d.values, side="right") + (n - 1)   # 基準日より後の n 本目
    out = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    valid = pos < len(arr)
    out[valid] = pd.DatetimeIndex(arr[pos[valid]])
    return out


def weekly_margin_long(weekly: pd.DataFrame, cal: pd.DatetimeIndex,
                       lag: int = WEEKLY_MARGIN_LAG) -> pd.DataFrame:
    """週次信用 → long [available_date, Code, margin_imbalance, short_to_long, margin_balance_change]。

    available_date = 基準日 + lag 取引営業日。margin_balance_change = 信用残合計(買+売)の前週比。
    """
    cols = ["available_date", "Code", "margin_imbalance", "short_to_long",
            "margin_balance_change"]
    base = mg.margin_imbalance(weekly)            # [Date, Code, margin_imbalance, short_to_long]
    if base.empty:
        return pd.DataFrame(columns=cols)
    base["Date"] = pd.to_datetime(base["Date"]).dt.normalize()   # merge キーの型を揃える
    base["Code"] = base["Code"].astype(str)
    w = weekly.copy()
    w["Date"] = pd.to_datetime(w["Date"]).dt.normalize()
    w["bal"] = w["LongVol"] + w["ShrtVol"]
    w = w.sort_values(["Code", "Date"])
    w["margin_balance_change"] = w.groupby("Code")["bal"].pct_change()
    m = base.merge(w[["Date", "Code", "margin_balance_change"]], on=["Date", "Code"], how="left")
    m["Date"] = pd.to_datetime(m["Date"]).dt.normalize()
    m["Code"] = m["Code"].astype(str)
    m["available_date"] = shift_business_days(m["Date"], lag, cal).values
    return m.dropna(subset=["available_date"])[cols]


def short_position_long(positions: pd.DataFrame) -> pd.DataFrame:
    """大量空売り残高 → long [available_date(=DiscDate), Code, short_interest, short_shares]。

    DiscDate は公表日そのもの（ラグ不要）。報告者を銘柄別に合算。short_interest は対 SO 比率、
    short_shares は残高株数（days_to_cover 用）。符号は materialize 時に負号化（short_interest）。
    """
    cols = ["available_date", "Code", "short_interest", "short_shares"]
    if positions.empty:
        return pd.DataFrame(columns=cols)
    df = positions.copy()
    df["available_date"] = pd.to_datetime(df["DiscDate"]).dt.normalize()
    df["Code"] = df["Code"].astype(str)
    g = df.groupby(["available_date", "Code"], as_index=False).agg(
        short_interest=("ShrtPosToSO", "sum"), short_shares=("ShrtPosShares", "sum"))
    return g[cols]


def sector_short_long(ratio: pd.DataFrame, cal: pd.DatetimeIndex,
                      lag: int = SECTOR_SHORT_LAG) -> pd.DataFrame:
    """業種別空売り比率 → long [available_date, S33, sector_short_ratio]（翌営業日に利用可能）。

    材化側で S33 → 構成銘柄へブロードキャスト（sector-PIT 限界は docs/16 に明記）。
    """
    cols = ["available_date", "S33", "sector_short_ratio"]
    base = mg.sector_short_ratio(ratio)           # [Date, S33, sector_short_ratio]
    if base.empty:
        return pd.DataFrame(columns=cols)
    base = base.copy()
    base["Date"] = pd.to_datetime(base["Date"]).dt.normalize()
    base["available_date"] = shift_business_days(base["Date"], lag, cal).values
    return base.dropna(subset=["available_date"])[cols]
