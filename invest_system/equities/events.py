"""イベント系シグナル（決算サプライズ / 会社予想の改訂 = PEAD）。

会社予想(FEPS 等)の上方修正・実績の予想超過は、将来リターンの正のシグナルとして
知られる（PEAD／ガイダンス改訂アノマリー）。開示日 DiscDate 基準で point_in_time
整合できる long 形式を返す。純関数（ネットワーク不要・テスト可能）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_revision(fund_long: pd.DataFrame, field: str = "FEPS",
                      date_col: str = "DiscDate", code_col: str = "Code"
                      ) -> pd.DataFrame:
    """会社予想 field の改訂率（銘柄ごとに開示順 pct_change）→ [Code, DiscDate, fcst_revision]。"""
    cols = [code_col, date_col, "fcst_revision"]
    if fund_long.empty or field not in fund_long.columns:
        return pd.DataFrame(columns=cols)
    df = fund_long.dropna(subset=[date_col]).copy()
    df[field] = pd.to_numeric(df[field], errors="coerce")   # 予想列は文字列のことがある
    df = df.dropna(subset=[field]).sort_values([code_col, date_col])
    df = df[~df.duplicated([code_col, date_col], keep="last")]
    prev = df.groupby(code_col)[field].shift(1)
    df["fcst_revision"] = (df[field] - prev) / prev.abs().replace(0, np.nan)
    return df.dropna(subset=["fcst_revision"])[cols]


def earnings_surprise(fund_long: pd.DataFrame, actual: str = "EPS",
                      forecast: str = "FEPS", date_col: str = "DiscDate",
                      code_col: str = "Code") -> pd.DataFrame:
    """実績 actual の対・直近予想 forecast 乖離（同一開示行内）→ [Code, DiscDate, surprise]。

    surprise = (実績 − 直近予想)/|直近予想|。予想超過(正)は PEAD の正シグナル。
    ※ /fins/summary は実績(累計)と通期予想が混在し得るため、厳密な四半期SUEには
       追加整形が要る。本関数は素朴な同一行比較（MVP）。
    """
    cols = [code_col, date_col, "surprise"]
    if fund_long.empty or actual not in fund_long.columns or \
            forecast not in fund_long.columns:
        return pd.DataFrame(columns=cols)
    df = fund_long.dropna(subset=[date_col]).copy()
    df[actual] = pd.to_numeric(df[actual], errors="coerce")
    df[forecast] = pd.to_numeric(df[forecast], errors="coerce")
    df = df.dropna(subset=[actual, forecast])
    df["surprise"] = (df[actual] - df[forecast]) / df[forecast].abs().replace(0, np.nan)
    return df.dropna(subset=["surprise"])[cols]
