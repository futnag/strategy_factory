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


def dividend_forecast_revision(fund_long: pd.DataFrame,
                               adj_cum: pd.DataFrame | None = None,
                               field: str = "FDivAnn",
                               date_col: str = "DiscDate",
                               code_col: str = "Code") -> pd.DataFrame:
    """予想年間配当（1株）の改訂率（**株式分割調整済み**）→ [Code, DiscDate, div_revision]。

    配当は経営者のコミットメント・シグナル（増配サプライズ＝株主還元イベント）。1株配当は
    分割・併合で機械的に変わるため、adj_cum（日次 AdjFactor の累積積 wide＝
    `load_wide("adj_factor").where(notna,1).cumprod()`）を与えると、前回開示値を
    「2開示日間に適用された係数 cum[d1]/cum[d0]」倍して同一ベースで比較する
    （2:1 分割→係数0.5→配当半減は改訂率 0 と正しく判定）。None なら無調整
    （既存 forecast_revision と同じ素朴比較）。
    """
    cols = [code_col, date_col, "div_revision"]
    if fund_long.empty or field not in fund_long.columns:
        return pd.DataFrame(columns=cols)
    df = fund_long.dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[field] = pd.to_numeric(df[field], errors="coerce")
    df = df.dropna(subset=[field]).sort_values([code_col, date_col])
    df = df[~df.duplicated([code_col, date_col], keep="last")]
    parts = []
    for code, g in df.groupby(code_col):
        s = g[field]
        prev = s.shift(1)
        if adj_cum is not None and str(code) in adj_cum.columns:
            cum = adj_cum[str(code)]
            pos = cum.index.searchsorted(g[date_col].to_numpy(), side="right") - 1
            cumv = np.where(pos >= 0, cum.to_numpy()[np.clip(pos, 0, None)], 1.0)
            ratio = np.divide(cumv, np.roll(cumv, 1),
                              out=np.ones_like(cumv), where=np.roll(cumv, 1) != 0)
            ratio[0] = 1.0
            prev = prev * ratio                      # 前回値を現在の株数ベースへ
        rev = (s - prev) / prev.abs().replace(0, np.nan)
        out = g[[code_col, date_col]].copy()
        out["div_revision"] = rev
        parts.append(out)
    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    return res.dropna(subset=["div_revision"])[cols]


def buyback_intensity(fund_long: pd.DataFrame, date_col: str = "DiscDate",
                      code_col: str = "Code") -> pd.DataFrame:
    """自己株比率（TrShFY/ShOutFY）の開示順差分 → [Code, DiscDate, buyback]。

    正＝自己株式の取得（実現した自社株買い）、負＝新株発行等の希薄化。**比率なので
    株式分割・併合に不変**（分子分母が同倍率）。限界：自己株消却は比率を下げる方向に
    出る（株主還元なのに負号）が、消却は取得の後段で頻度も低く、シグナルの主成分は
    取得/発行と割り切る。発行済株式数（ShOutFY）は自己株を含む総数。
    """
    cols = [code_col, date_col, "buyback"]
    need = {code_col, date_col, "TrShFY", "ShOutFY"}
    if fund_long.empty or not need.issubset(fund_long.columns):
        return pd.DataFrame(columns=cols)
    df = fund_long.dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    tr = pd.to_numeric(df["TrShFY"], errors="coerce")
    sh = pd.to_numeric(df["ShOutFY"], errors="coerce")
    df["_ratio"] = tr / sh.where(sh > 0)
    df = df.dropna(subset=["_ratio"]).sort_values([code_col, date_col])
    df = df[~df.duplicated([code_col, date_col], keep="last")]
    df["buyback"] = df.groupby(code_col)["_ratio"].diff()
    return df.dropna(subset=["buyback"])[cols]


def expected_announcement_month(fund_long: pd.DataFrame, rebal_dates,
                                date_col: str = "DiscDate",
                                code_col: str = "Code") -> pd.DataFrame:
    """各リバランス日 t で「翌月に決算発表が見込まれる銘柄」を True にするマスク。

    決算発表予定(earnings-calendar)は前向きのみのため、過去の発表日 DiscDate から各銘柄の
    「発表月パターン」を作り、翌月(t+1)の暦月がそのパターンに含まれれば発表見込みとする。
    会計カレンダーは年次で安定するため、月パターン自体は構造的属性として扱う（リターンには
    先読みを持ち込まない）。初開示前の銘柄は False（PIT）。

    Returns: bool DataFrame（index=リバランス日, columns=Code）。
    """
    rebal = pd.DatetimeIndex(sorted(pd.to_datetime(list(rebal_dates))))
    if fund_long.empty or date_col not in fund_long.columns:
        return pd.DataFrame(False, index=rebal, columns=[])
    nxt = pd.Series([(t + pd.offsets.MonthBegin(1)).month for t in rebal], index=rebal)
    df = fund_long.dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    out = {}
    for code, g in df.groupby(code_col):
        # 四半期決算の規則的な発表月＝最頻4か月（予想修正等の散発開示月を除く）
        months = set(int(m) for m in g[date_col].dt.month.value_counts().head(4).index)
        first = g[date_col].min()
        out[str(code)] = (nxt.isin(months).to_numpy() & (rebal >= first))
    return pd.DataFrame(out, index=rebal)


def days_to_next_announcement(fund_long: pd.DataFrame, dates,
                              default_interval: float = 91.0,
                              date_col: str = "DiscDate", code_col: str = "Code"
                              ) -> pd.DataFrame:
    """各営業日 d で「次回決算発表までの予測日数」パネル（PIT・日次イベント戦略用）。

    予測＝直近開示日(≤d) ＋ その銘柄の典型開示間隔（中央値, 四半期≈91日にクランプ）。
    発表直後は ~91、次回が近づくほど 0 へ減少、新規開示でリセット。初開示前は NaN。
    過去の DiscDate のみ使用＝先読み無し（実際の予定日との誤差は run-up 窓幅で吸収）。

    Returns: float DataFrame（index=日付, columns=Code, 値=予測日数。負=予定超過/直前）。
    """
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
    if fund_long.empty or date_col not in fund_long.columns:
        return pd.DataFrame(index=dates)
    df = fund_long.dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    dnp = dates.to_numpy()
    out = {}
    for code, g in df.groupby(code_col):
        dd = g[date_col].drop_duplicates().sort_values()
        diffs = dd.diff().dropna().dt.days
        interval = float(diffs.median()) if len(diffs) else default_interval
        interval = min(max(interval, 60.0), 130.0)
        last = pd.merge_asof(pd.DataFrame({"d": dates}),
                             pd.DataFrame({"dd": dd.to_numpy()}),
                             left_on="d", right_on="dd", direction="backward")["dd"]
        next_exp = last.to_numpy() + np.timedelta64(int(interval), "D")
        out[str(code)] = (next_exp - dnp) / np.timedelta64(1, "D")
    return pd.DataFrame(out, index=dates)

