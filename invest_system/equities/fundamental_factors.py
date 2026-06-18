"""fins_summary 由来の新ファンダ特徴量（GKX 特徴量拡充 B・EDINET 非依存）。

決算サマリー（J-Quants・提出日アンカー PIT）＋日次 adj_close から、既存 value/quality/size に
無い特徴量を追加する。**既存の roe/roa/op_margin/accruals/earnings_yield 等は再実装しない**。

開示レベル（disclosure_features）で会計年度整合に算出 → 材化側で `fundamentals.point_in_time`
（DiscDate アンカー）で月末 as-of。価格依存（52週高値・Dimson β）と季節性は日次/月次で算出。

SUE 連携（重要・実データで確認）：FY 開示の実績 EPS と、当該 FY（CurFYEn）の四半期開示が持つ
全年度予想 FEPS を突合する。日本企業は期中に予想を実績へ寄せるため「直近予想」差は小さくなり
がち → **直近予想差（sue_recent）と期初予想差（sue_initial）の両方**を出す（GKX 的に ML 入力は
増やしてよい）。スケールは株価（surprise yield・材化側で月末終値除し）＝赤字でも頑健（handoff 確定）。

符号（handoff §3-3）：SUE・予想改訂・52週高値・安定度（負ボラ）は house 符号（大きいほどロング）。
成長・持続可能成長は raw（方向中立）。equity_growth は資産成長の代理だが EDINET investment 軸とは
別物（docs/17 明記）。Dimson β は低ベータ＝ロングで負号。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .price_factors import _rolling_beta_residvar

_INTERIM = {"1Q", "2Q", "3Q"}


def disclosure_features(fund_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fins_summary long → (FY 開示単位の特徴量, 予想改訂 long)。純関数。

    返り値 out_fy: [Code, DiscDate(=発表日), sue_recent_raw, sue_initial_raw, sales_growth,
    profit_growth, equity_growth, roe_stability, margin_stability, sustainable_growth]。
    out_rev: [Code, DiscDate, forecast_revision_raw]（各開示での全年度予想 FEPS の前回差）。
    SUE は raw サプライズ（EPS 円）で返し、株価除しは材化側で行う。
    """
    df = fund_long.copy()
    df["DiscDate"] = pd.to_datetime(df["DiscDate"])
    df["Code"] = df["Code"].astype(str)
    for c in ("EPS", "FEPS", "Sales", "NP", "Eq", "OP", "PayoutRatioAnn"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["fy"] = pd.to_datetime(df.get("CurFYEn"), errors="coerce").dt.year
    df["per"] = df.get("CurPerType").astype(str)
    df = df.sort_values(["Code", "DiscDate"])

    # 予想改訂：各 (Code, fy) で FEPS の前回開示比（raw 差）。発表日アンカー。
    rev = df.dropna(subset=["fy", "FEPS"]).copy()
    rev["forecast_revision_raw"] = rev.groupby(["Code", "fy"])["FEPS"].diff()
    out_rev = rev.dropna(subset=["forecast_revision_raw"])[
        ["Code", "DiscDate", "forecast_revision_raw"]]

    # FY 開示（実績）と四半期開示（予想）の突合
    fy_rows = df[df["per"] == "FY"].dropna(subset=["fy"])
    actual = (fy_rows.groupby(["Code", "fy"])
              .agg(actual_eps=("EPS", "last"), DiscDate=("DiscDate", "last"),
                   Sales_fy=("Sales", "last"), NP_fy=("NP", "last"),
                   Eq_fy=("Eq", "last"), OP_fy=("OP", "last"),
                   payout=("PayoutRatioAnn", "last")).reset_index())
    interim = df[df["per"].isin(_INTERIM)].dropna(subset=["fy", "FEPS"])
    recent = interim.groupby(["Code", "fy"])["FEPS"].last().rename("recent_fc")
    initial = interim.groupby(["Code", "fy"])["FEPS"].first().rename("initial_fc")
    t = (actual.merge(recent, on=["Code", "fy"], how="left")
         .merge(initial, on=["Code", "fy"], how="left").sort_values(["Code", "fy"]))

    t["sue_recent_raw"] = t["actual_eps"] - t["recent_fc"]
    t["sue_initial_raw"] = t["actual_eps"] - t["initial_fc"]
    t["sales_growth"] = t.groupby("Code")["Sales_fy"].pct_change()
    t["profit_growth"] = t.groupby("Code")["NP_fy"].pct_change()
    t["equity_growth"] = t.groupby("Code")["Eq_fy"].pct_change()
    roe = t["NP_fy"] / t["Eq_fy"].where(t["Eq_fy"] > 0)
    margin = t["OP_fy"] / t["Sales_fy"].where(t["Sales_fy"] > 0)
    t["roe"], t["op_margin"] = roe, margin
    # 安定度＝過去 ROE/マージンの trailing 標準偏差に負号（高安定＝ロング）。最低3点。
    t["roe_stability"] = -t.groupby("Code")["roe"].transform(
        lambda s: s.rolling(4, min_periods=3).std())
    t["margin_stability"] = -t.groupby("Code")["op_margin"].transform(
        lambda s: s.rolling(4, min_periods=3).std())
    # 持続可能成長率＝ROE×(1−配当性向)。PayoutRatioAnn は比率（実データで確認・÷100 不要）。
    payout = t["payout"].clip(lower=0.0, upper=1.0)
    t["sustainable_growth"] = roe * (1.0 - payout)

    out_fy = t[["Code", "DiscDate", "sue_recent_raw", "sue_initial_raw", "sales_growth",
                "profit_growth", "equity_growth", "roe_stability", "margin_stability",
                "sustainable_growth"]]
    return out_fy, out_rev


def high_52w(adj_close: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """52週高値乖離（George-Hwang 2004）：close / trailing 252 営業日 max。高い（≈1）=ロング側。"""
    return adj_close / adj_close.rolling(window, min_periods=int(window * 0.8)).max()


def seasonality(adj_close: pd.DataFrame, min_years: int = 5,
                max_years: int = 10) -> pd.DataFrame:
    """季節性（Heston-Sadka 2008）：過去複数年の**同暦月**月次リターン平均。高い=ロング側。

    月次リターンの 12,24,…,max_years×12 ヶ月ラグの平均（min_years 以上の観測を要求）。
    履歴 ~10 年では推定がノイジー＝**弱い実験的特徴**（docs/17 明記）。PIT：過去のみ参照。
    """
    mret = adj_close.resample("ME").last().pct_change()
    parts = [mret.shift(12 * k).to_numpy() for k in range(1, max_years + 1)]
    arr = np.stack(parts)                                  # (max_years, T, N)
    cnt = np.sum(~np.isnan(arr), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nansum(np.nan_to_num(arr), axis=0) / np.where(cnt > 0, cnt, np.nan)
    mean = np.where(cnt >= min_years, mean, np.nan)
    return pd.DataFrame(mean, index=mret.index, columns=mret.columns)


def dimson_beta(adj_close: pd.DataFrame, market: Optional[pd.Series] = None,
                window: int = 252, lags: int = 1) -> pd.DataFrame:
    """Dimson β（薄商い対応）：当期＋ラグ市場への（単回帰）β係数和に負号（低ベータ＝ロング）。

    厳密な Dimson は当期＋ラグの**重回帰**係数和だが、ベクトル化のため当期βとラグβ（各単回帰・
    `_rolling_beta_residvar` 流用）の和で近似する（thin-trading 補正の集約係数版・docs/17 明記）。
    既存 `price_factors.beta`（当期のみ・負号）と整合。
    """
    ret = adj_close.pct_change()
    mkt = market if market is not None else ret.mean(axis=1)
    beta, _ = _rolling_beta_residvar(ret, mkt, window)
    total = beta
    for lag in range(1, lags + 1):
        b_lag, _ = _rolling_beta_residvar(ret, mkt.shift(lag), window)
        total = total.add(b_lag, fill_value=0.0)
    return -total
