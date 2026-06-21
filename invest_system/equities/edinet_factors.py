"""EDINET 三表由来の CF系・BS明細系ファクター（GKX Phase 2・#7）。

handoff §3 の特徴量を、生ライン項目（edinet_taxonomy の正準フィールド）から自前計算する。
加工済み比率は取り込まない（ベンダー間で定義が割れるため）。各特徴量は raw で出力し、
クロスセクション標準化（factors.cross_sectional_zscore / cross_sectional_rank）と
セクター中立化（factors.sector_neutralize）は呼び出し側で重ねる（既存 Phase 1 と同じ流儀）。

二層で計算する:
  ① 開示レベル（derive_disclosure_features）… ファンダのみで決まる比率・前年比・3年平均。
     会計年度で整合的に計算し、会計基準移行・非連続年をまたぐ YoY は NaN 化（遡及再表示で
     BS が壊れるため）。これを長形式の列として持ち、提出日アンカーの as-of に流す。
  ② 価格依存（_yield_factors）… FCF 利回り・CF/P。①の as-of パネルを時価総額で割る。

符号は raw（会計上の自然な向き）。プレミアム方向は各 docstring に明記（ML 入力なので
向き付けは推定器/ランク側に委ねる）。FCF は「営業CF＋投資CF」に定義固定（探索しない）。
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .edinet_fundamentals import add_basis_transition, build_edinet_long
from .fundamentals import point_in_time

# 開示レベルで付与する派生ファクター（as-of パネルに載せる列）。
DERIVED_FACTORS = [
    "fcf", "fcf_mean3", "ma_year",          # FCF（利回りは価格と合成）
    "asset_growth",                          # 資産成長（investment 軸。低い=プレミアム）
    "net_share_issuance",                    # 純株式発行（低い=プレミアム）
    "accruals",                              # アクルーアル（低い=高品質）
    "gross_profitability",                   # 粗利益性（高い=高品質）
    "ebitda_margin",                         # EBITDA マージン
    "roic",                                  # ROIC（高い=高品質）
    "leverage",                              # D/E（有利子負債/自己資本）
    "rd_intensity",                          # R&D 集約度
]


def _safe(s: pd.Series) -> pd.Series:
    """0 と負を分母に使えない比率の分母用：>0 のみ残し他は NaN。"""
    return s.where(s > 0)


def attach_split_cf(long: pd.DataFrame, base: Optional[str] = None) -> pd.DataFrame:
    """各 (Code, period_end) に分割累積係数 split_cf＝AdjC/C（as-of ≤ period_end）を付与。

    J-Quants の調整後終値 AdjC は最新日を係数 1 に正規化した back-adjusted 価格で、
    AdjC/C ＝「その日以降の分割の累積係数 Π_{s>d} AdjFactor」（store.rebuild_adjusted と同義）。
    分割調整後の発行株数 = 生株数 / split_cf となり、純粋な分割は前年比 0 に潰れる。
    PIT 不変：純発行に効く比 split_cf(t-1)/split_cf(t) は (t-1,t] の分割のみに依存し
    AdjC の正規化基準（最新日）に依らない＝将来の分割を覗かない。

    価格 wide（Silver 層）が無ければ long をそのまま返す＝raw 株数で計算（後方互換・安全）。
    """
    if (long.empty or "shares_outstanding" not in long.columns
            or "period_end" not in long.columns):
        return long
    try:                                                 # 価格 wide は best-effort
        from ..data import store
        adjc = store.load_wide("AdjC", base=base) if base else store.load_wide("AdjC")
        c = store.load_wide("C", base=base) if base else store.load_wide("C")
    except Exception:  # noqa: BLE001
        return long
    if adjc.empty or c.empty:
        return long
    cf = (adjc / c.where(c != 0)).replace([np.inf, -np.inf], np.nan).sort_index()
    cf.index = pd.to_datetime(cf.index)
    df = long.copy()
    pe = pd.to_datetime(df["period_end"], errors="coerce")
    out = pd.Series(1.0, index=df.index, dtype="float64")
    for code, idx in df.groupby(df["Code"].astype(str)).groups.items():
        if code not in cf.columns:
            continue
        s = cf[code].dropna()
        s = s[~s.index.duplicated(keep="last")]
        sub = pe.loc[idx].dropna()                          # period_end（重複あり得る）
        if s.empty or sub.empty:
            continue
        uniq = pd.DatetimeIndex(sub.unique()).sort_values()  # 一意日付で as-of（≤period_end）
        asof = s.reindex(s.index.union(uniq)).sort_index().ffill().reindex(uniq)
        out.loc[sub.index] = sub.map(asof).values
    df["split_cf"] = out.where(out > 0, 1.0).fillna(1.0)
    return df


def derive_disclosure_features(long: pd.DataFrame) -> pd.DataFrame:
    """長形式（開示×銘柄）に §3 のファンダ派生ファクターを列として付与（純関数）。

    YoY 系（資産成長・純株式発行）は、会計基準移行（basis_changed）または会計年度が
    非連続（前期との期末年差≠1）の開示で NaN にする＝遡及再表示の断絶を持ち込まない。
    """
    if long.empty:
        return long
    df = add_basis_transition(long)                      # Code,period_end 昇順＋basis_changed
    g = df.groupby("Code", sort=False)

    yr = pd.to_datetime(df["period_end"]).dt.year
    contiguous = (yr.values - g["period_end"].shift(1).pipe(
        lambda s: pd.to_datetime(s).dt.year) == 1)
    valid_yoy = pd.Series(contiguous, index=df.index) & ~df["basis_changed"]

    ta = _safe(df["total_assets"])
    sales = _safe(df["net_sales"])

    # FCF（定義固定：営業CF＋投資CF）と 3 年平均（M&A 年の振れを平滑）。
    df["fcf"] = df["cfo"] + df["cfi"]
    df["fcf_mean3"] = g["fcf"].transform(lambda s: s.rolling(3, min_periods=2).mean())
    # M&A 歪み年フラグ：投資 CF 流出が総資産比で大きい（買収年）。
    df["ma_year"] = (df["cfi"] < -0.15 * df["total_assets"]).astype("float64")

    # 資産成長（前年比）。移行/非連続は NaN。低成長ほどプレミアム（investment 因子）。
    df["asset_growth"] = (df["total_assets"] / g["total_assets"].shift(1) - 1.0
                          ).where(valid_yoy)
    # 純株式発行（前年比）。低い（希薄化少）ほどプレミアム。分割調整後株数の前年比で計算＝
    # 純粋な分割は ≈0 に潰す（split_cf があれば調整株数、無ければ raw＝後方互換）。attach_split_cf。
    shares_adj = (df["shares_outstanding"] / _safe(df["split_cf"])
                  if "split_cf" in df.columns else df["shares_outstanding"])
    df["net_share_issuance"] = (
        shares_adj / shares_adj.groupby(df["Code"], sort=False).shift(1) - 1.0
    ).where(valid_yoy)

    # アクルーアル（簡易・CF ベース）：低い（利益の質が高い）ほどプレミアム。
    df["accruals"] = (df["profit"] - df["cfo"]) / ta
    # 粗利益性（Novy-Marx 2013）：高いほどプレミアム。
    df["gross_profitability"] = df["gross_profit"] / ta
    # EBITDA マージン = (営業利益＋減価償却) / 売上。
    df["ebitda_margin"] = (df["operating_income"] + df["depreciation"]) / sales
    # ROIC = NOPAT / 投下資本。実効税率は [0,1] にクリップ。投下資本＝有利子負債＋純資産。
    tax_rate = (df["income_taxes"] / df["pretax_income"]).clip(0, 1)
    nopat = df["operating_income"] * (1.0 - tax_rate)
    df["roic"] = nopat / _safe(df["interest_debt"] + df["net_assets"])
    # レバレッジ D/E（有利子負債 / 自己資本=純資産）。
    df["leverage"] = df["interest_debt"] / _safe(df["net_assets"])
    # R&D 集約度。
    df["rd_intensity"] = df["rd_expense"] / sales
    return df


def _yield_factors(efp: dict[str, pd.DataFrame], mcap: pd.DataFrame) -> dict:
    """価格依存ファクター：FCF 利回り・CF/P（as-of の生額 ÷ 時価総額）。高い=割安。"""
    m = mcap.where(mcap > 0)
    out = {}
    if "fcf" in efp:
        out["fcf_yield"] = efp["fcf"] / m
    if "fcf_mean3" in efp:
        out["fcf_yield_3y"] = efp["fcf_mean3"] / m          # M&A 平滑版（推奨）
    if "cfo" in efp:
        out["cf_to_price"] = efp["cfo"] / m
    return {k: v.reindex(index=mcap.index, columns=mcap.columns) for k, v in out.items()}


def edinet_factor_panels(rebal_dates, codes: Optional[Iterable] = None,
                         mcap: Optional[pd.DataFrame] = None, lag_days: int = 1
                         ) -> dict[str, pd.DataFrame]:
    """§3 ファクターの as-of wide パネル群（raw）。提出日アンカー・FYE 非依存。

    返り値 {factor: DataFrame(index=rebal, columns=Code)}。mcap（時価総額 wide）を渡すと
    FCF 利回り・CF/P も加わる（時価総額は J-Quants 株価×株数で呼び出し側が用意）。
    標準化・セクター中立は factors の各関数を重ねて得る（raw を返す）。
    """
    long = derive_disclosure_features(attach_split_cf(build_edinet_long()))
    rebal = pd.DatetimeIndex(sorted(pd.to_datetime(list(rebal_dates)))).normalize()
    if long.empty:
        return {f: pd.DataFrame(index=rebal, dtype="float64") for f in DERIVED_FACTORS}
    if codes is not None:
        want = {str(c) for c in codes}
        long = long[long["Code"].isin(want)]

    fields = DERIVED_FACTORS + (["cfo"] if mcap is not None else [])
    efp = point_in_time(long, rebal, fields, date_col="DiscDate",
                        code_col="Code", lag_days=lag_days)
    out = {f: efp[f] for f in DERIVED_FACTORS if f in efp}
    if mcap is not None:
        out.update(_yield_factors(efp, mcap.reindex(index=rebal)))
    return out
