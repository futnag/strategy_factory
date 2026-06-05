"""クロスセクション・ファクターの算出・標準化・交絡中立化。

バリュー/クオリティ/サイズはすべて「生株価 C ＋ 開示時点の財務」で内部整合的に
計算する（分割調整後 AdjC はリターン/モメンタム専用）。フロー変数は四半期累計の
歪みを避けるため予想通年値（FEPS/FSales/FNP/FOP）を優先する。

交絡（セクター・サイズ）の中立化は pillar C の核心：ファクターの見かけの効果から
業種・規模という共通要因を除き、固有の効果を取り出す。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def market_cap(raw_price: pd.DataFrame, shares_out: pd.DataFrame,
               treasury: pd.DataFrame | None = None) -> pd.DataFrame:
    """時価総額 = 生株価 × (発行済株式数 − 自己株式数)。ラベルで自動整合。"""
    sh = shares_out
    if treasury is not None:
        sh = shares_out.sub(treasury, fill_value=0.0)
    return raw_price * sh


def value_quality_size_factors(pit: dict[str, pd.DataFrame], raw_price: pd.DataFrame,
                               adj_price: pd.DataFrame | None = None,
                               mom_lookback: int = 12, mom_skip: int = 1
                               ) -> dict[str, pd.DataFrame]:
    """PIT財務（フィールド別wide）＋価格から各ファクターのwideパネルを生成。

    値の向きは「大きいほど割安/高品質/小型」になるよう符号付け（=ロング側が
    期待プレミアム方向）。欠損は NaN のまま（後段のz化で自然に除外）。
    """
    def f(name: str) -> pd.DataFrame:
        return pit.get(name, pd.DataFrame(index=raw_price.index))

    shares = f("ShOutFY").sub(f("TrShFY"), fill_value=0.0)
    mcap = raw_price * shares

    out: dict[str, pd.DataFrame] = {}
    # --- バリュー（高いほど割安）---
    out["earnings_yield"] = f("FEPS") / raw_price          # 予想E/P
    out["book_to_market"] = f("Eq") / mcap                  # B/M
    out["cf_yield"] = f("CFO") / mcap                       # 営業CF利回り
    out["sales_yield"] = f("FSales") / mcap                 # 予想売上利回り
    out["div_yield"] = f("FDivAnn") / raw_price             # 予想配当利回り
    # --- クオリティ（高いほど良質）---
    eq = f("Eq").replace(0.0, np.nan)
    ta = f("TA").replace(0.0, np.nan)
    fsales = f("FSales").replace(0.0, np.nan)
    out["roe"] = f("FNP") / eq                              # 予想ROE
    out["roa"] = f("FNP") / ta                              # 予想ROA
    out["op_margin"] = f("FOP") / fsales                    # 予想営業利益率
    out["equity_ratio"] = f("EqAR")                         # 自己資本比率
    # --- サイズ（小型ほど大きい値＝小型プレミアム方向）---
    out["size"] = -np.log(mcap.where(mcap > 0))
    # --- モメンタム（12-1, 調整後価格）---
    if adj_price is not None:
        out["momentum"] = adj_price.shift(mom_skip) / adj_price.shift(mom_lookback) - 1.0

    # ユニバース列に整列
    return {k: v.reindex(index=raw_price.index, columns=raw_price.columns)
            for k, v in out.items()}


def cross_sectional_zscore(df: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    """各日付（行）で銘柄横断にzスコア化。winsor で±σにクリップ（外れ値抑制）。"""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0)
    z = df.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0)
    if winsor is not None and winsor > 0:
        z = z.clip(lower=-winsor, upper=winsor)
    return z


def sector_neutralize(df: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """各日付で同一セクター内の平均を引き、業種共通成分を除去（交絡制御）。

    sector: index=Code, 値=セクターコード(S33等)。df の列に整合させて使用。
    """
    sec = sector.reindex([str(c) for c in df.columns])
    sec.index = df.columns
    groups = sec.fillna("NA")
    # 列方向に groupby して各行から群平均を引く
    demeaned = df.copy()
    for _, cols in groups.groupby(groups):
        block = df[cols.index]
        demeaned[cols.index] = block.sub(block.mean(axis=1), axis=0)
    return demeaned
