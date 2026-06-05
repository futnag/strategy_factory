"""ユニバース構築：普通株の抽出＋流動性フィルタ。

ETF/REIT/インフラファンド等を除外し、十分な流動性（売買代金）を持つ
普通株に絞る。流動性の薄い銘柄はクロスセクション研究のノイズ源かつ
実運用で約定困難なため除外する（実務的な前処理）。
"""
from __future__ import annotations

import pandas as pd

# Mkt コード（V2 /equities/master）:
#   プライム 0111 / スタンダード 0112 / グロース 0113
#   ETF・REIT 等は 0109（その他）
_COMMON_MARKETS = {"0111", "0112", "0113"}


def filter_common_stocks(listed: pd.DataFrame) -> pd.DataFrame:
    """普通株のみ抽出（市場区分で判定、無ければ素通し）。"""
    df = listed.copy()
    if "Mkt" in df.columns:
        df = df[df["Mkt"].astype(str).isin(_COMMON_MARKETS)]
    elif "MarketCode" in df.columns:  # V1 後方互換
        df = df[df["MarketCode"].astype(str).isin(_COMMON_MARKETS)]
    return df.reset_index(drop=True)


def select_universe(listed: pd.DataFrame, turnover_panel: pd.DataFrame,
                    top_n: int = 500, min_obs: int = 12) -> list[str]:
    """普通株のうち、観測十分かつ売買代金中央値の大きい上位 top_n 銘柄。

    listed         : /equities/master（Code, Mkt, S33 ...）
    turnover_panel : index=リバランス日, columns=Code, 値=売買代金(Va) の wide
    min_obs        : 必要な非NULL観測月数（流動性の継続性を担保）
    """
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    cols = [c for c in turnover_panel.columns if str(c) in common]
    sub = turnover_panel[cols]
    enough = sub.notna().sum(axis=0) >= min_obs
    sub = sub.loc[:, enough[enough].index]
    med = sub.median(axis=0, skipna=True).sort_values(ascending=False)
    return [str(c) for c in med.head(top_n).index]
