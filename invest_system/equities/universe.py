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


def point_in_time_universe(turnover_panel: pd.DataFrame, top_n: int = 300,
                           lookback: int = 12, min_obs: int = 6) -> pd.DataFrame:
    """時変ユニバースの所属マスク（先読み・生存者バイアスを排除）。

    各日付 t で「過去 lookback 行（t を含む）」の売買代金中央値が上位 top_n、かつ
    t 時点で実際に取引している（turnover 非NaN）銘柄のみ True にする。未来の流動性は
    一切使わない。turnover_panel は普通株に絞った wide（index=リバランス日, col=Code）。

    Returns: bool DataFrame（index=日付, columns=Code, True=その時点のユニバース所属）。
    """
    mask = pd.DataFrame(False, index=turnover_panel.index,
                        columns=turnover_panel.columns)
    for i, t in enumerate(turnover_panel.index):
        window = turnover_panel.iloc[max(0, i - lookback + 1): i + 1]
        obs = window.notna().sum(axis=0)
        med = window.median(axis=0, skipna=True)
        trading = turnover_panel.loc[t].notna()
        eligible = med[(obs >= min_obs) & trading]
        top = eligible.sort_values(ascending=False).head(top_n).index
        mask.loc[t, top] = True
    return mask


def universe_members(mask: pd.DataFrame) -> list[str]:
    """マスクで一度でも所属した銘柄の和集合（財務等を取得すべき superset）。"""
    ever = mask.any(axis=0)
    return [str(c) for c in ever[ever].index]


def apply_universe_mask(factor: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """各時点で非所属の銘柄を NaN にして、ランキング対象から除外する。"""
    m = mask.reindex(index=factor.index, columns=factor.columns).fillna(False)
    return factor.where(m)
