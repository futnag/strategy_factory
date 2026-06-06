"""投資部門別フローのロードとシグナル化（需給の分析層）。

/equities/investor-types（週次・市場区分別・13主体×売買）から、海外勢の純買い
（FrgnBal=買−売）などの需給シグナルを作る。市場区分は時期で変遷するが TokyoNagoya
（東京・名古屋の合算）は全期間連続するため、長期の市場タイミング信号に向く。

派生は純関数（渡したDataFrameに作用）でネットワーク不要・テスト可能。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..data.sources import jquants as jq

# 各主体の差引（Bal=Buy−Sell）列。net flow に使う。
_BAL = {"foreign": "FrgnBal", "individual": "IndBal", "inv_trust": "InvTrBal",
        "broker": "BrkBal", "prop": "PropBal", "bank": "BankBal",
        "trust_bank": "TrstBnkBal", "business": "BusCoBal"}


def load_investor_types(base: Optional[str] = None) -> pd.DataFrame:
    """投資部門別キャッシュをロード（canonical all.parquet 優先、無ければ全連結）。"""
    root = (Path(base) if base else jq._CACHE) / "investor_types"
    if not root.exists():
        return pd.DataFrame()
    canon = root / "all.parquet"
    if canon.exists():
        df = pd.read_parquet(canon)
        if not df.empty and "_empty" not in df.columns:
            return df
    frames = []
    for p in sorted(root.glob("*.parquet")):
        d = pd.read_parquet(p)
        if not d.empty and "_empty" not in d.columns:
            frames.append(d)
    return (pd.concat(frames, ignore_index=True).drop_duplicates()
            if frames else pd.DataFrame())


def section_net_flow(df: pd.DataFrame, investor: str = "foreign",
                     section: str = "TokyoNagoya", date_col: str = "EnDate"
                     ) -> pd.Series:
    """指定主体・区分の週次ネットフロー（差引）系列（index=週末日）。"""
    col = _BAL.get(investor, investor)
    if df.empty or col not in df.columns or "Section" not in df.columns:
        return pd.Series(dtype="float64")
    sub = df[df["Section"] == section].dropna(subset=[date_col])
    s = sub.set_index(date_col)[col].sort_index()
    return s[~s.index.duplicated(keep="last")]


def net_flow_intensity(df: pd.DataFrame, investor: str = "foreign",
                       section: str = "TokyoNagoya") -> pd.Series:
    """ネットフローを売買総額で正規化（=需給の強度, -1..1目安）。区分依存の規模を除く。"""
    bal = _BAL.get(investor, investor)
    tot = bal.replace("Bal", "Tot")
    if df.empty or bal not in df.columns or tot not in df.columns:
        return pd.Series(dtype="float64")
    sub = df[df["Section"] == section].dropna(subset=["EnDate"]).copy()
    sub = sub.set_index("EnDate").sort_index()
    sub = sub[~sub.index.duplicated(keep="last")]
    denom = sub[tot].replace(0, pd.NA)
    return (sub[bal] / denom).astype("float64")
