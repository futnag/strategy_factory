"""daily_regime: 決算窓フラグ（§2 PEAD 適用除外の窓）— 実 DiscDate から PIT。

PEAD は**発表後**ドリフト → 窓は (DiscDate, DiscDate+window]（発表当日は跨がない＝翌営業日から）。
この窓は (i) PEAD 一次戦略のロング窓、(ii) §2 フィルタ適用除外窓、の双方に使う（同一定義）。
DiscDate は実開示日（EDINET/TDnet）＝ PIT。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def earnings_window_flag(disc_dates: pd.DataFrame, daily_index,
                         *, codes: Optional[list[str]] = None,
                         window: int = 5, date_col: str = "DiscDate",
                         code_col: str = "Code") -> pd.DataFrame:
    """[Code, DiscDate] → Date×Code の決算窓 bool（発表翌営業日〜+window 営業日が True）。"""
    didx = pd.DatetimeIndex(daily_index)
    dd = disc_dates[[code_col, date_col]].copy()
    dd[code_col] = dd[code_col].astype(str)
    dd[date_col] = pd.to_datetime(dd[date_col])
    cols = [str(c) for c in (codes if codes is not None else sorted(dd[code_col].unique()))]
    flag = pd.DataFrame(False, index=didx, columns=cols)
    colset = set(cols)
    for code, g in dd[dd[code_col].isin(colset)].groupby(code_col):
        loc = flag.columns.get_loc(code)
        for d in g[date_col]:
            start = int(didx.searchsorted(d, side="right"))      # DiscDate より後の最初の営業日
            end = min(start + window, len(didx))
            if start < len(didx):
                flag.iloc[start:end, loc] = True
    return flag


def load_disc_dates(codes: Optional[list[str]] = None) -> pd.DataFrame:
    """EDINET 開示インデックスから [Code, DiscDate] を読む（実開示日・PIT）。"""
    from invest_system.research.sector_regime.data_loader import get_data_root
    fp = get_data_root() / "processed" / "edinet_annual_index.parquet"
    df = pd.read_parquet(fp)
    ccol = next((c for c in ("Code", "code", "secCode", "SecCode") if c in df.columns), None)
    dcol = next((c for c in ("DiscDate", "disclosure_date", "submitDateTime", "Date") if c in df.columns), None)
    if ccol is None or dcol is None:
        raise ValueError(f"edinet index に Code/DiscDate 相当列が無い: {list(df.columns)[:20]}")
    out = pd.DataFrame({"Code": df[ccol].astype(str).str.zfill(4), "DiscDate": pd.to_datetime(df[dcol])})
    out = out.dropna().drop_duplicates()
    if codes is not None:
        out = out[out["Code"].isin([str(c) for c in codes])]
    return out.sort_values(["Code", "DiscDate"]).reset_index(drop=True)
