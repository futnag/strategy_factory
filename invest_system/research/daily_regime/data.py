"""daily_regime: 既存基盤への再利用配線（薄い adapter・新規計算を持たない）。

store.load_wide / equities.price_factors / data の S33 マスタを束ね、日足の
close/adj_close/turnover(=Va)/volume、Amihud（横断）、ADV、値幅制限・stale フラグ、
チャネル区分を供給する。robust 正規化（median/MAD）は step 2 の features.py（本書では未実装）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from invest_system.data.store import load_wide
from invest_system.equities import price_factors
from invest_system.research.sector_regime.data_loader import get_data_root

# チャネル区分（§4・4.4 と連動）。決算窓で扱いを変える（masking は step 4 の event_mask.py）。
RETURN_VOL_CHANNEL = ("realized_vol", "log_return")    # 決算窓でダミー化する側
LIQUIDITY_CHANNEL = ("amihud", "dollar_volume")         # 決算窓も更新を継続する側


def load_s33_map() -> pd.Series:
    """Code → S33（equities_master.parquet）。"""
    fp = get_data_root() / "jquants" / "equities_master.parquet"
    listed = pd.read_parquet(fp)
    if "S33" not in listed.columns or "Code" not in listed.columns:
        raise ValueError("equities_master に Code/S33 列がありません")
    return (
        listed.assign(Code=listed["Code"].astype(str))
        .dropna(subset=["S33"])
        .drop_duplicates("Code")
        .set_index("Code")["S33"]
        .astype(str)
    )


@dataclass
class DailyPanels:
    """日足 wide パネル束（index=日付, columns=Code）。"""
    close: pd.DataFrame
    adj_close: pd.DataFrame
    turnover: pd.DataFrame       # Va = 売買代金（円）
    volume: pd.DataFrame
    upper_limit: pd.DataFrame
    lower_limit: pd.DataFrame


def load_daily_panels(
    start: Optional[str] = None,
    end: Optional[str] = None,
    codes: Optional[list[str]] = None,
) -> DailyPanels:
    """Silver wide から必要フィールドを読み、（任意で）銘柄部分集合に絞る。"""
    base = str(get_data_root())

    def g(field: str) -> pd.DataFrame:
        df = load_wide(field, start=start, end=end, base=base)
        if codes is not None and not df.empty:
            keep = [c for c in df.columns if str(c) in set(map(str, codes))]
            df = df[keep]
        return df

    return DailyPanels(
        close=g("close"), adj_close=g("adj_close"), turnover=g("turnover"),
        volume=g("volume"), upper_limit=g("upper_limit"), lower_limit=g("lower_limit"),
    )


def amihud_panel(panels: DailyPanels, *, window: int = 60) -> pd.DataFrame:
    """横断 Amihud 非流動性（≤t trailing・price_factors を再利用）。"""
    return price_factors.amihud_illiquidity(panels.adj_close, panels.turnover, window=window)


def adv_panel(panels: DailyPanels, *, window: int = 60) -> pd.DataFrame:
    """trailing 平均売買代金 ADV（ゲート用・≤t のみ）。"""
    mp = max(2, int(window * 0.8))
    return panels.turnover.rolling(window, min_periods=mp).mean()


def stale_flag(panels: DailyPanels, *, window: int = 60, thresh: float = 0.6) -> pd.DataFrame:
    """stale price フラグ（zero-return 比率の代理・§4.3）。True=出来高僅少で価格停滞。"""
    z = price_factors.zero_return_days(panels.adj_close, window=window)
    return (z >= thresh).fillna(False)


def limit_flag(panels: DailyPanels) -> pd.DataFrame:
    """値幅制限到達フラグ（§4.3）。upper/lower limit のいずれか。"""
    ul = panels.upper_limit.fillna(0).astype(bool)
    ll = panels.lower_limit.fillna(0).astype(bool)
    return ul | ll
