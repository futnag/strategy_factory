"""daily_regime: ファンダ/決算データの PIT（(a)ループ・最大の地雷）。

財務は**開示日 DiscDate** でアライン（**期末日 CurFYEn でない**）。期末→開示まで数ヶ月のラグが
あり、期末日で使うと未公表財務でトレード＝深刻なリーク。既存 `equities.fundamentals.point_in_time`
（DiscDate≤t−lag の最新開示のみ・lag_days≥1）を**再利用**し、リークガードと意図的リーク版（期末日
アライン）を足す。恒久 leak_test で「期末日アラインは赤」を毎ループ確認する。
"""
from __future__ import annotations

import pandas as pd

from invest_system.equities.fundamentals import point_in_time

from .contracts import LookAheadError


def asof_field(fund_long: pd.DataFrame, rebal_dates, field: str, *, lag_days: int = 1) -> pd.DataFrame:
    """DiscDate アンカーの as-of パネル（PIT・date×code）。既存 point_in_time を再利用。"""
    out = point_in_time(fund_long, rebal_dates, [field], lag_days=lag_days)
    return out.get(field, pd.DataFrame())


def leaky_period_end_asof(fund_long: pd.DataFrame, rebal_dates, field: str,
                          *, period_col: str = "CurFYEn") -> pd.DataFrame:
    """⚠リーク版：DiscDate でなく**期末日 CurFYEn** でアライン（未公表財務でトレード＝先読み）。"""
    fl = fund_long.rename(columns={period_col: "_pe"})
    out = point_in_time(fl, rebal_dates, [field], date_col="_pe", lag_days=0)
    return out.get(field, pd.DataFrame())


def eps_asof_panel(fund_long: pd.DataFrame, daily_index, *, lag_days: int = 1) -> pd.DataFrame:
    """日次 as-of EPS パネル（DiscDate アンカー・階段関数）。value 一次の E/P 分子。"""
    return asof_field(fund_long, daily_index, "EPS", lag_days=lag_days)


def pead_window_panel(fund_long: pd.DataFrame, daily_index, *, D: int = 20,
                      lag_days: int = 1) -> pd.DataFrame:
    """正 FY サプライズ（sue_recent>0）後の D 営業日窓 bool（PIT）。PEAD 一次のロング窓。

    FY 実績 EPS − 直近予想（`disclosure_features` の sue_recent_raw）＞0 を正サプライズとし、開示翌
    営業日から D 営業日 True。interim の累計EPS vs 通期予想（apples-to-oranges）は使わない。
    """
    from invest_system.equities.fundamental_factors import disclosure_features
    out_fy, _ = disclosure_features(fund_long)
    pos = out_fy[pd.to_numeric(out_fy["sue_recent_raw"], errors="coerce") > 0]
    didx = pd.DatetimeIndex(daily_index)
    codes = sorted(pos["Code"].astype(str).unique())
    flag = pd.DataFrame(False, index=didx, columns=codes)
    for code, g in pos.groupby(pos["Code"].astype(str)):
        loc = flag.columns.get_loc(code)
        for d in pd.to_datetime(g["DiscDate"]):
            start = int(didx.searchsorted(d, side="right"))      # 開示翌営業日（lag_days=1 と整合）
            end = min(start + D, len(didx))
            if start < len(didx):
                flag.iloc[start:end, loc] = True
    return flag


def assert_fundamental_pit(panel: pd.DataFrame, fund_long: pd.DataFrame, field: str,
                           *, date_col: str = "DiscDate", code_col: str = "Code",
                           lag_days: int = 1) -> None:
    """as-of パネルの各 (date, code) 値が **開示日+lag より前に出現していない**ことをアサート。

    期末日アライン（未公表をトレード）は値が開示前に現れるため、ここで赤になる。
    """
    fl = fund_long.dropna(subset=[date_col]).copy()
    fl[date_col] = pd.to_datetime(fl[date_col])
    fl[code_col] = fl[code_col].astype(str)
    for code in panel.columns:
        g = fl[fl[code_col] == str(code)]
        vals = pd.to_numeric(g[field], errors="coerce")
        s = panel[code].dropna()
        for d, v in s.items():
            disc = g.loc[vals == v, date_col]
            if len(disc):
                earliest = pd.Timestamp(disc.min()) + pd.Timedelta(days=lag_days)
                if pd.Timestamp(d) < earliest:
                    raise LookAheadError(
                        f"{code}@{pd.Timestamp(d).date()}: 値 {v} が開示+lag "
                        f"{earliest.date()} より前に出現（期末日漏れの疑い）")
