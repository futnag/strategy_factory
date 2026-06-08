"""データセット・カタログ：各データの更新頻度・キャッシュ場所・取得方法を一元管理。

差分更新エンジン（updater）が参照する。maintained=True は「完全時系列として最新に
保つ」対象（信用・空売り）。daily_quotes は研究時オンデマンド取得のため maintained
=False（全営業日の全銘柄ミラーはしない＝コア網羅＋オンデマンド方針）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from .sources import jquants as jq


@dataclass(frozen=True)
class Dataset:
    name: str
    cadence: str                                  # "daily" | "weekly"
    cache_subdir: str                             # data/jquants 配下
    fetch_by_date: Callable[[str], pd.DataFrame]  # (YYYYMMDD) -> df（キャッシュ付）
    stem_to_date: Callable[[str], Optional[str]]  # キャッシュ名 stem -> YYYYMMDD or None
    maintained: bool = True                       # 差分更新の既定対象か


def _suffix_date(stem: str) -> Optional[str]:
    """'date_20260225' / 'calc_20260225' -> '20260225'。"""
    last = stem.split("_")[-1]
    return last if (last.isdigit() and len(last) == 8) else None


def _plain_date(stem: str) -> Optional[str]:
    """'20260225' -> '20260225'。"""
    return stem if (stem.isdigit() and len(stem) == 8) else None


DATASETS: dict[str, Dataset] = {
    "weekly_margin": Dataset(
        "weekly_margin", "weekly", "margin_weekly",
        lambda d: jq.fetch_weekly_margin(date=d), _suffix_date),
    "margin_alert": Dataset(
        "margin_alert", "daily", "margin_alert",
        lambda d: jq.fetch_margin_alert(date=d), _suffix_date),
    "short_positions": Dataset(
        "short_positions", "daily", "short_positions",
        lambda d: jq.fetch_short_positions(calc_date=d), _suffix_date),
    # 日経225オプション四本値（IV含む・各営業日の全契約）
    "options_225": Dataset(
        "options_225", "daily", "options_225",
        lambda d: jq.fetch_index_options(d), _plain_date),
    # 全銘柄日次株価はオンデマンド（完全ミラーしない）
    "daily_quotes": Dataset(
        "daily_quotes", "daily", "daily",
        lambda d: jq.fetch_daily_quotes(d), _plain_date, maintained=False),
}


# --- range-refresh 型（by-date に乗らない：from/to or 銘柄別で全体を再取得） -----
@dataclass(frozen=True)
class RefreshSpec:
    """期間を指定して「全体を最新化」する小型データセット（指数・投資部門別）。"""
    name: str
    refresh: Callable[[str, str], int]   # (start, until) -> 取得行数
    maintained: bool = True


def _refresh_investor_types(start: str, until: str) -> int:
    """投資部門別を全履歴で再取得し固定ファイルへ（週次・軽量＝1呼び出し）。"""
    df = jq.fetch_investor_types(frm=start, to=until, refresh=True, canonical=True)
    return len(df)


def _index_codes() -> list[str]:
    """キャッシュ済み指数コード（無ければ直近日付の一覧から）。"""
    d = jq._CACHE / "indices"
    codes = [p.stem.split("_", 1)[1] for p in d.glob("code_*.parquet")] \
        if d.exists() else []
    if not codes:
        snap = jq.fetch_index_bars(date="20260529")
        if "Code" in snap.columns:
            codes = [str(x) for x in snap["Code"].dropna().unique()]
    return sorted(set(codes))


def _refresh_indices(start: str, until: str) -> int:
    """各指数を全履歴で再取得（銘柄別・各1回）。"""
    rows = 0
    for c in _index_codes():
        rows += len(jq.fetch_index_bars(code=c, frm=start, to=until, refresh=True))
    return rows


REFRESH_DATASETS: dict[str, RefreshSpec] = {
    "investor_types": RefreshSpec("investor_types", _refresh_investor_types),
    "indices": RefreshSpec("indices", _refresh_indices),
}
