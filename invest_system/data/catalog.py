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
    # 全銘柄日次株価はオンデマンド（完全ミラーしない）
    "daily_quotes": Dataset(
        "daily_quotes", "daily", "daily",
        lambda d: jq.fetch_daily_quotes(d), _plain_date, maintained=False),
}
