r"""TDnet 適時開示一覧を前向きに蓄積。

毎営業日1回実行する想定。直近の開示日（既定=今日）を ``data/tdnet/{YYYYMMDD}.parquet``
に保存（冪等・再開可能）。**公開閲覧は直近約1ヶ月のみ**＝バックフィル不可。

    # Windows (PowerShell)
    $env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\update_tdnet.py
    $env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\update_tdnet.py --days 5

設計: docs/47 §2.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invest_system.data.sources.tdnet import (  # noqa: E402
    filter_tagged, load_tdnet, update_tdnet)

CACHE = "data/tdnet"


def recent_biz_dates(n: int) -> list[str]:
    """直近 n 営業日の YYYYMMDD リスト（新しい順）。"""
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return [d.strftime("%Y%m%d") for d in idx[::-1]]


def main() -> int:
    ap = argparse.ArgumentParser(description="TDnet 適時開示 前向き蓄積")
    ap.add_argument("--days", type=int, default=1,
                    help="遡る営業日数（既定=今日のみ）")
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    dates = recent_biz_dates(max(1, args.days))
    print(f"=== TDnet 適時開示 前向き蓄積（{len(dates)} 営業日）===")
    rep = update_tdnet(args.cache, dates=dates)
    print("report:", rep)

    df = load_tdnet(args.cache)
    print(f"\n蓄積累計: {len(df)} 件 / {df['disclosure_date'].nunique() if len(df) else 0} 開示日")
    if len(df):
        for tag, label in [("buyback_announce", "自社株買い発表"),
                           ("tob_related", "TOB関連"),
                           ("guidance_revision", "業績修正")]:
            sub = filter_tagged(df, tag)
            if len(sub):
                print(f"  {label}: {len(sub)} 件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())