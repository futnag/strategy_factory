r"""JSF（日本証券金融）貸借取引データを手動DL生ファイルから前向きに蓄積。

JSF サイトから**手動DL**した日次ファイル（CSV/TSV）を `--raw` ディレクトリに置いて実行すると、
`data/jsf/{YYYYMMDD}.parquet` に正準スキーマで保存する（冪等）。**ネットワーク不要・スクレイピング
なし**（リポジトリ規律：自動巡回は禁止、手動DLのみ）。公式 API なし＝バックフィル限定・前向き蓄積。

    # Windows (PowerShell)
    $env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\update_jsf.py --raw data\jsf_raw
    # 上書き再取込:
    $env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\update_jsf.py --raw data\jsf_raw --overwrite

設計: docs/49 §2-3。代替の公式ソース＝J-Quants Premium 貸借（契約時）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invest_system.data.sources.jsf import ingest_dir, load_jsf, squeeze_flags  # noqa: E402

CACHE = "data/jsf"


def main() -> int:
    ap = argparse.ArgumentParser(description="JSF 貸借取引 前向き蓄積（手動DL→parquet）")
    ap.add_argument("--raw", required=True,
                    help="手動DLした JSF 生ファイル（CSV/TSV）のディレクトリ")
    ap.add_argument("--glob", default="*.csv", help="生ファイルの glob（既定 *.csv）")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--overwrite", action="store_true", help="既存日付を上書き")
    args = ap.parse_args()

    print(f"=== JSF 貸借取引 前向き蓄積（raw={args.raw}）===")
    rep = ingest_dir(args.raw, args.cache, glob=args.glob, overwrite=args.overwrite)
    print("report:", rep)

    df = load_jsf(args.cache)
    n_days = df["Date"].dt.normalize().nunique() if len(df) else 0
    print(f"\n蓄積累計: {len(df)} 行 / {n_days} 営業日")
    if len(df):
        sf = squeeze_flags(df)
        n_sq = int(sf.to_numpy().sum()) if not sf.empty else 0
        print(f"  逆日歩発生（借株逼迫）: 延べ {n_sq} 銘柄日")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
