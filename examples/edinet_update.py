"""EDINET 書類一覧の by-date 差分更新（欠損日のみ取得・冪等再開）。

専用 DataUpdater（base=data/edinet・別キー EDINET_API_KEY）で EDINET_DATASETS を回す。
J-Quants の夜間更新（base=data/jquants）とは分離し互いを乱さない。空（書類なしの日）も
マーカー保存されるため再実行は冪等・中断しても再開できる。Phase 2 夜間ジョブ（ops repo）
にはこのスクリプト相当を追加する。**末尾で所有者別パネル（build_ownership_panel.py）も
オフライン再生成**する（冪等・ネット不要・`J_OWNERSHIP_BUILD=0` で無効化。docs/50 §8.1）。

縦覧期間の都合上、古い TOB/大量保有は API 上 null 化されるため初回ミラーの遡及は ~5 年が
実効上限（有報は ~10 年）。初回は --plan で件数を見てから流すとよい。

usage:
  python examples/edinet_update.py --plan                       # 取得予定（欠損日数）だけ表示
  python examples/edinet_update.py --start 2021-01-01           # start〜前営業日まで差分取得
  python examples/edinet_update.py --start 2026-05-01 --until 2026-06-12
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env                     # noqa: E402
from invest_system.data.catalog import EDINET_DATASETS       # noqa: E402
from invest_system.data.sources import edinet as ed          # noqa: E402
from invest_system.data.updater import DataUpdater           # noqa: E402


def build_updater(start: str) -> DataUpdater:
    """EDINET 専用の差分更新器（base=data/edinet）。"""
    return DataUpdater(datasets=EDINET_DATASETS, refresh_datasets={},
                       base="data/edinet", manifest_path="data/edinet/manifest.json",
                       start=start)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01",
                    help="ミラー開始日（既定 2021-01-01＝EDINET 一次ソース窓）")
    ap.add_argument("--until", default=None, help="終端日（既定＝前営業日）")
    ap.add_argument("--plan", action="store_true", help="取得予定（欠損日数）のみ表示")
    args = ap.parse_args()

    ed._api_key()                                # キー欠如は早期に明示エラー
    up = build_updater(args.start)
    until = args.until or (pd.Timestamp.today().normalize()
                           - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if args.plan:
        miss = up.plan("edinet_docs", until)
        head, tail = (miss[0], miss[-1]) if miss else ("-", "-")
        print(f"edinet_docs 欠損日: {len(miss)} 日（{head} 〜 {tail}）")
        return 0

    print(f"=== EDINET 差分更新: {args.start} 〜 {until} ===")
    rep = up.update(until=until, verbose=True)
    print(rep)

    # 夜間: EDINET docs 更新後に所有者別パネルをオフライン再生成（冪等・既定ON・失敗は無害化）。
    if get_env("J_OWNERSHIP_BUILD", "1") == "1":
        print("\n=== 所有者別パネル オフライン再生成（build_ownership_panel.py）===")
        try:
            subprocess.run([sys.executable,
                            str(Path(__file__).with_name("build_ownership_panel.py"))],
                           check=False, timeout=2400)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] ownership panel build skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
