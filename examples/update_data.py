"""差分更新CLI（要件4）：ローカルのデータを最新化（新規日だけ取得）。

使い方:
  .venv\\Scripts\\python.exe examples\\update_data.py            # 今日まで最新化
  .venv\\Scripts\\python.exe examples\\update_data.py 2026-06-05  # 指定日まで

既存キャッシュ＋マニフェストから「取得済み日」を自動把握し、欠損日だけ取得する。
祝日・未公表は空マーカーが残り再取得しない＝冪等・再開可能。Standard 推奨間隔
J_QUANTS_MIN_INTERVAL=0.7。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.updater import DataUpdater  # noqa: E402


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    until = sys.argv[1] if len(sys.argv) > 1 else None
    up = DataUpdater()
    target = pd.Timestamp(until) if until else pd.Timestamp.today().normalize()
    print(f"=== 差分更新（〜{target:%Y-%m-%d}）===")
    by_date = [n for n, d in up.datasets.items() if d.maintained]
    refresh = [n for n, d in up.refresh_datasets.items() if d.maintained]
    print(f"対象: by-date={by_date} / refresh={refresh}")
    print("\n[計画] by-date 欠損日数:")
    for name in by_date:
        print(f"  {name}: {len(up.plan(name, target))} 日")
    if refresh:
        print(f"[計画] refresh（全体を最新化）: {refresh}")
    print("\n[実行]:")
    rep = up.update(until=until)
    fetched = sum(r.get("fetched", 0) for r in rep.values())
    refreshed = sum(r.get("refreshed_rows", 0) for r in rep.values())
    print(f"\n完了: by-date 新規 {fetched} 件 / refresh {refreshed:,} 行。ローカルは最新です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
