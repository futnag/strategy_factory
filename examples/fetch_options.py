"""日経225オプション四本値の一括取得（差分更新器で options_225 を最新化）。

各営業日の全契約（IV・理論価格・原資産NK225・建玉等, 1日≈9,600行）を取得。差分のみ
取得・冪等・再開可能。Standard 前提（J_QUANTS_MIN_INTERVAL=0.7 推奨）。約2,400営業日で
初回は約35分・数百MB級。以降は `update` で日次差分のみ。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.updater import DataUpdater  # noqa: E402


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    up = DataUpdater()
    plan = up.plan("options_225", __import__("pandas").Timestamp.today().normalize())
    print(f"日経225オプション一括取得：欠損 {len(plan)} 営業日（差分のみ）…")
    rep = up.update(names=["options_225"])
    print("完了:", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
