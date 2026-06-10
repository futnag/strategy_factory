"""外部価格（TSMOM 11資産＋ヘッジ先物）の無人差分更新 CLI（Phase 2・D5）。

data/investers/ の正本ファイルへ、Yahoo Finance から**新規日付のみ**追記する
（既存履歴は不変・重複日クロスチェックで別系列の混入を遮断）。詳細は
invest_system/data/external_fetch.py。

使い方:
  .venv\\Scripts\\python.exe examples\\update_external.py            # Phase2 の 11 キー
  .venv\\Scripts\\python.exe examples\\update_external.py gold wti   # キー指定
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.data.external_fetch import update_external_prices  # noqa: E402


def main() -> int:
    keys = sys.argv[1:] or None
    rep = update_external_prices(keys)
    print("=== 外部価格 差分更新（Yahoo・検証付き追記）===")
    for _, r in rep.iterrows():
        bp = "" if r["overlap_bp"] != r["overlap_bp"] else f" 照合{r['overlap_bp']:.0f}bp"
        print(f"  {r['key']:12s} {str(r['symbol']):10s} {r['status']:10s} "
              f"+{int(r['n_new'])}日 最終={r['last']}{bp}")
    n_ok = int((rep["status"] == "OK").sum())
    print(f"完了: {n_ok}/{len(rep)} キー更新")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
