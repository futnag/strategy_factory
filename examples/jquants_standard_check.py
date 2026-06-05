"""Standardプラン検証：高速レート・10年遡及・配信遅延の解消を実測。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import jquants as jq  # noqa: E402

# 無料枠では到達不可だった範囲を狙う
RECENT = ["20260415", "20260515", "20260529"]   # 旧12週遅延の外（解消確認）
OLD = ["20240115", "20220104", "20200106", "20180115",
       "20160630", "20150105"]                   # 10年遡及＆境界探索


def main() -> int:
    print(f"MIN_INTERVAL={jq._MIN_INTERVAL}s  MAX_RETRIES={jq._MAX_RETRIES}")
    print("--- レート＆配信遅延（直近・旧枠外）---")
    t0 = time.monotonic()
    for d in RECENT:
        try:
            q = jq.fetch_daily_quotes(d, refresh=True)
            print(f"  {d}: rows={len(q)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {d}: {str(e)[:90]}")
    print("--- 10年遡及（古い日付・境界探索）---")
    for d in OLD:
        try:
            q = jq.fetch_daily_quotes(d, refresh=True)
            print(f"  {d}: rows={len(q)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {d}: {str(e)[:90]}")
    print(f"経過 {time.monotonic() - t0:.1f}s（{len(RECENT) + len(OLD)}回）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
