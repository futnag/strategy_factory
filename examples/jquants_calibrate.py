"""持続可能な呼び出し間隔のキャリブレーション。

環境変数 J_QUANTS_MIN_INTERVAL を変えつつ、リトライ無し(J_QUANTS_MAX_RETRIES=0)で
連続取得し、429を踏まずに済む間隔を実測する。窓内(2024-03-13〜2026-03-13)の
相異なる日付を使用。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import jquants as jq  # noqa: E402

DATES = ["20240415", "20240515", "20240617", "20240716", "20240815",
         "20240917", "20241015", "20241115", "20241216", "20250115",
         "20250217", "20250317", "20250415", "20250515"]


def main() -> int:
    print(f"MIN_INTERVAL={jq._MIN_INTERVAL}s, MAX_RETRIES={jq._MAX_RETRIES}")
    ok = err = 0
    t0 = time.monotonic()
    for d in DATES:
        try:
            q = jq.fetch_daily_quotes(d, refresh=True)
            ok += 1
            print(f"  {d}: OK rows={len(q)}")
        except Exception as e:  # noqa: BLE001
            err += 1
            print(f"  {d}: {str(e)[:70]}")
    dt = time.monotonic() - t0
    print(f"成功 {ok}/{len(DATES)}, 429等 {err},  経過 {dt:.1f}s, "
          f"平均 {dt / len(DATES):.1f}s/回")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
