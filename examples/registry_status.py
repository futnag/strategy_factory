"""永続グローバル・レジストリの俯瞰：scope別の累計試行 K と Sharpe分散を表示。

判定器が「これまで何試行をこの scope で行ったか（＝デフレートの基準）」を一覧する。
試行を重ねるほど K が増え、全戦略の DSR 基準が上がる＝選択バイアスの可視化。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.validation.registry import default_registry  # noqa: E402


def main() -> int:
    with default_registry() as reg:
        scopes = reg.list_scopes()
        if not scopes:
            print("（永続レジストリに試行は未記録）")
            return 0
        print(f"{'scope':<26}{'K(累計試行)':>12}{'V[SR]':>10}")
        print("-" * 48)
        for scope, k, srv in scopes:
            print(f"{scope:<26}{k:>12}{srv:>10.4f}")
        print("-" * 48)
        print(f"scope数={len(scopes)}  総試行={sum(k for _, k, _ in scopes)}")
        print("\n※ 同一 scope で試行(K)が増えるほど DSR 基準は上がる（p-hack不能）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
