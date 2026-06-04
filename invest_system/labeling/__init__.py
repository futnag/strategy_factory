"""ラベリング層（L5）：トリプルバリア法とメタラベリング用ラベル生成。

設計書 §7 / DP4 に対応。固定ホライズン法を排し、利確/損切/時間切れの
パス依存ラベルを生成する。``side`` を与えればメタラベリング（{0,1}）に対応。
"""

from .triple_barrier import (
    apply_pt_sl_on_t1,
    get_bins,
    get_events,
    get_vertical_barriers,
    get_vol,
)

__all__ = [
    "apply_pt_sl_on_t1",
    "get_bins",
    "get_events",
    "get_vertical_barriers",
    "get_vol",
]
