"""特徴量層（L3-）：分数階差分・マイクロ構造・因果フィルタ等。

設計書 §7 / DP2 に対応。まず分数階差分（記憶を保ちつつ定常化）を提供。
"""

from .frac_diff import (
    find_min_d,
    frac_diff_d_table,
    frac_diff_ffd,
    get_weights_ffd,
)

__all__ = [
    "find_min_d",
    "frac_diff_d_table",
    "frac_diff_ffd",
    "get_weights_ffd",
]
