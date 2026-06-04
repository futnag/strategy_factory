"""サンプリング層（L6）：非IID対処（サンプル独自性・逐次ブートストラップ）。

設計書 §4.3 / DP5 に対応。金融ラベルは時間的に重なり独立でない。重複を
独自性で測り、重みづけ・逐次ブートストラップでアウトオブサンプル精度を高める。
"""

from .uniqueness import (
    average_uniqueness,
    average_uniqueness_from_indicator,
    get_indicator_matrix,
    num_concurrent_events,
    sample_weights_by_return,
    sequential_bootstrap,
    time_decay,
)

__all__ = [
    "average_uniqueness",
    "average_uniqueness_from_indicator",
    "get_indicator_matrix",
    "num_concurrent_events",
    "sample_weights_by_return",
    "sequential_bootstrap",
    "time_decay",
]
