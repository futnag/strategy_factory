"""daily_regime: 日足価格・出来高レジーム検知（接続設計 v2.1）。

step 0–1（シーム契約＋ leak_tests ＋データ再利用配線）のみ実装済み。
検知器本体・pooling 推定器・multiscale 結合・戦略・コストは未実装（後続 step）。
"""
from .config import DailyRegimeConfig
from .contracts import (
    ContractViolation,
    LookAheadError,
    validate_anchor,
    validate_tiers,
)

__all__ = [
    "DailyRegimeConfig",
    "ContractViolation",
    "LookAheadError",
    "validate_anchor",
    "validate_tiers",
]
