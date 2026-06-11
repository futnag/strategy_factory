"""検証ハーネス（L8）：過学習・リーク・多重検定を構造的に封じる。

設計書 §5 / DP5-DP7 / DP10 に対応。戦略コードより先に整備する基盤。
"""

from .dsr import (
    sharpe_ratio,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    min_backtest_length,
    min_track_record_length,
)
from .purge_embargo import get_train_times, embargo_after
from .cpcv import CombinatorialPurgedKFold
from .pbo import PBOResult, pbo_cscv
from .registry import TrialRegistry

__all__ = [
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "min_backtest_length",
    "min_track_record_length",
    "get_train_times",
    "embargo_after",
    "CombinatorialPurgedKFold",
    "PBOResult",
    "pbo_cscv",
    "TrialRegistry",
]
