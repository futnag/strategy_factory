"""検証ファクトリ（Phase 1 MVP）：任意の戦略を厳格に判定する。

設計思想：
- 偽陽性の最大源は「先読み（リーク）」と「選択バイアス（提出までの試行数）」。
- 前者は AsOfView（戦略に未来を見せない構造）で、後者はグローバル試行レジストリ＋
  事前登録＋パラメータ格子のデフレートで封じる。
- 戦略は DSL でなくコードの契約（Strategy）で表現し、単純ルールは糖衣で書く。
"""

from .data_view import AsOf, AsOfView
from .engine import BacktestResult, backtest
from .judge import GridVerdict, StrategyVerdict, judge_grid
from .report_html import to_html, write_html
from .strategy import (
    CalendarStrategy, CompositeStrategy, CrossSectionalStrategy, EarningsRunup,
    GapReversal, PairsStrategy, SignalTimingStrategy, Strategy,
)

__all__ = [
    "AsOf",
    "AsOfView",
    "Strategy",
    "GapReversal",
    "SignalTimingStrategy",
    "CalendarStrategy",
    "PairsStrategy",
    "EarningsRunup",
    "CrossSectionalStrategy",
    "CompositeStrategy",
    "backtest",
    "BacktestResult",
    "judge_grid",
    "GridVerdict",
    "StrategyVerdict",
    "to_html",
    "write_html",
]
