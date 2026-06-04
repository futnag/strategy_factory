"""バックテスト層：purged CPCV で一次モデルを評価し Sharpe を分布として返す。

設計書 §5.2 / DP6 / DP10。単一パスの点推定でなく φ 本のパスの分布で評価する。
"""

from .cpcv_backtest import CPCVBacktestResult, cpcv_backtest

__all__ = ["CPCVBacktestResult", "cpcv_backtest"]
