"""取引所固有のデータ取込（L1）。"""

from .bitbank import fetch_trades, parse_trades

__all__ = ["fetch_trades", "parse_trades"]
