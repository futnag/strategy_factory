"""Phase 2 実運用配管（L10-L11 の最小実装）：シグナル→注文リスト→台帳・照合。

検証ファクトリ（research/）の戦略コードを**そのまま**使い、目標ウェイトを実際に
発注可能な単位（単元未満株・先物枚数・ロット）へ写像する層。戦略ロジックは持たない
（凍結済みパラメータの適用のみ＝DP10）。
"""

from .ledger import (
    ALERT_DD, DERISK_DD, STOP_DD, apply_actual_fills, drawdown_status,
    next_open_fills, yen_positions_pnl,
)
from .orders import equity_orders, hedge_contracts, lot_orders

__all__ = [
    "ALERT_DD",
    "DERISK_DD",
    "STOP_DD",
    "apply_actual_fills",
    "drawdown_status",
    "equity_orders",
    "hedge_contracts",
    "lot_orders",
    "next_open_fills",
    "yen_positions_pnl",
]
