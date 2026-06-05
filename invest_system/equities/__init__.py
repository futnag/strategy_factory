"""日本株クロスセクション・ファクター研究（pillar C の本拠地）。

J-Quants V2（無料枠：株価四本値＋財務サマリー、2年/12週遅延）を用い、
ポイントインタイム整合・交絡（セクター/サイズ）制御・DSRデフレートを備えた
クロスセクション・ファンダ・ファクター検証を行う。
"""

from .universe import filter_common_stocks, select_universe
from .panel import (
    assemble_panel,
    fetch_month_end_snapshots,
    forward_returns,
    trailing_momentum,
)
from .fundamentals import point_in_time
from .factors import (
    cross_sectional_zscore,
    market_cap,
    sector_neutralize,
    value_quality_size_factors,
)
from .backtest import long_short_returns

__all__ = [
    "filter_common_stocks",
    "select_universe",
    "assemble_panel",
    "fetch_month_end_snapshots",
    "forward_returns",
    "trailing_momentum",
    "point_in_time",
    "cross_sectional_zscore",
    "market_cap",
    "sector_neutralize",
    "value_quality_size_factors",
    "long_short_returns",
]
