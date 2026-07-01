"""回転手法（押し目買い指値）バックテスト。

生存者バイアス注意: 上場廃止銘柄を含まないデータでは成績が過大評価される。
"""
from .config import CONFIG, TUNABLE_PARAMS, filtered_equity_config, futures_config, futures_entry_methods
from .filters import build_blacklist_from_trades, load_regime
from .regime_adapt import REGIME_ADAPT_MODES, resolve_regime_cfg
from .data import load_from_external, load_from_wide, load_symbols
from .engine import Position, gap_adjust, resolve_exit, run_backtest
from .indicators import add_indicators
from .metrics import performance_report, plot_equity
from .run import main
from .synthetic import synthetic_data

__all__ = [
    "CONFIG",
    "TUNABLE_PARAMS",
    "REGIME_ADAPT_MODES",
    "build_blacklist_from_trades",
    "filtered_equity_config",
    "resolve_regime_cfg",
    "futures_config",
    "futures_entry_methods",
    "load_regime",
    "Position",
    "add_indicators",
    "gap_adjust",
    "load_from_external",
    "load_from_wide",
    "load_symbols",
    "main",
    "performance_report",
    "plot_equity",
    "resolve_exit",
    "run_backtest",
    "synthetic_data",
]