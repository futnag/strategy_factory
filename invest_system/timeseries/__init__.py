"""時系列ミーンリバージョン・統計的裁定ツールキット（柱D・KB §11）。

López de Prado の検証規律（本体）に対し、Ernie Chan の平均回帰・共和分・統計的裁定を
戦略仮説面で補完する。全関数は「渡された ≤t 系列」で因果計算＝先読みなし（DP12）。
"""
from __future__ import annotations

from .cointegration import (
    JohansenResult,
    cadf,
    hedge_ratio_ols,
    hedge_ratio_tls,
    johansen,
    spread_series,
)
from .kalman import KalmanHedge
from .mean_reversion import (
    adf_pvalue,
    half_life,
    hurst_exponent,
    ou_params,
    variance_ratio,
)
from .regime import (
    efficiency_ratio,
    equal_weight_market,
    expanding_tertile,
    trend_regime,
    vol_regime,
)

__all__ = [
    "adf_pvalue",
    "half_life",
    "hurst_exponent",
    "ou_params",
    "variance_ratio",
    "JohansenResult",
    "cadf",
    "hedge_ratio_ols",
    "hedge_ratio_tls",
    "johansen",
    "spread_series",
    "KalmanHedge",
    "efficiency_ratio",
    "equal_weight_market",
    "expanding_tertile",
    "trend_regime",
    "vol_regime",
]
