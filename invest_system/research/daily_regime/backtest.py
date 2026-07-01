"""daily_regime: §6.3 ベースライン比較（完全統制）— step 4。

B&H／無フィルタ／週足のみ／日足のみ／二層(スタブ) を**同一コスト・同一執行ラグ・同一評価窓**で
回す（既存 research.engine.backtest を再利用）。レジーム依存コスト（§6.4）を最初から適用。
**日足のみ＝二層の週足tilt=0 版**（multiscale λ=0）と一致させる（接続設計・step 0–1 整合）。

⚠ 週足アンカーは step 5 まで momentum スタブ → 本層の数字は**配管検証**であって中核証拠ではない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from invest_system.research.data_view import AsOfView
from invest_system.research.engine import BacktestResult, backtest

from .cost import regime_cost_panel
from .strategy import ConstantLong, MomentumPrimary, RegimeAwareStrategy


def summarize(res: BacktestResult) -> dict:
    """ネットリターン系列の要約（年率・Sharpe・最大DD・平均ターンオーバー）。"""
    r = res.returns.dropna()
    if len(r) < 3:
        return {"n": int(len(r)), "ann_ret": np.nan, "sharpe": np.nan,
                "maxdd": np.nan, "turnover": np.nan}
    af = res.ann_factor
    sharpe = float(r.mean() / r.std() * np.sqrt(af)) if r.std() > 0 else np.nan
    cum = (1.0 + r).cumprod()
    dd = float((cum / cum.cummax() - 1.0).min())
    return {"n": int(len(r)), "ann_ret": float(r.mean() * af), "sharpe": sharpe,
            "maxdd": dd, "turnover": float(res.turnover.mean())}


def _col(s: pd.Series, code: str) -> pd.DataFrame:
    return pd.DataFrame({code: s})


def run_value_baselines(
    code: str,
    close_panel: pd.DataFrame,
    daily_stressed_lam0: pd.Series,
    daily_stressed_lamX: pd.Series,
    weekly_pbear: pd.Series,
    amihud: pd.Series,
    *,
    lookback: int = 60,
    filter_thresh: float = 0.6,
    size_floor: float = 0.2,
    base_bps: float = 10.0,
    impact_coef: float = 20.0,
    execution_lag: int = 1,
    warmup: int = 252,
) -> dict:
    """§6.3 の5ベースラインを同一条件で回し {name: (BacktestResult, summary)} を返す。"""
    frames = {"d0": _col(daily_stressed_lam0, code), "dX": _col(daily_stressed_lamX, code),
              "wk": _col(weekly_pbear, code)}
    view = AsOfView({"close": close_panel[[code]], **frames})
    costs = regime_cost_panel(_col(amihud, code), base_bps=base_bps, impact_coef=impact_coef)
    dates = view.dates[warmup:-(1 + execution_lag)]            # 連続部分列・末尾は執行で除外

    mom = lambda: MomentumPrimary(code, lookback)              # noqa: E731
    rg = dict(filter_thresh=filter_thresh, size_floor=size_floor)
    strategies = {
        "B&H": ConstantLong(code),
        "no-filter": mom(),
        "weekly-only": RegimeAwareStrategy(mom(), regime_field="wk", **rg),
        "daily-only": RegimeAwareStrategy(mom(), regime_field="d0", **rg),   # = 方式A λ=0
        "two-layer": RegimeAwareStrategy(mom(), regime_field="dX", **rg),    # = 方式A λ>0（スタブ）
    }
    out = {}
    for name, strat in strategies.items():
        res = backtest(strat, view, costs_bps=costs, execution_lag=execution_lag, rebalance=dates)
        out[name] = (res, summarize(res))
    return out
