"""バックテスト・エンジン：戦略の目標ウェイト → コスト込みの損益系列。

各リバランス日 t で戦略に AsOf（t以前）を渡してウェイト w_t を得る。実現損益は
t→t+1 の各銘柄リターンと w_t の内積（＝将来価格で実現＝戦略は未来を見ないが
エンジンは実現値を計算）。取引コストは回転率 sum|Δw| に比例。空シグナル日は現金
（リターン0）。年率係数は日付間隔から自動推定（日次≈252, 月次≈12）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_view import AsOfView
from .strategy import Strategy


@dataclass
class BacktestResult:
    returns: pd.Series          # ネット周期リターン（index=決定日 t）
    gross: pd.Series            # コスト前
    turnover: pd.Series         # sum|Δw|（両側）
    n_positions: pd.Series      # 建玉数
    ann_factor: float           # 年率換算の周期数/年
    name: str
    params: dict = field(default_factory=dict)


def _ann_factor(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 3:
        return 252.0
    years = (idx[-1] - idx[0]).days / 365.25
    return float(len(idx) / years) if years > 0 else 252.0


def backtest(strategy: Strategy, view: AsOfView, *, price_field: str = "close",
             costs_bps: float = 15.0, rebalance=None) -> BacktestResult:
    """戦略を回してネット損益系列を返す。"""
    close = view.panels[price_field]
    fwd = close.pct_change().shift(-1)          # t→t+1 リターン（銘柄別）
    dates = pd.DatetimeIndex(rebalance) if rebalance is not None \
        else view.dates[:-1]                    # 最終日は将来未実現なので除外
    prev_w: pd.Series | None = None
    rows = []
    for t in dates:
        w = strategy.target_weights(view.asof(t))
        if len(w):
            r = float((w * fwd.loc[t].reindex(w.index)).sum())
            npos = int((w != 0).sum())
        else:
            w, r, npos = pd.Series(dtype="float64"), 0.0, 0
        if prev_w is None:
            turn = float(w.abs().sum())
        else:
            names = w.index.union(prev_w.index)
            cur = w.reindex(names).fillna(0.0)
            pre = prev_w.reindex(names).fillna(0.0)
            turn = float((cur - pre).abs().sum())
        rows.append((t, r - costs_bps / 1e4 * turn, r, turn, npos))
        prev_w = w
    df = pd.DataFrame(rows, columns=["date", "net", "gross", "turnover",
                                     "npos"]).set_index("date")
    return BacktestResult(df["net"], df["gross"], df["turnover"], df["npos"],
                          _ann_factor(df.index), strategy.name, strategy.params)
