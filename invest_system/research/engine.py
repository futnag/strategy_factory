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
    capacity_jpy: float = float("nan")   # 容量(¥): participation%×ADV 制約の AUM 上限


def _ann_factor(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 3:
        return 252.0
    years = (idx[-1] - idx[0]).days / 365.25
    return float(len(idx) / years) if years > 0 else 252.0


def backtest(strategy: Strategy, view: AsOfView, *, price_field: str = "close",
             costs_bps: float = 15.0, rebalance=None, execution_lag: int = 0,
             adv: pd.DataFrame | None = None, participation: float = 0.1
             ) -> BacktestResult:
    """戦略を回してネット損益系列を返す。

    execution_lag: 決定から執行までの遅延（バー数）。0=決定足の終値で執行（既定・
      従来）、1=翌足で執行（観測した終値で建てない＝同足の先読みを排除する現実寄り）。
    adv: 各銘柄の平均売買代金(¥)パネル（index=リバランス日, col=銘柄）。与えると
      容量(capacity_jpy)＝「最も流動性の低い建玉が participation×ADV に達するAUM上限」を
      算出（実運用で約定可能な規模の上限）。
    """
    close = view.panels[price_field]
    ret = close.pct_change()
    fwd = ret.shift(-(1 + execution_lag))       # 決定t→(t+lag)建て→翌足で実現
    drop = 1 + execution_lag
    dates = pd.DatetimeIndex(rebalance) if rebalance is not None \
        else view.dates[:-drop]                 # 実現できない末尾は除外
    prev_w: pd.Series | None = None
    rows = []
    capacity = float("inf")
    for t in dates:
        w = strategy.target_weights(view.asof(t))
        if len(w):
            r = float((w * fwd.loc[t].reindex(w.index)).sum())
            npos = int((w != 0).sum())
            if adv is not None and t in adv.index:
                a = adv.loc[t].reindex(w.index)
                wabs = w.abs()
                ok = (wabs > 0) & a.notna() & (a > 0)
                if bool(ok.any()):
                    capacity = min(capacity,
                                   float((participation * a[ok] / wabs[ok]).min()))
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
                          _ann_factor(df.index), strategy.name, strategy.params,
                          capacity_jpy=(capacity if capacity < float("inf")
                                        else float("nan")))
