"""戦略インターフェース（契約）と実装例。

Strategy 契約：各リバランス日 t に AsOf（t以前のデータ）を受け取り、目標ウェイト
（index=銘柄, 値=比率, 正=ロング/負=ショート）を返す。エンジンが返りを受けて
コスト込みで損益を計算する。状態を持たず history から再計算する設計＝PIT安全。

DSL ではなくコードの契約。単純ルールは GapReversal のような薄いクラスで“糖衣”化し、
複雑な戦略（ニュース・テキストマイニング等）は同契約を実装すれば差し込める。
"""
from __future__ import annotations

import pandas as pd

from .data_view import AsOf


class Strategy:
    """戦略の基底。name/params は判定器の試行レジストリ記録に使う。"""

    name: str = "strategy"
    params: dict = {}

    def target_weights(self, asof: AsOf) -> pd.Series:
        raise NotImplementedError


class GapReversal(Strategy):
    """ギャップ・リバーサル（要件2の例）。

    「当日始値が前日終値より threshold 以上ギャップダウンした銘柄」を、その後
    hold 日にわたり等加重で side 方向に保有する。close→close で評価（執行は終値、
    寄り執行の精緻化は後続）。threshold/hold/side はパラメータ＝判定器では“格子”
    として試行計数しデフレートする対象。
    """

    def __init__(self, threshold: float = 0.10, hold: int = 1, side: int = 1):
        self.threshold = float(threshold)
        self.hold = int(hold)
        self.side = int(side)
        self.name = f"gap_rev(th={threshold:g},hold={hold},side={side:+d})"
        self.params = {"threshold": threshold, "hold": hold, "side": side}

    def target_weights(self, asof: AsOf) -> pd.Series:
        op, cl = asof.frame("open"), asof.frame("close")
        n = len(cl)
        if n < 2:
            return pd.Series(dtype="float64")
        held: set = set()
        for k in range(self.hold):
            i = n - 1 - k                 # 直近=n-1。i 日のギャップ＝open[i]/close[i-1]-1
            if i < 1:
                break
            gap = op.iloc[i] / cl.iloc[i - 1] - 1.0
            held.update(gap[gap <= -self.threshold].dropna().index)
        latest = cl.iloc[-1]
        held = [c for c in held if pd.notna(latest.get(c))]   # 現在も上場
        if not held:
            return pd.Series(dtype="float64")
        w = float(self.side) / len(held)
        return pd.Series({c: w for c in held}, dtype="float64")


class CrossSectionalStrategy(Strategy):
    """事前計算済みの PIT ファクターをロングショートに変換する戦略。

    既存のクロスセクション・ファクター（value 等）を判定器に載せるための薄い
    アダプタ。factor は wide（index=リバランス日, col=銘柄）で、各行が t 時点で
    既知（PIT）であること。上位/下位 quantile を等加重ロング/ショート。
    """

    def __init__(self, factor: pd.DataFrame, quantile: float = 0.2,
                 name: str = "cross_sectional"):
        self.factor = factor
        self.quantile = float(quantile)
        self.name = name
        self.params = {"quantile": quantile}

    def target_weights(self, asof: AsOf) -> pd.Series:
        if asof.asof not in self.factor.index:
            return pd.Series(dtype="float64")
        row = self.factor.loc[asof.asof].dropna()
        if len(row) < 5:
            return pd.Series(dtype="float64")
        k = max(1, int(len(row) * self.quantile))
        order = row.sort_values()
        w = pd.Series(0.0, index=row.index, dtype="float64")
        w[order.index[-k:]] = 1.0 / k       # 上位ロング
        w[order.index[:k]] = -1.0 / k       # 下位ショート
        return w[w != 0.0]
