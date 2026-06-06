"""戦略インターフェース（契約）と実装例。

Strategy 契約：各リバランス日 t に AsOf（t以前のデータ）を受け取り、目標ウェイト
（index=銘柄, 値=比率, 正=ロング/負=ショート）を返す。エンジンが返りを受けて
コスト込みで損益を計算する。状態を持たず history から再計算する設計＝PIT安全。

DSL ではなくコードの契約。単純ルールは GapReversal のような薄いクラスで“糖衣”化し、
複雑な戦略（ニュース・テキストマイニング等）は同契約を実装すれば差し込める。
"""
from __future__ import annotations

import numpy as np
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


class SignalTimingStrategy(Strategy):
    """単一銘柄（指数等）を、外部シグナルの符号でタイミング建玉する時系列戦略。

    signal は publication-lag 反映済み（index=利用可能日, 値=シグナル）。各リバランス日
    t で「t 以前の最新シグナル」が threshold を超えれば side 方向に建玉、否なら現金。
    投資部門別フロー→TOPIX のような需給タイミング戦略を判定器に載せるためのIF。
    """

    def __init__(self, signal: pd.Series, code: str, threshold: float = 0.0,
                 side: int = 1, name: str | None = None):
        self.signal = signal.sort_index()
        self.code = code
        self.threshold = float(threshold)
        self.side = int(side)
        self.name = name or f"signal_timing({code},th={threshold:g},side={side:+d})"
        self.params = {"code": code, "threshold": threshold, "side": side}

    def target_weights(self, asof: AsOf) -> pd.Series:
        s = self.signal.loc[:asof.asof]                # t 以前の最新（先読み無し）
        if s.empty or pd.isna(s.iloc[-1]):
            return pd.Series(dtype="float64")
        if s.iloc[-1] > self.threshold:
            return pd.Series({self.code: float(self.side)}, dtype="float64")
        return pd.Series(dtype="float64")


class CalendarStrategy(Strategy):
    """暦日条件で単一銘柄（指数等）を建玉する季節性戦略（月末効果 turn-of-month 等）。

    各日 t の暦日が「月末側 dom_start 以上 または 月初側 dom_end 以下」なら side 方向に
    建玉、否なら現金。日付は既知＝先読み無し。dom_start/dom_end/side が試行格子。
    """

    def __init__(self, code: str, dom_start: int = 25, dom_end: int = 3,
                 side: int = 1, name: str | None = None):
        self.code = code
        self.dom_start = int(dom_start)
        self.dom_end = int(dom_end)
        self.side = int(side)
        self.name = name or (f"calendar({code},>= {dom_start}|<= {dom_end},"
                             f"side={side:+d})")
        self.params = {"code": code, "dom_start": dom_start, "dom_end": dom_end,
                       "side": side}

    def target_weights(self, asof: AsOf) -> pd.Series:
        d = asof.asof.day
        if d >= self.dom_start or d <= self.dom_end:
            return pd.Series({self.code: float(self.side)}, dtype="float64")
        return pd.Series(dtype="float64")


class PairsStrategy(Strategy):
    """2銘柄のスプレッド（対数比）z-scoreで平均回帰する相対価値戦略（ダラーニュートラル）。

    各 t で直近 lookback の対数比 log(A)-log(B) の z を見る。z>+entry は A 割高→A売り/B買い、
    z<-entry は A 割安→A買い/B売り、中間は現金。AsOf の過去価格のみ使用＝先読み無し。
    """

    def __init__(self, a: str, b: str, lookback: int = 60, entry: float = 1.5,
                 name: str | None = None):
        self.a, self.b = a, b
        self.lookback = int(lookback)
        self.entry = float(entry)
        self.name = name or f"pairs({a}-{b},lb={lookback},entry={entry:g})"
        self.params = {"a": a, "b": b, "lookback": lookback, "entry": entry}

    def target_weights(self, asof: AsOf) -> pd.Series:
        cl = asof.frame("close")
        if self.a not in cl.columns or self.b not in cl.columns:
            return pd.Series(dtype="float64")
        sub = cl[[self.a, self.b]].dropna()
        if len(sub) < self.lookback + 2:
            return pd.Series(dtype="float64")
        spread = np.log(sub[self.a]) - np.log(sub[self.b])
        win = spread.iloc[-self.lookback:]
        sd = win.std(ddof=0)
        if sd == 0:
            return pd.Series(dtype="float64")
        z = (spread.iloc[-1] - win.mean()) / sd
        if z > self.entry:
            return pd.Series({self.a: -0.5, self.b: 0.5}, dtype="float64")
        if z < -self.entry:
            return pd.Series({self.a: 0.5, self.b: -0.5}, dtype="float64")
        return pd.Series(dtype="float64")


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
