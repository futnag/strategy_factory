"""目標ウェイト → 発注リスト（Phase 2 実装仕様・docs/02 D5）。

小資金の現実への写像（純関数・ネット不要・テスト可能）：
- 株式スリーブ：ロング脚は単元未満株（かぶミニ等・1株単位）で個別保有。ショート脚は
  小資金では個別信用売りが構成不能（60銘柄×最低単元≫資本）のため、**合計ショート想定
  元本を指数先物（日経225マイクロ）の売りヘッジに置換**する。置換によるトラッキング
  エラーは「モデルの変更」ではなく「執行近似」であり、月次照合レポートの計測対象。
- TSMOM スリーブ：商品ごとのロット表（想定元本/ロット・刻み）で丸め、丸め誤差
  （実装ショートフォール）を明示する。ペーパー段階では刻み None＝想定元本のまま保持可。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_orders(weights: pd.Series, prices: pd.Series, capital: float,
                  lot: int = 1) -> tuple[pd.DataFrame, float]:
    """株式スリーブのロング注文表と、ショート脚の想定元本（ヘッジ対象）を返す。

    weights: 目標ウェイト（正=ロング個別保有、負=ショート→指数ヘッジへ集約）。
    prices : 発注参考価格（**無調整の実勢株価**。調整後ではない）。
    capital: スリーブ資本（円）。lot: 株数単位（かぶミニ=1株）。
    Returns: (orders, short_notional)
      orders 列 = [code, weight, price, shares, yen]（shares>0 のロングのみ。
      価格欠損・1株も買えない極小配分は脱落し、欠落分は照合で shortfall として現れる）。
    """
    w = weights.dropna()
    longs = w[w > 0]
    px = prices.reindex(longs.index)
    target_yen = longs * float(capital)
    shares = np.floor(target_yen / px / lot) * lot
    orders = pd.DataFrame({
        "code": longs.index.astype(str),
        "weight": longs.values,
        "price": px.values,
        "shares": shares.values,
    })
    orders = orders[orders["price"].notna() & (orders["shares"] > 0)]
    orders["shares"] = orders["shares"].astype(int)
    orders["yen"] = orders["shares"] * orders["price"]
    short_notional = float(w[w < 0].abs().sum() * capital)
    return orders.reset_index(drop=True), short_notional


def hedge_contracts(short_notional: float, index_price: float,
                    multiplier: float = 10.0) -> tuple[int, float]:
    """ショート想定元本 → 指数先物の売り枚数（最近接整数）と実ヘッジ額。

    日経225マイクロ＝指数×10円（multiplier=10）。枚数の粗さによる過小/過大ヘッジは
    返り値の実ヘッジ額（contracts×index_price×multiplier）との差で照合する。
    """
    if index_price <= 0 or short_notional <= 0:
        return 0, 0.0
    unit = index_price * multiplier
    n = int(round(short_notional / unit))
    return n, n * unit


def lot_orders(weights: pd.Series, prices: pd.Series, capital: float,
               lot_units: dict, lot_steps: dict | None = None) -> pd.DataFrame:
    """TSMOM スリーブ：目標想定元本をロット表で丸めた注文表。

    lot_units: {asset: 1ロットの数量係数}（1ロット想定元本 = price×unit。例：日経225
      マイクロ=10）。lot_steps: {asset: 刻み}（マイクロ先物=1、CFD=0.1、None=丸めない
      ＝ペーパーで想定元本のまま保持）。
    Returns 列 = [asset, weight, target_yen, lots, filled_yen, shortfall_yen]。
    """
    lot_steps = lot_steps or {}
    rows = []
    for a, w in weights.dropna().items():
        px = prices.get(a, np.nan)
        target = float(w * capital)
        unit = lot_units.get(a)
        step = lot_steps.get(a)
        if unit is None or pd.isna(px) or px <= 0:
            rows.append((a, w, target, np.nan, target, 0.0))
            continue
        lot_value = px * unit
        raw = target / lot_value
        lots = raw if step is None else round(raw / step) * step
        filled = lots * lot_value
        rows.append((a, w, target, lots, filled, target - filled))
    return pd.DataFrame(rows, columns=["asset", "weight", "target_yen", "lots",
                                       "filled_yen", "shortfall_yen"])
