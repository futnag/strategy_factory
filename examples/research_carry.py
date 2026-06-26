r"""仮説検証 ⑦：クロスアセット・キャリー（**データ制約あり**）。

意図は「トレンド(TSMOM・既載)とは別の独立プレミア＝キャリー」。だが local データを検証した結果、
**FXキャリーは外国短期金利が無く(JP/USのみ)、商品キャリーは先物カーブが無い(現物のみ)** ため、
分散されたクロスアセット・キャリーは構築不能。唯一クリーンに作れるのは **国債のターム・キャリー
（JGB/UST＝10y − 政策金利が正なら順イールド＝正キャリー+ロールダウン）**。低 breadth（2資産）かつ
本期間は BOJ の YCC 解除・米利上げという歴史的金利正常化に支配される点を明示の上、完全性のため判定する。

債券月次/日次トータルリターンは近似（carry − Duration×Δy）。実行:
  .venv\Scripts\python.exe examples\research_carry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.external import load_macro  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, SignalTimingStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

OOS = "2024-01"
DUR = 9.0          # 10年国債の近似デュレーション


def _bond_ret(y: pd.Series) -> pd.Series:
    """利回り(%)系列 → 近似日次トータルリターン（キャリー − Dur×Δy）。"""
    dy = y.diff()
    return (y / 100.0 / 252.0) - DUR * (dy / 100.0)


def main() -> int:
    print("=== ⑦ クロスアセット・キャリー（国債ターム・キャリーのみ＝データ制約）===")
    mac = load_macro(["jp_10y", "jp_policy", "us_10y", "us_ff"]).ffill().dropna(how="all")
    jgb_ret, ust_ret = _bond_ret(mac["jp_10y"]), _bond_ret(mac["us_10y"])
    bond_px = pd.DataFrame({
        "jgb": (1.0 + jgb_ret.fillna(0.0)).cumprod() * 100.0,
        "ust": (1.0 + ust_ret.fillna(0.0)).cumprod() * 100.0,
    }).dropna()
    term_jgb = (mac["jp_10y"] - mac["jp_policy"]).reindex(bond_px.index)
    term_ust = (mac["us_10y"] - mac["us_ff"]).reindex(bond_px.index)
    view = AsOfView({"close": bond_px})
    print(f"国債近似トータルリターン {len(bond_px)} 営業日"
          f"（{bond_px.index.min():%Y-%m}〜{bond_px.index.max():%Y-%m}）")
    print(f"  直近ターム: JGB(10y−policy)={term_jgb.iloc[-1]:+.2f}%  "
          f"UST(10y−ff)={term_ust.iloc[-1]:+.2f}%")

    # 正キャリー（順イールド）なら当該国債をロング（long/flat）。前日基準＝PIT
    grid = [
        SignalTimingStrategy(term_jgb.shift(1).dropna(), "jgb", 0.0, 1, name="jgb_termcarry"),
        SignalTimingStrategy(term_ust.shift(1).dropna(), "ust", 0.0, 1, name="ust_termcarry"),
    ]
    with default_registry() as reg:
        v = judge_grid(
            grid, view, scope="bond_term_carry",
            hypothesis="順イールド（10y−政策金利>0＝正キャリー+ロールダウン）の国債をロングすると"
                       "ターム・プレミアムを収穫できる（トレンドと別系統のキャリー）",
            economic_rationale="ターム・プレミアム/キャリー&ロールは債券の主要な期待超過収益源で、"
                               "反対側にヘッジ需要・規制資本・中銀という価格非感応の主体がいる。TSMOM(トレンド)とは別系統。",
            registry=reg, costs_bps=2.0, execution_lag=0)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜）・買い持ち比較 ---")
    for code, ret in (("jgb", jgb_ret), ("ust", ust_ret)):
        bh = ret.reindex(bond_px.index).dropna()
        print(f"  買い持ち {code}: SR(ann)={sharpe_ratio(bh) * np.sqrt(252):+.2f}")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(252) if is_.size >= 60 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(252) if oos.size >= 60 else np.nan
        print(f"  {r.name:<16} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")
    print("\n  ※ 2資産・本期間は金利正常化に支配＝低 breadth。判定は参考（データ制約）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
