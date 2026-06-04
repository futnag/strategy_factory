"""分数階差分デモ：メモリ vs 定常性のトレードオフ（KB §3.1 の再現）。

合成価格系列について d を 0→1 で動かし、ADF統計量と元系列との相関を表示。
小さな d で定常性を確保しつつ高い相関（メモリ）を保てること、そして d=1 の
リターン化が相関≈0でメモリを失うことを示す。

実行（リポジトリ root から）: python examples/frac_diff_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.features.frac_diff import find_min_d, frac_diff_d_table  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(4)
    n = 2000
    # 価格 = ドリフト無しのランダムウォーク（I(1)：d=0 では非定常）
    price = pd.Series(1000.0 + np.cumsum(rng.normal(0.0, 1.0, n)),
                      index=pd.date_range("2015-01-01", periods=n, freq="B"),
                      name="price")

    # thresh をやや緩める：d→0 で窓幅が系列長を超えないようにし小さな d も評価
    table = frac_diff_d_table(
        price, d_grid=np.round(np.arange(0.0, 1.01, 0.1), 2), thresh=5e-4)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("=== 分数階差分：d vs ADF vs メモリ保存（KB §3.1 の再現）===")
    print(table)

    min_d, corr, _ = find_min_d(
        price, d_grid=np.round(np.arange(0.0, 1.01, 0.05), 2), thresh=5e-4)
    if min_d is None:
        print("\n指定グリッドで定常化する d が見つかりませんでした（系列長/thresh を調整）。")
    else:
        print(f"\n定常化する最小 d = {min_d}  (元系列との相関 corr = {corr:.4f})")
        print("→ d=1 のリターン化は corr≈0 でメモリ喪失。小さな d で定常性とメモリを両立。")


if __name__ == "__main__":
    main()
