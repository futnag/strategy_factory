"""因果フィルタのデモ：コライダーバイアスの符号反転と特徴量選別。

Part 1: 合流点（collider）を回帰に含めると、正のリスクプレミアムが負に「符号反転」
        する様子を実証（資料群の核心的警告）。
Part 2: ペアワイズ LiNGAM で各特徴量を「原因（採用）/結果＝コライダー（除外）」に
        分類し、下流変数を自動除去する。

実行（リポジトリ root から）: python examples/causal_filter_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.features.causal import (  # noqa: E402
    causal_filter,
    classify_features,
    collider_bias_beta,
)

_S = 1.0 / np.sqrt(2.0)   # 単位分散ラプラスの scale


def part1_collider_bias() -> None:
    rng = np.random.default_rng(0)
    n = 50000
    x = rng.laplace(0, _S, n)
    eps = rng.laplace(0, _S, n)
    zeta = rng.laplace(0, _S, n)
    beta, gamma, delta = 0.5, 1.0, 2.0
    y = beta * x + eps                       # x は y の真の原因（+0.5）
    z = gamma * y + delta * x + zeta         # z は x と y の共通の結果（合流点）

    b_simple = np.cov(x, y)[0, 1] / np.var(x)
    W = np.column_stack([x, z, np.ones(n)])
    b_multi = np.linalg.lstsq(W, y, rcond=None)[0][0]

    print("=== Part 1: コライダーバイアスによる符号反転 ===")
    print(f"真の効果 β                       : +{beta:.2f}")
    print(f"x のみで回帰（正しい）           : {b_simple:+.3f}")
    print(f"合流点 z を制御（誤り）          : {b_multi:+.3f}")
    print(f"閉形式 (β−δγ)/(1+γ²)             : {collider_bias_beta(beta, delta, gamma):+.3f}")
    print("→ 合流点を入れると買いシグナル(+)が売りシグナル(−)に反転\n")


def part2_causal_filter() -> None:
    rng = np.random.default_rng(7)
    n = 8000
    c1 = rng.laplace(0, _S, n)
    c2 = rng.laplace(0, _S, n)
    y = 0.7 * c1 + 0.5 * c2 + 0.4 * rng.laplace(0, _S, n)   # c1,c2 が y の原因
    e1 = 0.8 * y + 0.4 * rng.laplace(0, _S, n)              # y の下流（コライダー）
    e2 = -0.6 * y + 0.4 * rng.laplace(0, _S, n)             # y の下流（負係数）

    X = pd.DataFrame({
        "signal_c1 (cause)": c1,
        "signal_c2 (cause)": c2,
        "realized_e1 (collider)": e1,
        "drawdown_e2 (collider)": e2,
    })
    cls = classify_features(X, y)
    filtered, _ = causal_filter(X, y)

    print("=== Part 2: 因果フィルタによる特徴量選別 ===")
    pd.set_option("display.float_format", lambda v: f"{v:+.3f}")
    print(cls)
    print(f"\n採用（原因）: {list(filtered.columns)}")
    print("→ y の下流（コライダー）を自動除去。残った原因のみがリスクプレミアムの帰属に妥当")
    print("（注：非ガウス性に依存。ガウスデータでは方向同定は不能）")


def main() -> None:
    part1_collider_bias()
    part2_causal_filter()


if __name__ == "__main__":
    main()
