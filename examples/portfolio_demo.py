"""ポートフォリオ構築デモ：Markowitz の不安定性をノイズ除去/HRP/NCO で緩和。

観測数 T が変数数 N に近い（ノイズの多い）共分散で、Markowitz 最小分散は逆行列により
誤差を増幅し、極端・高レバ・不安定な重みを生む。RMT ノイズ除去・HRP・NCO がこれを
どれだけ緩和するかを、条件数・集中度・サブサンプル安定性で比較する。

実行（リポジトリ root から）: python examples/portfolio_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.portfolio.allocation import (  # noqa: E402
    hrp_weights,
    min_variance_weights,
    nco_weights,
)
from invest_system.portfolio.denoise import cov_to_corr, denoise_covariance  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    n, k, t = 30, 3, 50                      # T≈N → ノイズ過多
    loadings = rng.normal(0, 1, (n, k))
    factors = rng.normal(0, 1, (t, k))
    returns = factors @ loadings.T + rng.normal(0, 0.5, (t, n))
    assets = [f"a{i:02d}" for i in range(n)]
    cov = pd.DataFrame(np.cov(returns, rowvar=False), index=assets, columns=assets)
    cov_dn = denoise_covariance(cov, q=t / n)

    def cond(c):
        v = c.values if isinstance(c, pd.DataFrame) else c
        return np.linalg.cond(cov_to_corr(v))

    print("=== 共分散ノイズ除去（RMT / Marčenko-Pastur）===")
    print(f"相関行列の条件数 : 元 {cond(cov):.1f}  →  ノイズ除去後 {cond(cov_dn):.1f}")

    methods = {
        "Markowitz(min-var)": min_variance_weights(cov),
        "Markowitz+ノイズ除去": min_variance_weights(cov_dn),
        "HRP": hrp_weights(cov),
        "NCO(ノイズ除去後)": nco_weights(cov_dn),
    }
    print("\n=== 配分の集中度・レバレッジ ===")
    print(f"{'手法':24}{'最大|w|':>9}{'負の数':>8}{'Σ|w|':>9}")
    for name, w in methods.items():
        print(f"{name:24}{w.abs().max():>9.2f}{int((w < 0).sum()):>8}{w.abs().sum():>9.2f}")

    # 安定性：ブートストラップ再標本で重みがどれだけ動くか（小さいほど安定）
    boot_rng = np.random.default_rng(1)
    print("\n=== 安定性（ブートストラップ再標本での重み変化 平均Σ|Δw|, 小さいほど安定）===")
    for name, fn in (("Markowitz(min-var)", min_variance_weights),
                     ("HRP", hrp_weights), ("NCO", nco_weights)):
        base = fn(cov)
        deltas = []
        for _ in range(5):
            idx = boot_rng.integers(0, t, t)
            cov_b = pd.DataFrame(np.cov(returns[idx], rowvar=False),
                                 index=assets, columns=assets)
            deltas.append(float((fn(cov_b) - base).abs().sum()))
        print(f"{name:24}{np.mean(deltas):>9.2f}")
    print("→ Markowitz は逆行列で誤差増幅＝集中・高レバ・不安定。ノイズ除去/HRP/NCO で緩和。")


if __name__ == "__main__":
    main()
