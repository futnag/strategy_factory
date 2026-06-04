"""サンプル独自性デモ：非IID（ラベルの重なり）を可視化。

重なりの大きいトリプルバリア・ラベルでは、見かけのサンプル数より「実効的な
独立サンプル数」が遥かに少ない。逐次ブートストラップは標準ブートストラップより
独自性の高い（重複の少ない）標本を抽出することを示す。

実行（リポジトリ root から）: python examples/uniqueness_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.labeling.triple_barrier import (  # noqa: E402
    get_bins,
    get_events,
    get_vertical_barriers,
    get_vol,
)
from invest_system.sampling.uniqueness import (  # noqa: E402
    average_uniqueness,
    get_indicator_matrix,
    num_concurrent_events,
    sample_weights_by_return,
    sequential_bootstrap,
)


def _mean_uniqueness_of_draw(ind_mat: pd.DataFrame, draw: list) -> float:
    """抽出標本（重複あり）の平均独自性。"""
    sub = ind_mat.iloc[:, draw]
    conc = sub.sum(axis=1)
    vals = []
    for k in range(sub.shape[1]):
        active = sub.iloc[:, k] > 0
        vals.append(float((1.0 / conc[active]).mean()))
    return float(np.mean(vals))


def main() -> None:
    rng = np.random.default_rng(0)
    n = 1200
    close = pd.Series(
        1000 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))),
        index=pd.date_range("2020-01-01", periods=n, freq="h"), name="close")

    trgt = get_vol(close, span=50)
    t_events = trgt.dropna().index
    vb = get_vertical_barriers(close, t_events, num_bars=20)   # 重なり大
    events = get_events(close, vb.index, pt_sl=[2, 2], trgt=trgt,
                        min_ret=0.0, vertical_barriers=vb)
    bins = get_bins(events, close)
    t1 = bins["t1"]

    conc = num_concurrent_events(close.index, t1)
    avgu = average_uniqueness(close.index, t1)
    w = sample_weights_by_return(close.index, t1, close)

    print("=== サンプル独自性（重なりの大きいトリプルバリア・ラベル）===")
    print(f"ラベル数（見かけ）        : {len(t1)}")
    print(f"同時イベント数 平均/最大   : {conc.mean():.1f} / {int(conc.max())}")
    print(f"平均独自性 mean           : {avgu.mean():.3f}")
    print(f"実効サンプル数 (Σ独自性)   : {avgu.sum():.0f}  "
          f"（見かけの {avgu.sum() / len(t1):.0%}）")
    print(f"リターン帰属重み min/max   : {w.min():.3f} / {w.max():.3f} "
          f"(平均1, 総和={w.sum():.0f})")

    # 逐次 vs 標準ブートストラップ（先頭60ラベルで比較）
    sub_t1 = t1.iloc[:60]
    ind = get_indicator_matrix(close.index, sub_t1)
    m = ind.shape[1]
    seq = sequential_bootstrap(ind, size=m, random_state=0)
    std = list(rng.integers(0, m, size=m))
    print("\n=== 逐次 vs 標準ブートストラップ（先頭60ラベル）===")
    print(f"標準ブートストラップ 平均独自性 : {_mean_uniqueness_of_draw(ind, std):.3f}")
    print(f"逐次ブートストラップ 平均独自性 : {_mean_uniqueness_of_draw(ind, seq):.3f}")
    print("→ 逐次ブートストラップの方が独自性が高い＝冗長性の低い標本を抽出")
    print("\n（これらの重みは cpcv_backtest(..., sample_weight=w) に直結）")


if __name__ == "__main__":
    main()
