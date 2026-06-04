"""トリプルバリア・ラベリングのデモ（合成バー）。

ランダムウォーク価格に対し、動的ボラ目標でトリプルバリアを適用し、
ラベル分布・各バリアの接触割合・平均保有期間を表示する。
固定ホライズン法と異なり、途中のストップアウト（パス依存）を反映する。

実行（リポジトリ root から）: python examples/triple_barrier_demo.py
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


def main() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    close = pd.Series(
        1000 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))),
        index=pd.date_range("2020-01-01", periods=n, freq="h"), name="close")

    trgt = get_vol(close, span=50, lookback=1).dropna()
    t_events = trgt.index                       # 各バーをイベントとする
    vb = get_vertical_barriers(close, t_events, num_bars=5)
    events = get_events(close, vb.index, pt_sl=[2, 2], trgt=trgt,
                        min_ret=0.0, vertical_barriers=vb)
    bins = get_bins(events, close)

    # 各バリアの接触内訳：垂直＝t1 が垂直バリアと一致、そうでなければ符号で利確/損切
    is_vertical = events["t1"].eq(vb.reindex(events.index))
    n_vert = int(is_vertical.sum())
    n_pt = int(((~is_vertical) & (bins["bin"] > 0)).sum())
    n_sl = int(((~is_vertical) & (bins["bin"] < 0)).sum())

    pos = close.index.searchsorted
    holding = pos(bins["t1"].to_numpy()) - pos(bins.index.to_numpy())

    print("=== トリプルバリア・ラベリング（合成バー）===")
    print(f"イベント数              : {len(bins)}")
    print(f"ラベル分布 (+1/0/-1)    : "
          f"{int((bins['bin'] > 0).sum())} / "
          f"{int((bins['bin'] == 0).sum())} / "
          f"{int((bins['bin'] < 0).sum())}")
    print(f"接触内訳 (利確/損切/時間): {n_pt} / {n_sl} / {n_vert}")
    print(f"平均保有バー数          : {holding.mean():.1f}  "
          f"(最大 {int(holding.max())} = 垂直バリア)")
    print(f"平均実現リターン        : {bins['ret'].mean() * 100:.3f}%")
    print("→ 固定ホライズン法と違い、途中の利確/損切（パス依存）を反映してラベル化")


if __name__ == "__main__":
    main()
