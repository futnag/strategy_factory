"""メタラベリング・デモ：補正的AIが Precision を高める。

「モメンタムが効くのは低ボラ局面だけ」という構造を仕込む。一次モデル（モメンタムの
符号）は高Recall・低Precision。二次（メタ）モデルがボラ特徴から「いつ賭けるか」を
学習し、空振りを排除して Precision を高める様子を、パージ済ホールドアウトで示す。

実行（リポジトリ root から）: python examples/meta_labeling_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from invest_system.labeling.meta_labeling import bet_size_from_prob  # noqa: E402
from invest_system.labeling.triple_barrier import (  # noqa: E402
    get_bins,
    get_events,
    get_vertical_barriers,
    get_vol,
)


def build():
    rng = np.random.default_rng(3)
    n, block = 4000, 40
    n_blocks = n // block + 1
    low_vol = np.repeat(rng.random(n_blocks) < 0.5, block)[:n]   # 局面（ブロック単位）
    mom = np.where(low_vol, 0.50, -0.50)            # 低ボラ=モメンタム / 高ボラ=平均回帰
    sig = np.where(low_vol, 0.006, 0.020)
    eps = rng.normal(0.0, sig)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = mom[t] * r[t - 1] + eps[t]
    close = pd.Series(1000 * np.exp(np.cumsum(r)),
                      index=pd.date_range("2020-01-01", periods=n, freq="h"),
                      name="close")

    feats = pd.DataFrame({
        "r1": close.pct_change(1, fill_method=None),
        "r3": close.pct_change(3, fill_method=None),
        "r10": close.pct_change(10, fill_method=None),
        "rvol": close.pct_change(1, fill_method=None).rolling(10).std(),
    }).dropna()

    t_events = feats.index
    side = np.sign(feats["r3"]).replace(0.0, 1.0)               # 一次モデル＝モメンタム符号
    vb = get_vertical_barriers(close, t_events, num_bars=6)
    side = side.reindex(vb.index)
    events = get_events(close, vb.index, [1, 1], get_vol(close, 50),
                        min_ret=0.0, vertical_barriers=vb, side=side)
    bins = get_bins(events, close)                             # bin∈{0,1}, ret は side調整済
    idx = bins.index
    return feats.loc[idx], bins["bin"].astype(int), bins["ret"], bins["t1"]


def main() -> None:
    X, meta_y, ret, t1 = build()
    n = len(X)
    cut = X.index[int(n * 0.6)]

    # パージ済ホールドアウト：train の t1 が test 開始以降に重なるものを除外
    is_test = X.index >= cut
    is_train = (X.index < cut) & (t1.to_numpy() < cut)
    Xtr, ytr = X[is_train], meta_y[is_train]
    Xte = X[is_test]
    ret_te = ret.to_numpy()[is_test]

    clf = RandomForestClassifier(n_estimators=120, max_depth=4,
                                 random_state=0, n_jobs=-1).fit(Xtr, ytr)
    col1 = list(clf.classes_).index(1)
    meta_prob = clf.predict_proba(Xte)[:, col1]

    # 一次モデルのみ（全ベット）vs メタラベリング（確度で取捨＋サイジング）
    size = np.clip(bet_size_from_prob(meta_prob), 0.0, 1.0)
    acts = size > 0
    meta_ret = size * ret_te

    def sharpe(x):
        x = x[x != 0]
        return x.mean() / x.std() if x.size > 1 and x.std() > 0 else float("nan")

    print("=== メタラベリング：一次モデルのみ vs メタ補正（パージ済ホールドアウト）===")
    print(f"テスト・ベット候補数      : {is_test.sum()}")
    print(f"{'':22}{'一次のみ':>12}{'メタ補正':>12}")
    print(f"{'実行ベット数':20}{is_test.sum():>12}{int(acts.sum()):>12}")
    print(f"{'Precision(勝率)':20}{(ret_te > 0).mean():>12.1%}"
          f"{(ret_te[acts] > 0).mean():>12.1%}")
    print(f"{'Sharpe(per-bet)':20}{sharpe(ret_te):>12.3f}{sharpe(meta_ret):>12.3f}")
    print(f"{'平均リターン/bet':20}{ret_te.mean() * 100:>11.3f}%"
          f"{meta_ret[acts].mean() * 100:>11.3f}%")
    print("→ メタモデルが低ボラ局面に絞って参加 → 賭け数を抑え Precision/Sharpe を改善")


if __name__ == "__main__":
    main()
