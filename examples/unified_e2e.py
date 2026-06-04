"""全部品統合 End-to-End（capstone）。

これまで作った全モジュールを1本のパイプラインに統合する：
  L3 分数階差分 → L4 因果フィルタ → L5 トリプルバリア
  → L6 サンプル独自性重み → L7 メタラベリング → L8 DSR（実効サンプル数で補正）

仕掛け：特徴量に意図的な「コライダー（リターンの下流＝先読みリーク）」を混入させ、
  (1) L4 因果フィルタがそれを検出・除去してリークを防ぐこと、
  (2) 残った原因特徴＋独自性重み＋メタラベリングが正直な戦略を作ること
を実証する。データは「低ボラ=モメンタム/高ボラ=平均回帰」の局面構造を持つ。

実行（リポジトリ root から）: python examples/unified_e2e.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from invest_system.features.causal import causal_filter, classify_features  # noqa: E402
from invest_system.features.frac_diff import frac_diff_ffd  # noqa: E402
from invest_system.labeling.meta_labeling import bet_size_from_prob  # noqa: E402
from invest_system.labeling.triple_barrier import (  # noqa: E402
    get_bins,
    get_events,
    get_vertical_barriers,
    get_vol,
)
from invest_system.sampling.uniqueness import (  # noqa: E402
    average_uniqueness,
    sample_weights_by_return,
)
from invest_system.validation import dsr as dsrmod  # noqa: E402


def build():
    rng = np.random.default_rng(3)
    n, block = 5000, 40
    nb = n // block + 1
    low = np.repeat(rng.random(nb) < 0.5, block)[:n]
    mom = np.where(low, 0.55, -0.55)                 # 低ボラ=モメンタム/高ボラ=平均回帰
    sig = np.where(low, 0.006, 0.020)
    eps = rng.normal(0.0, sig)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = mom[t] * r[t - 1] + eps[t]
    close = pd.Series(1000.0 * np.exp(np.cumsum(r)),
                      index=pd.date_range("2020-01-01", periods=n, freq="h"), name="close")

    feats = pd.DataFrame({                            # L3：原因候補（分数階差分＋モメンタム）
        "fd": frac_diff_ffd(close, 0.4, thresh=1e-3),
        "r1": close.pct_change(1, fill_method=None),
        "r3": close.pct_change(3, fill_method=None),
        "r10": close.pct_change(10, fill_method=None),
        "rvol": close.pct_change(1, fill_method=None).rolling(10).std(),
    }).dropna()

    t_events = feats.index
    side = np.sign(feats["r3"]).replace(0.0, 1.0)     # 一次モデル＝モメンタム符号
    vb = get_vertical_barriers(close, t_events, num_bars=6)
    side = side.reindex(vb.index)
    events = get_events(close, vb.index, [1, 1], get_vol(close, 50),
                        min_ret=0.0, vertical_barriers=vb, side=side)
    bins = get_bins(events, close)                    # ret(side調整), bin{0,1}, t1
    idx = bins.index
    return close, feats.loc[idx], bins, rng


def main() -> None:
    close, feats, bins, rng = build()
    ret, meta_y, t1 = bins["ret"], bins["bin"].astype(int), bins["t1"]

    # 意図的なコライダー：実現リターンの下流（先読みリーク）
    collider = ret.to_numpy() + 0.5 * ret.std() * rng.laplace(0, 1, len(ret))
    X_all = feats.copy()
    X_all["leaky_collider"] = collider

    # === [L4] 因果フィルタ（target = 連続リターン）===
    # 閾値 -0.05：ノイズの多い金融データでは「明確に下流（コライダー）」のみ除外し、
    # 方向が曖昧（score≈0）な特徴は保持する保守的な運用にする。
    thr = -0.05
    cls = classify_features(X_all, ret, threshold=thr)
    X_causal, _ = causal_filter(X_all, ret, threshold=thr)
    pd.set_option("display.float_format", lambda v: f"{v:+.4f}")
    print(f"=== [L4] 因果フィルタ：方向スコア（閾値 {thr} 未満＝明確な下流/コライダー→除外）===")
    print(cls.sort_values("score", ascending=False))
    print(f"採用（原因）: {list(X_causal.columns)}\n")

    # === リーク検証：コライダー込み vs 因果フィルタ後 ===
    cut_t = X_all.index[int(len(X_all) * 0.6)]
    is_test = X_all.index >= cut_t
    is_train = (X_all.index < cut_t) & (t1.to_numpy() < cut_t)      # パージ

    def test_precision(X):
        clf = RandomForestClassifier(80, max_depth=4, random_state=0,
                                     n_jobs=-1).fit(X[is_train], meta_y[is_train])
        prob = clf.predict_proba(X[is_test])[:, list(clf.classes_).index(1)]
        acts = prob > 0.5
        hit = (ret.to_numpy()[is_test][acts] > 0).mean() if acts.any() else float("nan")
        return hit, int(acts.sum())

    p_leak, n_leak = test_precision(X_all)
    p_clean, n_clean = test_precision(X_causal)
    print("=== リーク検証（test precision）===")
    print(f"コライダー込み : {p_leak:.1%} (act {n_leak})  ← 未来情報リークで過大評価")
    print(f"因果フィルタ後 : {p_clean:.1%} (act {n_clean})  ← 正常\n")

    # === [L3+L4+L6+L7] 正直な統合パイプライン ===
    w = sample_weights_by_return(close.index, t1, close)            # L6 独自性重み
    clf = RandomForestClassifier(120, max_depth=4, random_state=0, n_jobs=-1)
    clf.fit(X_causal[is_train], meta_y[is_train], sample_weight=w[is_train].to_numpy())
    meta_prob = clf.predict_proba(X_causal[is_test])[:, list(clf.classes_).index(1)]
    size = np.clip(bet_size_from_prob(meta_prob), 0.0, 1.0)         # L7 ベットサイジング
    ret_te = ret.to_numpy()[is_test]
    acts = size > 0

    def sharpe(x):
        x = x[x != 0]
        return x.mean() / x.std() if x.size > 1 and x.std() > 0 else float("nan")

    n_eff = int(average_uniqueness(close.index, t1[is_test]).sum())  # L6 実効サンプル数
    dsr = dsrmod.deflated_sharpe_ratio(sharpe(size * ret_te), sr_variance=0.0,
                                       n_trials=1, n_obs=max(n_eff, 2),
                                       skew=0.0, kurt=3.0)
    print("=== [L3+L4+L6+L7] 統合パイプライン（パージ済 test）===")
    print(f"{'':22}{'一次のみ':>12}{'メタ補正':>12}")
    print(f"{'実行ベット数':20}{is_test.sum():>12}{int(acts.sum()):>12}")
    print(f"{'Precision(勝率)':20}{(ret_te > 0).mean():>12.1%}"
          f"{(ret_te[acts] > 0).mean():>12.1%}")
    print(f"{'Sharpe(per-bet)':20}{sharpe(ret_te):>12.3f}{sharpe(size * ret_te):>12.3f}")
    print(f"\nメタ戦略 DSR（実効n={n_eff}, 単一設定）: {dsr:.3f}")
    print("→ 因果フィルタでリークを除去し、独自性重み＋メタラベリングで Precision を改善。"
          "\n  全部品（L3/L4/L5/L6/L7/L8）が一本のパイプラインで連携する。")


if __name__ == "__main__":
    main()
