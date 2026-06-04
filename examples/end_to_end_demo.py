"""最小 End-to-End デモ：全部品を連結した「過学習を封じた戦略評価ループ」。

パイプライン：
  合成価格 → 分数階差分＋モメンタム特徴量 → トリプルバリア・ラベル
  → purged CPCV で一次モデル学習 → φ本のパスの Sharpe 分布
  → 試行レジストリ → DSR（多重検定補正後の有意性）

Part 1: 単一戦略の OOS Sharpe を「分布」として観察（点推定の否定）。
Part 2: K 個の設定を試して最良を選ぶ → DSR が多重検定の膨張を補正。

実行（リポジトリ root から）: python examples/end_to_end_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from invest_system.backtest.cpcv_backtest import cpcv_backtest  # noqa: E402
from invest_system.labeling.triple_barrier import (  # noqa: E402
    get_bins,
    get_events,
    get_vertical_barriers,
    get_vol,
)
from invest_system.features.frac_diff import frac_diff_ffd  # noqa: E402
from invest_system.validation.cpcv import CombinatorialPurgedKFold  # noqa: E402
from invest_system.validation.registry import TrialRegistry  # noqa: E402


def build_dataset(rng, n=2500, momentum=0.30):
    """モメンタムを仕込んだ合成価格から特徴量とトリプルバリア・ラベルを構築。"""
    eps = rng.normal(0.0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = momentum * r[t - 1] + eps[t]      # 弱い自己相関（学習可能な構造）
    close = pd.Series(
        1000.0 * np.exp(np.cumsum(r)),
        index=pd.date_range("2018-01-01", periods=n, freq="h"), name="close")

    feats = pd.DataFrame({
        "fd": frac_diff_ffd(close, 0.4, thresh=1e-3),
        "r1": close.pct_change(1, fill_method=None),
        "r3": close.pct_change(3, fill_method=None),
        "r10": close.pct_change(10, fill_method=None),
        "vol": get_vol(close, span=50),
    }).dropna()

    t_events = feats.index
    vb = get_vertical_barriers(close, t_events, num_bars=10)
    events = get_events(close, vb.index, pt_sl=[1.5, 1.5], trgt=feats["vol"],
                        min_ret=0.0, vertical_barriers=vb)
    bins = get_bins(events, close)
    bins = bins[bins["bin"] != 0]                # 0 ラベル（同値）は除外

    idx = bins.index
    return feats.loc[idx], bins["bin"].astype(int), bins["ret"], bins["t1"]


def main() -> None:
    rng = np.random.default_rng(0)
    X, y, ret, t1 = build_dataset(rng)
    cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo_pct=0.01)
    print(f"イベント数 = {len(X)}  特徴量 = {list(X.columns)}  "
          f"CPCV分割 = {cv.get_n_splits()}  パス = {cv.get_n_paths()}\n")

    # === Part 1：単一戦略の OOS Sharpe 分布 ===
    def factory():
        return RandomForestClassifier(n_estimators=80, max_depth=4,
                                      random_state=0, n_jobs=-1)

    res = cpcv_backtest(X, y, ret, t1, cv, factory)
    print("=== Part 1: CPCV による OOS Sharpe 分布（per-period, 単一戦略）===")
    print(f"平均 / 標準偏差 : {res.mean_sharpe:.4f} / {res.std_sharpe:.4f}")
    print(f"範囲           : [{res.min_sharpe:.4f}, {res.max_sharpe:.4f}]")
    print(f"負のパス割合    : {res.frac_negative:.0%}")
    print("→ 点推定でなく分布。負のパスが多いほど過学習/脆弱の疑い\n")

    # === Part 2：K 設定の探索 → 試行レジストリ → DSR ===
    reg = TrialRegistry(":memory:")
    scope = "e2e_rf"
    configs = [
        {"max_depth": d, "max_features": mf, "random_state": s}
        for s, (d, mf) in enumerate(
            [(2, 0.5), (3, 0.5), (3, 1.0), (4, 0.5), (4, 1.0),
             (5, 0.5), (5, 1.0), (6, 0.5)])
    ]
    best = {"sr": -np.inf, "uuid": None, "cfg": None, "res": None}
    for c in configs:
        def factory(c=c):
            return RandomForestClassifier(n_estimators=80, n_jobs=-1, **c)
        r = cpcv_backtest(X, y, ret, t1, cv, factory)
        tid = reg.preregister(
            scope=scope,
            hypothesis=f"RF(depth={c['max_depth']},mf={c['max_features']}) has edge",
            economic_rationale="momentum autocorrelation captured by lagged-return features",
            params=c)
        reg.record_result(tid, sharpe=r.mean_sharpe, n_obs=len(y),
                          skew=0.0, kurt=3.0)
        if r.mean_sharpe > best["sr"]:
            best.update(sr=r.mean_sharpe, uuid=tid, cfg=c, res=r)

    dsr = reg.deflated_sharpe(best["uuid"])
    print(f"=== Part 2: {len(configs)} 設定を探索し最良を選択 → DSR で補正 ===")
    print(f"試行数 K            : {reg.trial_count(scope)}")
    print(f"最良設定            : depth={best['cfg']['max_depth']}, "
          f"max_features={best['cfg']['max_features']}")
    print(f"最良の平均Sharpe     : {best['sr']:.4f}/期  "
          f"(負パス割合 {best['res'].frac_negative:.0%})")
    print(f"DSR（真Sharpe>0確率）: {dsr:.3f}")
    verdict = "有意（DSR≥0.95）" if dsr >= 0.95 else "未達（DSR<0.95）＝多重検定で割引かれた"
    print(f"→ 生の最良Sharpeに対し、K設定を試した事実を補正 → {verdict}")
    reg.close()


if __name__ == "__main__":
    main()
