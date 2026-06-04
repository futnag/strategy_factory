"""実 bitbank データでの End-to-End（Phase 1）。

実 BTC/JPY 4時間足を bitbank 公開API（キー不要）から取得し、研究スタック全体を通す：
  分数階差分 d 探索 → 特徴量 → トリプルバリア → purged CPCV → DSR。

honest設計：
  - DSR の観測数には L6 の「実効サンプル数（Σ独自性）」を用い、ラベルの重なり（非IID）を補正。
  - 8 設定を試し最良を選ぶ → DSR が多重検定の膨張を補正。
  - 素朴な特徴量で DSR が低く出るのは「フレームワークが偽アルファを作らない」正しい挙動。
    真のアウトオブサンプルは実データでのみ得られる（López de Prado）。

実行（リポジトリ root から）: python examples/bitbank_e2e.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from invest_system.backtest.cpcv_backtest import cpcv_backtest  # noqa: E402
from invest_system.data.sources.bitbank import fetch_candlesticks  # noqa: E402
from invest_system.features.frac_diff import find_min_d, frac_diff_ffd  # noqa: E402
from invest_system.labeling.triple_barrier import (  # noqa: E402
    get_bins,
    get_events,
    get_vertical_barriers,
    get_vol,
)
from invest_system.sampling.uniqueness import average_uniqueness  # noqa: E402
from invest_system.validation.cpcv import CombinatorialPurgedKFold  # noqa: E402
from invest_system.validation.registry import TrialRegistry  # noqa: E402

YEARS = ("2023", "2024", "2025", "2026")


def main() -> None:
    print(f"bitbank 公開APIから 4時間足 BTC/JPY を取得中 ... {YEARS}")
    close = fetch_candlesticks("btc_jpy", "4hour", YEARS)["close"]
    print(f"取得: {len(close)} 本  期間 {close.index[0].date()} 〜 {close.index[-1].date()}  "
          f"価格 {close.iloc[0]:,.0f} → {close.iloc[-1]:,.0f} JPY\n")

    # 1) 分数階差分の d 探索（実データで KB §3.1 を確認）
    min_d, corr, table = find_min_d(
        close, d_grid=np.round(np.arange(0.0, 1.01, 0.1), 2), thresh=1e-4)
    print("=== 分数階差分 d 探索（実BTC/JPY）===")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(table[["adf_stat", "corr", "stationary"]])
    d = min_d if min_d is not None else 0.4
    print(f"採用 d = {d}  (元系列との相関 {corr if corr is not None else float('nan'):.3f})\n")

    # 2) 特徴量
    feats = pd.DataFrame({
        "fd": frac_diff_ffd(close, d, thresh=1e-4),
        "r1": close.pct_change(1, fill_method=None),
        "r3": close.pct_change(3, fill_method=None),
        "r6": close.pct_change(6, fill_method=None),
        "vol": get_vol(close, 50),
    }).dropna()

    # 3) トリプルバリア・ラベル（24h 保有 = 4h×6）
    t_events = feats.index
    vb = get_vertical_barriers(close, t_events, num_bars=6)
    events = get_events(close, vb.index, [1.5, 1.5], get_vol(close, 50),
                        min_ret=0.0, vertical_barriers=vb)
    bins = get_bins(events, close)
    bins = bins[bins["bin"] != 0]
    idx = bins.index
    X, y, ret, t1 = feats.loc[idx], bins["bin"].astype(int), bins["ret"], bins["t1"]

    # L6：実効サンプル数（ラベル重なりを補正）→ DSR の観測数に使う
    n_eff = int(average_uniqueness(close.index, t1).sum())
    print(f"イベント数 {len(X)}  →  実効サンプル数(Σ独自性) {n_eff}  "
          f"（見かけの {n_eff / len(X):.0%}）\n")

    # 4) 8 設定を purged CPCV で評価 → 試行レジストリ → DSR
    cv = CombinatorialPurgedKFold(6, 2, embargo_pct=0.01)
    reg = TrialRegistry(":memory:")
    scope = "btcjpy_4h_rf"
    configs = [{"max_depth": dd, "max_features": mf, "random_state": s}
               for s, (dd, mf) in enumerate(
                   [(2, 0.5), (3, 0.5), (3, 1.0), (4, 0.5),
                    (4, 1.0), (5, 0.5), (5, 1.0), (6, 0.5)])]
    best = {"sr": -np.inf, "uuid": None, "res": None, "cfg": None}
    for c in configs:
        res = cpcv_backtest(
            X, y, ret, t1, cv,
            lambda c=c: RandomForestClassifier(n_estimators=80, n_jobs=-1, **c))
        tid = reg.preregister(
            scope=scope,
            hypothesis=f"RF(depth={c['max_depth']},mf={c['max_features']}) predicts 4h direction",
            economic_rationale="short-term momentum / microstructure persistence in BTC/JPY",
            params=c)
        reg.record_result(tid, sharpe=res.mean_sharpe, n_obs=n_eff,
                          skew=0.0, kurt=3.0)
        if res.mean_sharpe > best["sr"]:
            best.update(sr=res.mean_sharpe, uuid=tid, res=res, cfg=c)

    r = best["res"]
    dsr = reg.deflated_sharpe(best["uuid"])
    print("=== purged CPCV バックテスト（実BTC/JPY, 8設定の最良）===")
    print(f"最良設定               : depth={best['cfg']['max_depth']}, "
          f"max_features={best['cfg']['max_features']}")
    print(f"OOS Sharpe 平均/標準偏差 : {r.mean_sharpe:.4f} / {r.std_sharpe:.4f}  "
          f"範囲[{r.min_sharpe:.4f}, {r.max_sharpe:.4f}]  負パス {r.frac_negative:.0%}")
    print(f"試行数 K               : {reg.trial_count(scope)}   実効観測数: {n_eff}")
    print(f"DSR（真Sharpe>0確率）  : {dsr:.3f}")
    verdict = "有意なエッジ（DSR≥0.95）" if dsr >= 0.95 else "有意なエッジは検出されず（DSR<0.95）"
    print(f"→ {verdict}")
    print("\n素朴な特徴量で DSR が低いのは正常な結果：フレームワークが偽アルファを作らない。")
    print("真のエッジ探索は、因果フィルタ・代替データ・特徴量設計の拡充で継続する。")
    reg.close()


if __name__ == "__main__":
    main()
