"""検証ハーネス End-to-End デモ（合成データ）。

多重検定の罠を実演：真のエッジが無い20個のランダム戦略から「最良」を選ぶと、
生 Sharpe は高く見えるが DSR はそれが偽物だと見抜く。さらに CPCV で
Sharpe を「分布」として観察する。

実行（リポジトリ root から）: python examples/validation_harness_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 単体実行時にリポジトリ root を import パスへ追加（pytest は pyproject で解決）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.validation import (  # noqa: E402
    CombinatorialPurgedKFold,
    TrialRegistry,
    sharpe_ratio,
)

PERIODS_PER_YEAR = 252


def annualize(sr_per_period: float) -> float:
    return sr_per_period * np.sqrt(PERIODS_PER_YEAR)


def main() -> None:
    rng = np.random.default_rng(42)
    n_obs = 1000
    n_trials = 20
    scope = "demo_random"

    reg = TrialRegistry(":memory:")
    best = {"sharpe": -np.inf, "uuid": None, "returns": None, "i": -1}

    # --- 真のエッジが無いランダム戦略を20本試行し、全てをレジストリに記録 ---
    for i in range(n_trials):
        returns = rng.normal(0.0, 0.01, size=n_obs)  # 期待リターン0 = エッジ無し
        sr = sharpe_ratio(returns)
        tid = reg.preregister(
            scope=scope,
            hypothesis=f"random configuration #{i} has positive edge",
            economic_rationale="NONE - this is a noise strategy (demonstration only)",
            strategy_id=f"rand-{i}",
        )
        reg.record_result(tid, sharpe=sr, n_obs=n_obs, skew=0.0, kurt=3.0)
        if sr > best["sharpe"]:
            best.update(sharpe=sr, uuid=tid, returns=returns, i=i)

    k = reg.trial_count(scope)
    var = reg.sharpe_variance(scope)
    dsr_best = reg.deflated_sharpe(best["uuid"])

    print("=== 多重検定の罠（20本のランダム戦略から最良を選択）===")
    print(f"試行数 K                  : {k}")
    print(f"試行間 Sharpe 分散         : {var:.6e}")
    print(f"最良戦略 #{best['i']} の Sharpe  : {best['sharpe']:.4f}/期 "
          f"(年率 {annualize(best['sharpe']):.2f})")
    print(f"DSR（真Sharpe>0の確率）    : {dsr_best:.3f}")
    verdict = "偽物を示唆（DSR<0.95）" if dsr_best < 0.95 else "(まれに)高DSR"
    print(f"→ 生Sharpeは高く見えるが、DSRは {verdict}")

    # --- CPCV：最良戦略の Sharpe を「分布」として観察 ---
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    X = pd.DataFrame({"ret": best["returns"]}, index=idx)
    t1 = pd.Series(idx, index=idx)
    cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo_pct=0.01)
    path_sr = [
        annualize(sharpe_ratio(best["returns"][test_idx]))
        for _, test_idx in cv.split(X, t1)
        if best["returns"][test_idx].std(ddof=1) > 0
    ]
    path_sr = np.array(path_sr)
    print("\n=== CPCV による Sharpe 分布（年率, 最良戦略）===")
    print(f"分割数                     : {cv.get_n_splits()}  パス数: {cv.get_n_paths()}")
    print(f"テストSharpe 平均/標準偏差   : {path_sr.mean():.2f} / {path_sr.std():.2f}")
    print(f"テストSharpe 範囲            : [{path_sr.min():.2f}, {path_sr.max():.2f}]")
    print("→ 点推定でなく分布で見れば、ばらつき＝過信の危険が定量化できる")

    reg.close()


if __name__ == "__main__":
    main()
