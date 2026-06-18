"""throwaway 診断：価格系・流動性系ファクター（GKX Phase 1）の月次IC・被覆・冗長性（K不変）。

**判定ではない。** judge_grid / 永続レジストリ（K）は触らない＝K不変。新特徴量の予測力を
見たいだけの使い捨て計測（月次クロスセクション Spearman IC・被覆率・新旧ペアワイズ相関）で、
戦略認定ではない。PIT ユニバース（時変・生存者バイアス排除）上で評価する。

手元実行: .venv/Scripts/python.exe examples/diag_price_factors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.feature_store import load_feature  # noqa: E402
from invest_system.data.store import load_wide  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, point_in_time_universe)

FACTORS = ["mom_1m_reversal", "mom_3m", "mom_6m", "mom_36m_reversal",
           "industry_momentum", "rvol_60", "rvol_252", "ivol", "beta",
           "max_ret", "ret_skew", "amihud_illiq", "turnover", "dollar_volume",
           "zero_ret_days"]
# 新（材化済）↔ 既存材化（feature_store 日次）の冗長性チェック対
REDUNDANCY = [("mom_6m", "momentum_12_1"), ("rvol_60", "vol_20"),
              ("mom_1m_reversal", "reversal_5")]
MIN_NAMES = 20


def _cs_ranks_corr(x: pd.Series, y: pd.Series) -> float:
    c = x.dropna().index.intersection(y.dropna().index)
    if len(c) < MIN_NAMES:
        return np.nan
    return x[c].rank().corr(y[c].rank())


def monthly_ic(factor: pd.DataFrame, fwd: pd.DataFrame, mask) -> tuple:
    f = apply_universe_mask(factor, mask)
    ics, cov = [], []
    for t in f.index:
        if t not in fwd.index:
            continue
        cov.append(int(f.loc[t].notna().sum()))
        ic = _cs_ranks_corr(f.loc[t], fwd.loc[t])
        if pd.notna(ic):
            ics.append(ic)
    return ics, cov


def main() -> None:
    adj_me = load_wide("adj_close").resample("ME").last()
    if adj_me.empty:
        print("Silver adj_close が空。先に store.materialize_all を実行してください。")
        return
    fwd = adj_me.pct_change().shift(-1)                  # 翌月リターン（先読みなし）
    va_me = load_wide("turnover").resample("ME").last()  # Va（流動性）
    mask = point_in_time_universe(va_me, top_n=500, lookback=12, min_obs=6)
    print(f"PIT ユニバース：月末 {len(mask.index)} 期 / 上位500 / "
          f"{adj_me.index.min():%Y-%m}〜{adj_me.index.max():%Y-%m}\n")

    print(f"=== 月次クロスセクション IC（Spearman, 翌月リターン・PITユニバース）===")
    print(f"{'factor':<20}{'meanIC':>8}{'IR':>7}{'n月':>6}{'平均被覆':>9}")
    for name in FACTORS:
        f = load_feature(name)
        if f.empty:
            print(f"{name:<20}{'--':>8}{'(未材化)':>16}")
            continue
        ics, cov = monthly_ic(f, fwd, mask)
        if ics:
            m = float(np.mean(ics))
            ir = m / np.std(ics, ddof=1) if len(ics) > 1 and np.std(ics, ddof=1) > 0 else np.nan
            print(f"{name:<20}{m:>8.3f}{ir:>7.2f}{len(ics):>6}{int(np.mean(cov)):>9}")
        else:
            print(f"{name:<20}{'--':>8}{'被覆不足':>16}")

    print("\n=== 新旧ペアワイズ相関（冗長性・PITユニバース内の時間平均 CS Spearman）===")
    for new, old in REDUNDANCY:
        a, b = load_feature(new), load_feature(old)
        if a.empty or b.empty:
            print(f"  {new} vs {old}: 一方未材化")
            continue
        b = b.resample("ME").last() if not isinstance(b.index, pd.DatetimeIndex) or \
            b.index.inferred_freq != "ME" else b
        a, b = apply_universe_mask(a, mask), apply_universe_mask(b, mask)
        idx = a.index.intersection(b.index)
        cors = [c for c in (_cs_ranks_corr(a.loc[t], b.loc[t]) for t in idx) if pd.notna(c)]
        print(f"  {new:<18} vs {old:<16}: 平均CS相関 {np.mean(cors):+.2f}（n={len(cors)}）"
              if cors else f"  {new} vs {old}: 被覆不足")

    print("\n※ K 不変・throwaway 診断（戦略認定ではない）。符号は『大きいほどロング側』。")


if __name__ == "__main__":
    main()
