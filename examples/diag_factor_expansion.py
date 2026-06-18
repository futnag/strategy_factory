"""throwaway 診断：特徴量拡充 A（信用/空売り・微細構造）＋B（fins新ファンダ）の月次IC・被覆・冗長性。

**判定ではない。** judge_grid / 永続レジストリ（K）は触らない＝K不変。PIT ユニバース（時変・
生存者バイアス排除）上で月次クロスセクション Spearman IC・被覆率・既存 factor との冗長性相関を
計測するだけ（ML 入力の素材確認）。判定は Phase 4 で一度だけ。EDINET 非依存。

手元実行: .venv/Scripts/python.exe examples/diag_factor_expansion.py
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

A1 = ["margin_imbalance", "short_to_long", "margin_balance_change", "short_interest",
      "days_to_cover", "sector_short_ratio"]
A2 = ["parkinson_vol", "garman_klass_vol", "roll_spread", "corwin_schultz_spread",
      "vpin", "rsi"]
B = ["sue_recent", "sue_initial", "forecast_revision", "sales_growth", "profit_growth",
     "equity_growth", "roe_stability", "margin_stability", "sustainable_growth",
     "high_52w", "seasonality", "dimson_beta"]
# 新↔既存（price_factors 材化済）の冗長性チェック対
REDUNDANCY = [("parkinson_vol", "rvol_60"), ("corwin_schultz_spread", "amihud_illiq"),
              ("dimson_beta", "beta"), ("high_52w", "mom_6m")]
MIN_NAMES = 20


def _cs_corr(x: pd.Series, y: pd.Series) -> float:
    c = x.dropna().index.intersection(y.dropna().index)
    return x[c].rank().corr(y[c].rank()) if len(c) >= MIN_NAMES else np.nan


def _ic_block(title: str, names: list[str], fwd: pd.DataFrame, mask) -> None:
    print(f"\n=== {title}：月次IC（Spearman, 翌月リターン・PITユニバース）===")
    print(f"{'factor':<22}{'meanIC':>8}{'IR':>7}{'n月':>6}{'平均被覆':>9}")
    for name in names:
        f = load_feature(name)
        if f.empty:
            print(f"{name:<22}{'(未材化)':>15}")
            continue
        fm = apply_universe_mask(f, mask)
        ics, cov = [], []
        for t in fm.index:
            if t not in fwd.index:
                continue
            cov.append(int(fm.loc[t].notna().sum()))
            ic = _cs_corr(fm.loc[t], fwd.loc[t])
            if pd.notna(ic):
                ics.append(ic)
        if ics:
            m = float(np.mean(ics))
            ir = m / np.std(ics, ddof=1) if len(ics) > 1 and np.std(ics, ddof=1) > 0 else np.nan
            print(f"{name:<22}{m:>8.3f}{ir:>7.2f}{len(ics):>6}{int(np.mean(cov)):>9}")
        else:
            print(f"{name:<22}{'被覆不足':>15}")


def main() -> None:
    adj_me = load_wide("adj_close").resample("ME").last()
    if adj_me.empty:
        print("Silver adj_close が空。先に store.materialize_all / feature_store を実行。")
        return
    fwd = adj_me.pct_change().shift(-1)
    va_me = load_wide("turnover").resample("ME").last()
    mask = point_in_time_universe(va_me, top_n=500, lookback=12, min_obs=6)
    print(f"PIT ユニバース：{len(mask.index)} 期 / 上位500 / "
          f"{adj_me.index.min():%Y-%m}〜{adj_me.index.max():%Y-%m}")
    _ic_block("A-1 信用/空売り", A1, fwd, mask)
    _ic_block("A-2 微細構造", A2, fwd, mask)
    _ic_block("B fins新ファンダ", B, fwd, mask)

    print("\n=== 新旧ペアワイズ相関（冗長性・PITユニバース内 時間平均 CS Spearman）===")
    for new, old in REDUNDANCY:
        a, b = load_feature(new), load_feature(old)
        if a.empty or b.empty:
            print(f"  {new} vs {old}: 一方未材化")
            continue
        a, b = apply_universe_mask(a, mask), apply_universe_mask(b, mask)
        idx = a.index.intersection(b.index)
        cors = [c for c in (_cs_corr(a.loc[t], b.loc[t]) for t in idx) if pd.notna(c)]
        print(f"  {new:<22} vs {old:<14}: 平均CS相関 {np.mean(cors):+.2f}（n={len(cors)}）"
              if cors else f"  {new} vs {old}: 被覆不足")
    print("\n※ K 不変・throwaway 診断（戦略認定ではない）。符号は『大きいほどロング側』。")


if __name__ == "__main__":
    main()
