"""PU学習（Positive-Unlabeled）で複合フォレンジック・スコアを最適化（使い捨て・K不変）。

手作り複合スコア（DSO動学＋監査人の等加重）を、確定不正を陽性・残り全firm-yearを未ラベル
とするPU学習（Elkan-Noto流＝未ラベルを負例として class_weight 補正）で学習重みに置き換え、
手作りルール（score≥3 で lift4.8x）を上回るかを検証する。極小陽性ゆえ leave-one-fraud-out CV で
正直に評価し、学習がルールを有意に超えるか／データ希少で同等かを判定する。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forensic_composite import build_firmyear, _cands, _lead_date  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "forensic"
FEATS = ["dso_change", "recv_minus_sales_growth", "f_small_num", "f_b2s_num"]


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    fy, _ = build_firmyear()
    fy["f_small_num"] = fy["f_small"].astype(float)
    fy["f_b2s_num"] = fy["f_b2s"].astype(float)
    # inf除去→winsorize(1/99%)→中央値補完（neutral）
    for c in ["dso_change", "recv_minus_sales_growth"]:
        fy[c] = fy[c].replace([np.inf, -np.inf], np.nan)
        lo, hi = fy[c].quantile(0.01), fy[c].quantile(0.99)
        fy[c] = fy[c].clip(lo, hi).fillna(fy[c].median())
    fy = fy.dropna(subset=FEATS).reset_index(drop=True)

    # 各社最新 firm-year（点時スクリーンの評価母集団）
    latest = fy.sort_values("period_end").drop_duplicates("Code", keep="last").copy()

    # 陽性＝不正事例の発覚前最新 firm-year を latest 上で特定
    cases = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    pos_codes = set()
    for c in cases:
        rev = _lead_date(c.get("revelation_date"))
        if pd.isna(rev):
            continue
        sub = fy[fy["Code"].isin(_cands(c.get("sec_code", ""))) & (fy["period_end"] < rev)]
        if len(sub):
            pos_codes.add(sub.sort_values("period_end").iloc[-1]["Code"])
    latest["y"] = latest["Code"].isin(pos_codes).astype(int)
    n_pos = int(latest["y"].sum())
    print("=" * 76)
    print("PU学習 複合スコア最適化（DSO動学＋監査人）")
    print("=" * 76)
    print(f"評価母集団(各社最新) {len(latest)}社 / 陽性(不正) {n_pos}社 "
          f"= base {n_pos/len(latest):.2%}")

    X = latest[FEATS].to_numpy()
    y = latest["y"].to_numpy()
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    # --- 全データfit：学習重み（どの特徴が効くか）---
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xs, y)
    print("\n学習係数（標準化特徴・正=不正方向）:")
    for f, w in sorted(zip(FEATS, clf.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {f:<26} {w:+.3f}")

    # --- leave-one-fraud-out CV：陽性のOOSスコア ---
    from sklearn.base import clone
    oos = np.full(len(latest), np.nan)
    pos_idx = np.where(y == 1)[0]
    for i in pos_idx:                            # 各陽性を抜いて学習→当該をOOS予測
        mask = np.ones(len(latest), bool)
        mask[i] = False
        m = clone(clf).fit(Xs[mask], y[mask])
        oos[i] = m.predict_proba(Xs[i:i + 1])[0, 1]
    # 未ラベルは全データfitのスコアで baseline 分布を作る（閾値用）
    score_all = clf.predict_proba(Xs)[:, 1]
    pos_oos = oos[pos_idx]

    def lift_at(top_frac):
        thr = np.quantile(score_all, 1 - top_frac)
        rec = float((pos_oos >= thr).mean())
        return rec, rec / top_frac
    print("\n【学習スコア lift（leave-one-fraud-out OOS）】")
    for tf in (0.10, 0.05, 0.027):
        rec, lift = lift_at(tf)
        print(f"  上位{tf*100:>4.1f}%で発火: recall={rec:5.0%}  lift≈{lift:.1f}x")
    print("\n比較（手作りルール・docs/29 §7.3）: 複合score≥3 lift4.8x / 単独ΔDSO lift3x")

    latest["pu_score"] = score_all
    latest[["Code", "period_end", "y", "pu_score"] + FEATS].to_parquet(
        OUT / "pu_scores.parquet")
    top = latest.sort_values("pu_score", ascending=False).head(15)
    print(f"\nPUスコア上位15社中の不正(陽性)数: {int(top['y'].sum())}/15")
    print("保存:", OUT / "pu_scores.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
