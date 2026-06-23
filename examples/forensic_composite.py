"""複合フォレンジック・スコア（使い捨て・K不変・オフライン）。

単独では誤発火の多い赤旗を束ね、「複数の独立赤旗が同時点灯」で特異度を上げる。
全社の firm-year を DSO動学（dso_panel）＋監査人（auditor_panel）で結合し、各社の最新報告で
スコア化。不正事例（発覚前PIT）のスコア分布と recall を全社base と比較して lift を測る。

フラグ（PIT・会計年度単位）:
  f_dso  : ΔDSO(YoY) > 全社p90
  f_rms  : 売掛金成長 − 売上成長 > 全社p90
  f_small: 監査法人が中小（非大手。2022+のみ取得可）
  f_b2s  : 監査法人 大手→中小 の交代
score = 合計（0–4）。score≥2 を複合フラグとする。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "forensic"


def _lead_date(s):
    m = re.search(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", str(s))
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
    else:
        m2 = re.search(r"(\d{4})年(\d{1,2})月", str(s))
        if not m2:
            return pd.NaT
        y, mo, d = int(m2.group(1)), int(m2.group(2)), 1
    try:
        return pd.Timestamp(y, mo, d)
    except Exception:                            # noqa: BLE001
        return pd.NaT


def _cands(sec):
    s = str(sec).strip().upper()
    out = {s}
    if len(s) == 4:
        out.add(s + "0")
    if len(s) == 5 and s.endswith("0"):
        out.add(s[:-1])
    return {c for c in out if c and c != "UNKNOWN"}


def build_firmyear():
    dso = pd.read_parquet(OUT / "dso_panel.parquet")
    dso["Code"] = dso["Code"].astype(str)
    dso["period_end"] = pd.to_datetime(dso["period_end"], errors="coerce")
    aud = pd.read_parquet(OUT / "auditor_panel.parquet")
    aud["Code"] = aud["Code"].astype(str)
    aud["period_end"] = pd.to_datetime(aud["period_end"], errors="coerce")
    fy = dso.merge(aud[["Code", "period_end", "is_bign", "big_to_small"]],
                   on=["Code", "period_end"], how="left")
    p90_dso = dso["dso_change"].quantile(0.90)
    p90_rms = dso["recv_minus_sales_growth"].quantile(0.90)
    fy["f_dso"] = fy["dso_change"] > p90_dso
    fy["f_rms"] = fy["recv_minus_sales_growth"] > p90_rms
    fy["f_small"] = (fy["is_bign"] == False)          # noqa: E712  (NaN→False)
    fy["f_b2s"] = fy["big_to_small"].fillna(False)
    fy["score"] = fy[["f_dso", "f_rms", "f_small", "f_b2s"]].sum(axis=1)
    return fy, (p90_dso, p90_rms)


def main() -> int:
    fy, (p90_dso, p90_rms) = build_firmyear()
    # 全社base＝各社の最新 firm-year で1回評価（点時スクリーンを模す）
    latest = fy.sort_values("period_end").drop_duplicates("Code", keep="last")
    base2 = (latest["score"] >= 2).mean()
    print("=" * 80)
    print("複合フォレンジック・スコア（DSO動学＋監査人）")
    print("=" * 80)
    print(f"firm-year {len(fy)} / 企業 {fy['Code'].nunique()}  "
          f"閾値 ΔDSO>p90={p90_dso:+.0f}日, 売掛-売上>p90={p90_rms:+.0%}")
    print(f"全社(最新)スコア分布: " + ", ".join(
        f"{k}={v}" for k, v in latest["score"].value_counts().sort_index().items()))
    print(f"全社 score≥2 base = {base2:.1%}")

    # --- 不正事例（発覚前PIT・最新 firm-year）---
    cases = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    rows = []
    for c in cases:
        rev = _lead_date(c.get("revelation_date"))
        if pd.isna(rev):
            continue
        sub = fy[fy["Code"].isin(_cands(c.get("sec_code", ""))) & (fy["period_end"] < rev)]
        if not len(sub):
            continue
        last = sub.sort_values("period_end").iloc[-1]
        rows.append({"company": c.get("company"), "sec": c.get("sec_code"),
                     "score": int(last["score"]), "f_dso": bool(last["f_dso"]),
                     "f_rms": bool(last["f_rms"]), "f_small": bool(last["f_small"]),
                     "f_b2s": bool(last["f_b2s"])})
    res = pd.DataFrame(rows)
    res.to_parquet(OUT / "composite_case.parquet")

    print(f"\n不正事例(スコア算出可) = {len(res)} 件")
    print("スコア分布:", ", ".join(
        f"{k}={v}" for k, v in res["score"].value_counts().sort_index().items()))
    for thr in (1, 2, 3):
        rec = (res["score"] >= thr).mean()
        base = (latest["score"] >= thr).mean()
        print(f"  score≥{thr}: recall={rec:5.0%} ({int((res['score']>=thr).sum())}/{len(res)})"
              f"  全社base={base:5.1%}  lift≈{rec/base if base>0 else 0:.1f}x")
    print("\n各フラグ単独の不正事例 recall:")
    for f in ["f_dso", "f_rms", "f_small", "f_b2s"]:
        print(f"  {f}: {res[f].mean():.0%}  全社base={latest[f].mean() if f in latest else float('nan'):.1%}")

    # オルツ
    o = res[res["sec"].astype(str).str.startswith("260A")]
    if len(o):
        print("\nオルツ:", o.iloc[0].to_dict())
    print("\n保存:", OUT / "composite_case.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
