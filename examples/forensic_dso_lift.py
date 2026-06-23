"""DSO/売掛金動学 × 不正事例の lift 検証（使い捨て・K不変・オフライン）。

forensic_dso_panel.py の全社DSOパネルと不正事例DBを突合し、発覚前(PIT)の
  ・DSO変化(日)         dso_change
  ・売掛金成長−売上成長   recv_minus_sales_growth
  ・DSO水準             dso
が、不正企業で全社分布の上位に偏るか（recall@p90 と lift）を評価。docs/29 の財務サマリ
フラグ（高アクルーアル lift0.8x 等）を上回るかを判定する。
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


# 収益偽装型（売掛金/DSOが効く型）＝架空売上・循環取引・売上前倒し・売上過大
_REVFAB = re.compile(r"架空売上|循環|売上前倒|売上.*過大|架空.*売上|過大.*売上|"
                     r"fictitious|circular|revenue", re.IGNORECASE)


def is_revfab(ft) -> bool:
    return bool(_REVFAB.search(str(ft)))


def main() -> int:
    panel = pd.read_parquet(OUT / "dso_panel.parquet")
    panel["Code"] = panel["Code"].astype(str)
    panel["period_end"] = pd.to_datetime(panel["period_end"], errors="coerce")
    cases = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))

    # 全社分布（percentile rank 用）
    dist = {
        "dso_change": panel["dso_change"].dropna().to_numpy(),
        "recv_minus_sales_growth": panel["recv_minus_sales_growth"].dropna().to_numpy(),
        "dso": panel["dso"].dropna().to_numpy(),
    }

    def pct(col, v):
        a = dist[col]
        return float((a < v).mean() * 100) if (np.isfinite(v) and len(a)) else np.nan

    rows = []
    print("=" * 88)
    print("DSO/売掛金動学 × 不正事例（発覚前の最新有報・全社percentile）")
    print("=" * 88)
    print(f"{'会社':<20}{'コード':<7}{'発覚':<9}{'DSO':>5}{'ΔDSO':>6}{'Δ%ile':>7}"
          f"{'売掛-売上':>9}{'%ile':>6}")
    for c in cases:
        rev = _lead_date(c.get("revelation_date"))
        if pd.isna(rev):
            continue
        cands = _cands(c.get("sec_code", ""))
        sub = panel[panel["Code"].isin(cands) & (panel["period_end"] < rev)]
        if not len(sub):
            continue
        last = sub.sort_values("period_end").iloc[-1]
        dso_v, dch, rms = last["dso"], last["dso_change"], last["recv_minus_sales_growth"]
        p_dch, p_rms = pct("dso_change", dch), pct("recv_minus_sales_growth", rms)
        rows.append({"company": c.get("company"), "confidence": c.get("confidence"),
                     "fraud_type": c.get("fraud_type"), "revfab": is_revfab(c.get("fraud_type")),
                     "dso": dso_v, "dso_change": dch, "p_dso_change": p_dch,
                     "recv_minus_sales_growth": rms, "p_rms": p_rms,
                     "p_dso": pct("dso", dso_v)})
        print(f"{str(c['company'])[:19]:<20}{str(c.get('sec_code','?'))[:6]:<7}"
              f"{rev.strftime('%Y-%m'):<9}{(f'{dso_v:.0f}' if np.isfinite(dso_v) else '—'):>5}"
              f"{(f'{dch:+.0f}' if np.isfinite(dch) else '—'):>6}"
              f"{(f'{p_dch:.0f}' if np.isfinite(p_dch) else '—'):>7}"
              f"{(f'{rms:+.0%}' if np.isfinite(rms) else '—'):>9}"
              f"{(f'{p_rms:.0f}' if np.isfinite(p_rms) else '—'):>6}")

    res = pd.DataFrame(rows)
    res.to_parquet(OUT / "dso_case_lift.parquet")
    print("\n" + "=" * 88)
    print(f"DSO観測のある事例 = {len(res)} 件")
    print("\n【DSO系フラグの recall@p90 vs 全社base(10%) = lift】")
    for col, lab in [("p_dso_change", "ΔDSO(YoY)>p90"),
                     ("p_rms", "売掛金成長−売上成長>p90"),
                     ("p_dso", "DSO水準>p90")]:
        v = res[col].dropna()
        if not len(v):
            continue
        rec = (v >= 90).mean()
        med = v.median()
        print(f"  {lab:<24} recall={rec:5.0%} ({int((v>=90).sum())}/{len(v)})  "
              f"中央percentile={med:.0f}  lift≈{rec/0.10:.1f}x")
    print("\n比較: 財務サマリ高アクルーアル lift0.8x / 黒字&CFマイナス lift0.7x（docs/29）。")

    # --- Step1: 型別（収益偽装型 vs それ以外）---
    print("\n" + "=" * 88)
    print("【型別】収益偽装型(架空売上/循環/売上前倒し) vs それ以外  ※ΔDSO観測あり対象")
    print("=" * 88)
    ev = res.dropna(subset=["p_dso_change"])
    rf, nf = ev[ev["revfab"]], ev[~ev["revfab"]]
    for lab, g in [("収益偽装型", rf), ("それ以外", nf)]:
        if not len(g):
            continue
        for col, cl in [("p_dso_change", "ΔDSO>p90"), ("p_rms", "売掛-売上>p90")]:
            v = g[col].dropna()
            rec = (v >= 90).mean() if len(v) else float("nan")
            print(f"  [{lab:<6}] {cl:<14} recall={rec:5.0%} ({int((v>=90).sum())}/{len(v)})"
                  f"  中央%ile={v.median():.0f}  lift≈{rec/0.10:.1f}x")
    # 収益偽装型で「ΔDSO>p90 または 売掛-売上>p90」
    if len(rf):
        any_hi = ((rf["p_dso_change"] >= 90) | (rf["p_rms"] >= 90))
        print(f"\n  収益偽装型の『ΔDSO>p90 または 売掛-売上>p90』recall="
              f"{any_hi.mean():.0%} ({int(any_hi.sum())}/{len(rf)})  lift≈{any_hi.mean()/0.19:.1f}x"
              f"（複合base≈19%）")
    print("\n保存:", OUT / "dso_case_lift.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
