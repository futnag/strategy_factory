"""不正事例DB × 全社赤旗の突合・評価（使い捨て診断・K不変・オフライン）。

forensic_redflags_baserate.py の全社パネルと、不正事例DB（workflowまたは手元JSON）を
突合し、「不正企業は赤旗分布の上位に偏るのか／発覚前(PIT)に立つのか」を評価する。
後知恵バイアスを避けるため、財務赤旗は DiscDate < 発覚日 のみ、EDINET 書類赤旗は
提出日 < 発覚日 のみで再集計する（＝発覚後の訂正・臨報スパイクを使わない）。

使い方: python examples/forensic_case_join.py --cases data/forensic/cases.json
        （省略時 data/forensic/test_cases.json）
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "forensic"
CORRECTION_DT = {"130", "150", "170", "190"}
EXTRAORD_DT = "180"


def _safe(p, cols=None):
    try:
        d = pd.read_parquet(p, columns=cols)
        return d if "_empty" not in d.columns else None
    except Exception:                            # noqa: BLE001
        return None


def _code_candidates(sec):
    """研究の証券コード（4-5桁/英数）→ J-Quants/EDINET 5桁の候補集合。"""
    s = str(sec).strip().upper()
    out = {s}
    if len(s) == 4:
        out.add(s + "0")                          # 7203→72030, 260A→260A0
    if len(s) == 5 and s.endswith("0"):
        out.add(s[:-1])
    return {c for c in out if c and c != "UNKNOWN"}


def _load_edinet_list():
    frames = []
    for p in sorted(glob.glob(str(DATA / "edinet/list/*.parquet"))):
        d = _safe(p, ["edinetCode", "secCode", "filerName", "docTypeCode",
                      "submitDateTime"])
        if d is not None:
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["docTypeCode"] = df["docTypeCode"].astype(str)
    df["submit"] = pd.to_datetime(df["submitDateTime"], errors="coerce")
    df["secCode"] = df["secCode"].astype(str)
    return df.dropna(subset=["submit"])


def _burst90(dates):
    d = np.sort(dates)
    if not len(d):
        return 0
    j, b = 0, 0
    for i in range(len(d)):
        while d[i] - d[j] > np.timedelta64(90, "D"):
            j += 1
        b = max(b, i - j + 1)
    return int(b)


def _lead_date(s):
    """冗長文字列から先頭の年月(日)を抽出（検証エージェントが日付欄に散文を書くため）。"""
    import re
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(OUT / "test_cases.json"))
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    edlist = _load_edinet_list()
    fins = _safe(OUT / "redflags_fins.parquet")
    fins["DiscDate"] = pd.to_datetime(fins["DiscDate"], errors="coerce")
    acc_all = fins["accruals"].dropna().to_numpy()   # 全社分布（NaN除外でpercentile）

    def acc_pct(v):
        return float((acc_all < v).mean() * 100) if np.isfinite(v) else float("nan")

    print("=" * 92)
    print("不正事例 × 全社赤旗 突合（PIT: 発覚前のみ＝財務DiscDate<発覚, EDINET提出<発覚）")
    print("=" * 92)
    print(f"{'会社':<20}{'コード':<7}{'発覚':<9}{'訂正':>4}{'burst':>6}{'有報':>4}"
          f"{'acc%':>6}{'CFO<0':>6}{'flag':>5}")
    rows = []
    for c in cases:
        rev = _lead_date(c.get("revelation_date"))
        rev_known = pd.notna(rev)
        cutoff = rev if rev_known else pd.Timestamp("2100-01-01")
        cands = _code_candidates(c.get("sec_code", ""))

        # --- EDINET（発覚前のみ）---
        sub = edlist[edlist["secCode"].isin(cands)]
        if not len(sub) and c.get("company"):
            sub = edlist[edlist["filerName"].astype(str).str.contains(
                str(c["company"]), na=False, regex=False)]
        pre = sub[sub["submit"] < cutoff]
        n_corr = int(pre["docTypeCode"].isin(CORRECTION_DT).sum())
        burst = _burst90(pre.loc[pre["docTypeCode"] == EXTRAORD_DT, "submit"].to_numpy())
        has120 = bool((pre["docTypeCode"] == "120").any())

        # --- 財務（発覚前の最新FY）---
        fc = fins[fins["Code"].astype(str).isin(cands) & (fins["DiscDate"] < cutoff)]
        acc = np.nan
        cfo_neg = None
        if len(fc):
            last = fc.sort_values("DiscDate").iloc[-1]
            acc = last["accruals"]
            if {"CFO", "NP"} <= set(fc.columns) and pd.notna(last["CFO"]) and pd.notna(last["NP"]):
                cfo_neg = bool((last["CFO"] < 0) and (last["NP"] > 0))
        ap_ = acc_pct(acc)

        # 個別フラグ（発覚前PIT）
        f_corr = n_corr >= 2
        f_burst = burst >= 4
        f_acc = np.isfinite(ap_) and ap_ >= 90
        f_cfo = cfo_neg is True
        flagged = f_corr or f_burst or f_acc or f_cfo
        evaluable = rev_known and (bool(len(sub)) or bool(len(fc)))
        print(f"{str(c['company'])[:19]:<20}{str(c.get('sec_code','?'))[:6]:<7}"
              f"{(rev.strftime('%Y-%m') if rev_known else '不明'):<9}{n_corr:>4}{burst:>6}"
              f"{'✓' if has120 else '-':>4}{(f'{ap_:.0f}' if np.isfinite(ap_) else '—'):>6}"
              f"{('✓' if f_cfo else '-'):>6}{'★' if flagged else '·':>5}")
        rows.append({"company": c.get("company"), "sec_code": c.get("sec_code"),
                     "confidence": c.get("confidence"), "rev": rev, "rev_known": rev_known,
                     "n_correction_pre": n_corr, "burst_pre": burst, "has_120_pre": has120,
                     "accruals_pct": ap_, "cfo_neg_ni_pos": f_cfo,
                     "f_corr": f_corr, "f_burst": f_burst, "f_acc": f_acc, "f_cfo": f_cfo,
                     "flagged": flagged, "evaluable": evaluable,
                     "edinet_found": bool(len(sub)), "fins_found": bool(len(fc))})
    res = pd.DataFrame(rows)
    res.to_parquet(OUT / "case_redflag_join.parquet")

    ev = res[res["evaluable"]]
    print("\n" + "=" * 92)
    print(f"評価可能(発覚日既知 & 発覚前データあり) = {len(ev)}/{len(res)} 件"
          f"（EDINET収録 {int(res['edinet_found'].sum())} / J-Q財務収録 {int(res['fins_found'].sum())}）")
    print(f"\n【PIT recall（発覚前に立った割合）】 vs 全社ベースレート（＝誤発火/lift）")
    def line(name, mask_col, base):
        sub = ev[ev[mask_col]]
        r = len(sub) / len(ev) if len(ev) else 0
        print(f"  {name:<24} recall={r:5.0%} ({len(sub):>2}/{len(ev)})   "
              f"全社base={base:5.1%}   lift≈{(r / base if base > 0 else 0):.1f}x")
    line("高アクルーアル(>p90)", "f_acc", 0.10)
    line("黒字&営業CFマイナス", "f_cfo", 0.044)
    line("訂正≥2(発覚前)", "f_corr", 0.188)
    line("臨報burst≥4(発覚前)", "f_burst", 0.038)
    line("複合(いずれか)", "flagged", 0.20)
    print("\n注意: 複合recallは見かけ高いが全社base≈20%＝誤発火が多くliftは低い。財務系(アクルーアル/"
          "CFO)は発覚前の“先行”指標、訂正/臨報は発覚前は弱く主に同時/遅行（バグ修正後の正直な姿）。")
    print("保存:", OUT / "case_redflag_join.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
