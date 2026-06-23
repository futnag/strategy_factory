"""単一有報内(within-filing)の前年列でΔDSOを計算し、IPO直後の被覆欠落を埋める。

forensic_dso_panel はクロス年度（複数有報）でΔDSOを取るため、IPO直後で1有報しか無い企業
（オルツ等＝収益偽装に最も狙われる）は前年が無くNaN。だが有報には前年列(Prior1Year)が載るので、
単一有報内で当年/前年の売掛金・売上を取れば初回提出でもYoYが計算できる。不正事例について
within-filing ΔDSO を求め、全社（クロス年度）分布と突合して recall を再評価する。
使い捨て・K不変・オフライン。
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "edinet" / "docs"
OUT = ROOT / "data" / "forensic"
RECV = ["jppfs_cor:NotesAndAccountsReceivableTrade",
        "jppfs_cor:NotesAndAccountsReceivableTradeAndContractAssets",
        "jppfs_cor:AccountsReceivableTrade",
        "jppfs_cor:AccountsReceivableTradeAndContractAssets"]
SALES = ["jppfs_cor:NetSales", "jppfs_cor:OperatingRevenue1",
         "jpigp_cor:RevenueIFRS"]


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except Exception:                            # noqa: BLE001
        return None


def _table(docid):
    zp = DOCS / f"{docid}_5.zip"
    if not zp.exists():
        return {}
    try:
        z = zipfile.ZipFile(zp)
    except Exception:                            # noqa: BLE001
        return {}
    out = {}
    for nm in z.namelist():
        if not (nm.endswith(".csv") and "asr" in nm.lower()):
            continue
        try:
            txt = z.read(nm).decode("utf-16")
        except Exception:                        # noqa: BLE001
            continue
        for line in txt.split("\n"):
            p = [x.strip().strip('"') for x in line.split("\t")]
            if len(p) >= 9 and (p[0] in RECV or p[0] in SALES):
                out.setdefault((p[0], p[2]), p[8])
    return out


def _pick(tbl, elems, ctxs):
    for e in elems:
        for c in ctxs:
            v = _num(tbl.get((e, c)))
            if v is not None:
                return v
    return None


def within_dso(docid):
    """単一有報から (DSO_cy, DSO_py, ΔDSO, recv_growth, sales_growth) を返す。"""
    t = _table(docid)
    if not t:
        return None
    # 連結優先で当年/前年の受取債権・売上を一貫ペアで取る
    for suffix in ("", "_NonConsolidatedMember"):
        rc = _pick(t, RECV, [f"CurrentYearInstant{suffix}"])
        rp = _pick(t, RECV, [f"Prior1YearInstant{suffix}"])
        sc = _pick(t, SALES, [f"CurrentYearDuration{suffix}"])
        sp = _pick(t, SALES, [f"Prior1YearDuration{suffix}"])
        if all(v and v > 0 for v in (rc, rp, sc, sp)):
            dso_cy, dso_py = rc / sc * 365, rp / sp * 365
            return {"dso_cy": dso_cy, "dso_py": dso_py, "dso_change": dso_cy - dso_py,
                    "recv_growth": rc / rp - 1, "sales_growth": sc / sp - 1,
                    "recv_minus_sales_growth": (rc / rp - 1) - (sc / sp - 1),
                    "consol": suffix == ""}
    return None


def _lead_date(s):
    m = re.search(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", str(s))
    if not m:
        return pd.NaT
    try:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))
    except Exception:                            # noqa: BLE001
        return pd.NaT


def _cands(sec):
    s = str(sec).strip().upper()
    return {c for c in {s, s + "0" if len(s) == 4 else s} if c and c != "UNKNOWN"}


def main() -> int:
    fl = pd.read_parquet(ROOT / "data" / "edinet" / "fundamentals_long.parquet")
    fl["period_end"] = pd.to_datetime(fl["period_end"], errors="coerce")
    fl["Code"] = fl["Code"].astype(str)
    dso = pd.read_parquet(OUT / "dso_panel.parquet")
    dist = dso["dso_change"].dropna().to_numpy()         # 全社（クロス年度）分布
    distr = dso["recv_minus_sales_growth"].dropna().to_numpy()

    def pct(a, v):
        return float((a < v).mean() * 100) if (np.isfinite(v) and len(a)) else np.nan

    cases = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    rows, hit = [], 0
    print("=" * 78)
    print("within-filing ΔDSO（単一有報内の前年列）＝IPO直後の被覆を補完")
    print("=" * 78)
    print(f"{'会社':<20}{'コード':<7}{'DSO_cy':>7}{'ΔDSO':>6}{'%ile':>6}{'売掛-売上':>9}{'%ile':>6}")
    for c in cases:
        rev = _lead_date(c.get("revelation_date"))
        if pd.isna(rev):
            continue
        sub = fl[fl["Code"].isin(_cands(c.get("sec_code", ""))) & (fl["period_end"] < rev)]
        if not len(sub):
            continue
        docid = sub.sort_values("period_end").iloc[-1]["docID"]
        w = within_dso(str(docid))
        if not w:
            continue
        hit += 1
        p_d, p_r = pct(dist, w["dso_change"]), pct(distr, w["recv_minus_sales_growth"])
        rows.append({"company": c.get("company"), "sec": c.get("sec_code"),
                     "dso_change": w["dso_change"], "p_dso_change": p_d,
                     "recv_minus_sales_growth": w["recv_minus_sales_growth"], "p_rms": p_r})
        print(f"{str(c['company'])[:19]:<20}{str(c.get('sec_code','?'))[:6]:<7}"
              f"{w['dso_cy']:>7.0f}{w['dso_change']:>+6.0f}{p_d:>6.0f}"
              f"{w['recv_minus_sales_growth']:>+9.0%}{p_r:>6.0f}")
    res = pd.DataFrame(rows)
    res.to_parquet(OUT / "dso_within_case.parquet")
    print(f"\nwithin-filing で ΔDSO 取得 = {hit} 件（クロス年度版46件→被覆拡大）")
    for col, lab in [("p_dso_change", "ΔDSO>p90"), ("p_rms", "売掛-売上>p90")]:
        v = res[col].dropna()
        rec = (v >= 90).mean()
        print(f"  {lab:<14} recall={rec:5.0%} ({int((v>=90).sum())}/{len(v)})  "
              f"中央%ile={v.median():.0f}  lift≈{rec/0.10:.1f}x")
    o = res[res["sec"].astype(str).str.startswith("260A")]
    if len(o):
        print(f"\nオルツ（補完後）: ΔDSO={o.iloc[0]['dso_change']:+.0f}日 "
              f"p{o.iloc[0]['p_dso_change']:.0f} / 売掛-売上 p{o.iloc[0]['p_rms']:.0f}"
              f" ← クロス年度版では取得不可だった")
    print("保存:", OUT / "dso_within_case.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
