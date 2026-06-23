"""全社の売掛金/DSO パネルを EDINET 有報XBRL から抽出（使い捨て・K不変・オフライン）。

docs/29 §7 の最有力先行指標「売掛金急増・DSO長期化」を全上場で材化する。
fundamentals_long（有報docIDと net_sales を保持）の各 docID について、対応する
data/edinet/docs/{docID}_5.zip の XBRL CSV（UTF-16 tab）を生バイト行スキャンし、
受取債権（売掛金）要素を抽出。net_sales と結合して DSO と YoY 動学を計算する。

出力: data/forensic/dso_panel.parquet
  Code, period_end, DiscDate, docID, net_sales, receivables, dso,
  dso_prev, dso_change, recv_growth, sales_growth, recv_minus_sales_growth
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "edinet" / "docs"
OUT = ROOT / "data" / "forensic"

# 受取債権（売掛金/売上債権）要素を優先順に（広い「受取手形及び売掛金」を優先）
RECV_ELEMS = [
    "jppfs_cor:NotesAndAccountsReceivableTrade",
    "jppfs_cor:NotesAndAccountsReceivableTradeAndContractAssets",
    "jppfs_cor:AccountsReceivableTrade",
    "jppfs_cor:AccountsReceivableTradeAndContractAssets",
    "jpigp_cor:TradeAndOtherReceivablesCA",
    "jpigp_cor:TradeAndOtherCurrentReceivablesCA",
    "jpigp_cor:TradeReceivablesCA",
]
CTX_PREF = ["CurrentYearInstant", "CurrentYearInstant_NonConsolidatedMember"]


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except Exception:                            # noqa: BLE001
        return None


def extract_recv(docid: str):
    zp = DOCS / f"{docid}_5.zip"
    if not zp.exists():
        return None
    try:
        z = zipfile.ZipFile(zp)
    except Exception:                            # noqa: BLE001
        return None
    names = [n for n in z.namelist() if n.endswith(".csv") and "aud" not in n.lower()]
    names.sort(key=lambda n: 0 if "asr" in n.lower() else 1)
    found = {}
    for nm in names:
        try:
            txt = z.read(nm).decode("utf-16")
        except Exception:                        # noqa: BLE001
            try:
                txt = z.read(nm).decode("utf-16-le")
            except Exception:                    # noqa: BLE001
                continue
        for line in txt.split("\n"):
            p = [x.strip().strip('"') for x in line.split("\t")]   # フィールドは"で囲まれる
            if len(p) < 9:
                continue
            key = (p[0], p[2])
            if p[0] in RECV_ELEMS and key not in found:
                found[key] = p[8]
        if found:
            break
    for elem in RECV_ELEMS:                       # 要素→コンテキストの優先で採用
        for ctx in CTX_PREF:
            v = _num(found.get((elem, ctx)))
            if v is not None:
                return v
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fl = pd.read_parquet(ROOT / "data" / "edinet" / "fundamentals_long.parquet")
    fl = fl.dropna(subset=["docID", "Code", "period_end"]).copy()
    fl["net_sales"] = pd.to_numeric(fl["net_sales"], errors="coerce")
    n = len(fl)
    print(f"対象有報: {n} 件 / {fl['Code'].nunique()} 社", flush=True)

    recv = np.full(n, np.nan)
    for i, docid in enumerate(fl["docID"].to_numpy()):
        v = extract_recv(str(docid))
        if v is not None:
            recv[i] = v
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n} 抽出済（受取債権ヒット {int(np.isfinite(recv[:i+1]).sum())}）",
                  flush=True)
    fl["receivables"] = recv
    print(f"受取債権ヒット計: {int(np.isfinite(recv).sum())}/{n}", flush=True)

    df = fl[["Code", "period_end", "DiscDate", "docID", "basis",
             "net_sales", "receivables"]].copy()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.dropna(subset=["period_end"]).sort_values(["Code", "period_end"])
    df = df.drop_duplicates(["Code", "period_end"], keep="last")
    df["dso"] = df["receivables"] / df["net_sales"].replace(0, np.nan) * 365.0
    g = df.groupby("Code")
    df["dso_prev"] = g["dso"].shift(1)
    df["dso_change"] = df["dso"] - df["dso_prev"]
    df["recv_growth"] = g["receivables"].pct_change()
    df["sales_growth"] = g["net_sales"].pct_change()
    df["recv_minus_sales_growth"] = df["recv_growth"] - df["sales_growth"]

    df.to_parquet(OUT / "dso_panel.parquet")
    ok = df["dso"].notna()
    print(f"\nDSO算出可: {int(ok.sum())} 行 / {df['Code'].nunique()} 社", flush=True)
    d = df.loc[ok, "dso"]
    print(f"DSO 分布(日): p10={d.quantile(.1):.0f} p50={d.quantile(.5):.0f} "
          f"p90={d.quantile(.9):.0f} p99={d.quantile(.99):.0f}", flush=True)
    dc = df["dso_change"].dropna()
    print(f"DSO変化(日) 分布: p50={dc.quantile(.5):+.0f} p90={dc.quantile(.9):+.0f} "
          f"p95={dc.quantile(.95):+.0f}", flush=True)
    rs = df["recv_minus_sales_growth"].dropna()
    print(f"売掛金成長−売上成長 分布: p50={rs.quantile(.5):+.1%} p90={rs.quantile(.9):+.1%} "
          f"p95={rs.quantile(.95):+.1%}", flush=True)
    print("保存:", OUT / "dso_panel.parquet", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
