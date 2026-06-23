"""赤旗の全社ベースレート（誤発火率）算出（使い捨て診断・K不変・オフライン）。

不正検知の「後知恵バイアス」対策：各赤旗は不正企業(陽性)で出るだけでなく、
**正常企業でどれだけ誤発火するか**を全上場で測らないと意味がない。本スクリプトは
data/ のローカルキャッシュのみで、ルールベース赤旗の base rate（=偽陽性の負荷）と
分布・閾値を出し、後で不正事例DBと突合できる per-filer / per-company パネルを保存する。

赤旗（今すぐ計算可）:
  EDINET 書類パターン（data/edinet/list）: 訂正報告書件数・臨時報告書バースト・有報有無
  財務（data/jquants/fins_summary, FY）: アクルーアル(NP-CFO)/TA・CFO≪NI・売上YoY
出力: 標準出力の base rate サマリ ＋ data/forensic/redflags_{edinet,fins}.parquet
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "forensic"
CORRECTION_DT = {"130", "150", "170", "190"}   # 訂正(有報/四半期/半期/臨報)
EXTRAORD_DT = "180"                              # 臨時報告書


def _safe(p, cols=None):
    try:
        d = pd.read_parquet(p, columns=cols)
        return d if "_empty" not in d.columns else None
    except Exception:                            # noqa: BLE001
        return None


def edinet_filing_flags() -> pd.DataFrame:
    """全 EDINET filer の書類パターン赤旗（訂正件数・臨報バースト・有報有無）。"""
    frames = []
    for p in sorted(glob.glob(str(DATA / "edinet/list/*.parquet"))):
        d = _safe(p, ["edinetCode", "secCode", "filerName", "docTypeCode",
                      "submitDateTime"])
        if d is not None:
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["docTypeCode"] = df["docTypeCode"].astype(str)
    df["submit"] = pd.to_datetime(df["submitDateTime"], errors="coerce")
    df = df.dropna(subset=["edinetCode", "submit"])

    rows = []
    for ec, g in df.groupby("edinetCode"):
        g = g.sort_values("submit")
        ext = g.loc[g["docTypeCode"] == EXTRAORD_DT, "submit"].to_numpy()
        burst = 0                                # 90日窓内の臨報最大数（two-pointer）
        if len(ext):
            j = 0
            for i in range(len(ext)):
                while ext[i] - ext[j] > np.timedelta64(90, "D"):
                    j += 1
                burst = max(burst, i - j + 1)
        sec = g["secCode"].dropna().astype(str)
        rows.append({
            "edinetCode": ec,
            "secCode": sec.iloc[-1] if len(sec) else None,
            "filerName": g["filerName"].dropna().iloc[-1] if g["filerName"].notna().any() else None,
            "n_filings": len(g),
            "n_correction": int(g["docTypeCode"].isin(CORRECTION_DT).sum()),
            "n_extraordinary": int((g["docTypeCode"] == EXTRAORD_DT).sum()),
            "extraordinary_burst90d": int(burst),
            "has_securities_report": bool((g["docTypeCode"] == "120").any()),
            "first": g["submit"].min(), "last": g["submit"].max(),
        })
    return pd.DataFrame(rows)


def fins_financial_flags() -> pd.DataFrame:
    """全社の財務赤旗（FY）：アクルーアル・CFO/NI・売上YoY（PIT: DiscDate）。"""
    frames = []
    for p in sorted(glob.glob(str(DATA / "jquants/fins_summary/*.parquet"))):
        d = _safe(p)
        if d is None or "CurPerType" not in d.columns:
            continue
        d = d[d["CurPerType"] == "FY"]
        keep = [c for c in ["Code", "DiscDate", "CurPerEn", "Sales", "NP", "CFO", "TA"]
                if c in d.columns]
        if {"Code", "NP", "CFO", "TA"} <= set(keep):
            frames.append(d[keep])
    f = pd.concat(frames, ignore_index=True)
    for c in ["Sales", "NP", "CFO", "TA"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f["DiscDate"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f = f.dropna(subset=["Code", "CurPerEn"]).drop_duplicates(["Code", "CurPerEn"])
    f = f.sort_values(["Code", "CurPerEn"])
    f["accruals"] = (f["NP"] - f["CFO"]) / f["TA"].replace(0, np.nan)
    f["cfo_to_ni"] = f["CFO"] / f["NP"].where(f["NP"] != 0, np.nan)
    f["sales_yoy"] = f.groupby("Code")["Sales"].pct_change()
    return f


def _pct(x, q):
    return float(np.nanpercentile(x, q)) if np.isfinite(x).any() else float("nan")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("赤旗の全社ベースレート（誤発火率）＝後知恵バイアスの定量化")
    print("=" * 78)

    # --- EDINET 書類パターン ---
    ed = edinet_filing_flags()
    ed.to_parquet(OUT / "redflags_edinet.parquet")
    n = len(ed)
    print(f"\n[EDINET] filer 数 = {n}")
    for thr in (1, 2, 3):
        r = (ed["n_correction"] >= thr).mean()
        print(f"  訂正報告書 ≥{thr} 件: {r:6.2%}（{int(r*n)}社）")
    for thr in (3, 4, 5):
        r = (ed["extraordinary_burst90d"] >= thr).mean()
        print(f"  臨時報告書 90日内 ≥{thr} 件(バースト): {r:6.2%}（{int(r*n)}社）")
    combo = ((ed["n_correction"] >= 2) | (ed["extraordinary_burst90d"] >= 4))
    print(f"  複合(訂正≥2 または バースト≥4): {combo.mean():6.2%}（{int(combo.sum())}社）"
          "  ← この比率が“誤警報の負荷”")

    # --- 財務 ---
    fn = fins_financial_flags()
    fn.to_parquet(OUT / "redflags_fins.parquet")
    a = fn["accruals"].to_numpy()
    print(f"\n[財務] FY開示 = {len(fn)} 件 / {fn['Code'].nunique()} 社")
    print(f"  アクルーアル(NP-CFO)/TA 分布: "
          f"p50={_pct(a,50):+.3f} p90={_pct(a,90):+.3f} p95={_pct(a,95):+.3f} "
          f"p99={_pct(a,99):+.3f}")
    hi = fn["accruals"] > _pct(a, 90)
    print(f"  高アクルーアル(>p90)該当: {hi.mean():6.2%}（定義上~10%。閾値={_pct(a,90):+.3f}）")
    cn = fn["cfo_to_ni"]
    neg = (cn < 0) & fn["NP"].gt(0)               # 黒字なのに営業CFマイナス＝危険
    print(f"  黒字かつ営業CFマイナス(CFO<0 & NP>0): {neg.mean():6.2%}"
          f"（{int(neg.sum())}件）← 架空売上の典型サイン")

    print("\n保存:", OUT / "redflags_edinet.parquet", "/", OUT / "redflags_fins.parquet")
    print("※ 後工程：不正事例DB(workflow出力)の各社がこれらの分布のどこに位置するかを突合し、"
          "\n  『不正企業は本当に上位に偏るのか／正常企業の誤発火はどれだけか』を評価する。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
