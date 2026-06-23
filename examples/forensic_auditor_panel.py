"""全社の監査法人パネルと変更検知（使い捨て・K不変・オフライン）。

各有報XBRLの `jpcrp_cor:AuditFirm1Consolidated`（無ければ NonConsolidated）から会計監査人名を
抽出し、(Code, period_end) パネル化。前年からの変更・**大手→中小の交代**を検知する
（ユーザー指摘の有力赤旗）。出力: data/forensic/auditor_panel.parquet
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "edinet" / "docs"
OUT = ROOT / "data" / "forensic"
AUD_ELEMS = ["jpcrp_cor:AuditFirm1Consolidated", "jpcrp_cor:AuditFirm1NonConsolidated"]
# 大手（Big-N）＋準大手の主要法人
BIG_N = ["新日本", "あずさ", "トーマツ", "あらた", "PwC", "ＰｗＣ", "プライスウォーターハウス"]


def extract_auditor(docid: str):
    zp = DOCS / f"{docid}_5.zip"
    if not zp.exists():
        return None
    try:
        z = zipfile.ZipFile(zp)
    except Exception:                            # noqa: BLE001
        return None
    names = [n for n in z.namelist() if n.endswith(".csv")]
    names.sort(key=lambda n: 0 if "aud" in n.lower() else 1)   # 監査報告CSV優先
    found = {}
    for nm in names:
        try:
            txt = z.read(nm).decode("utf-16")
        except Exception:                        # noqa: BLE001
            continue
        for line in txt.split("\n"):
            p = [x.strip().strip('"') for x in line.split("\t")]
            if len(p) < 9:
                continue
            if p[0] in AUD_ELEMS and p[0] not in found and p[8] and p[8] != "－":
                found[p[0]] = p[8]
        if found:
            break
    for e in AUD_ELEMS:
        if e in found:
            return found[e]
    return None


def is_bign(name) -> bool:
    return any(k in str(name) for k in BIG_N)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fl = pd.read_parquet(ROOT / "data" / "edinet" / "fundamentals_long.parquet")
    fl = fl.dropna(subset=["docID", "Code", "period_end"]).copy()
    n = len(fl)
    print(f"対象有報: {n} 件", flush=True)
    aud = []
    for i, d in enumerate(fl["docID"].to_numpy()):
        aud.append(extract_auditor(str(d)))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n}（監査人ヒット {sum(a is not None for a in aud)}）", flush=True)
    fl["auditor"] = aud
    print(f"監査人ヒット計: {sum(a is not None for a in aud)}/{n}", flush=True)

    df = fl[["Code", "period_end", "DiscDate", "auditor"]].copy()
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.dropna(subset=["period_end", "auditor"]).sort_values(["Code", "period_end"])
    df = df.drop_duplicates(["Code", "period_end"], keep="last")
    df["is_bign"] = df["auditor"].map(is_bign)
    g = df.groupby("Code")
    df["auditor_prev"] = g["auditor"].shift(1)
    df["bign_prev"] = g["is_bign"].shift(1)
    df["auditor_changed"] = (df["auditor_prev"].notna() &
                             (df["auditor"].str.replace(r"\s|　", "", regex=True)
                              != df["auditor_prev"].str.replace(r"\s|　", "", regex=True)))
    df["big_to_small"] = df["auditor_changed"] & df["bign_prev"].fillna(False) & ~df["is_bign"]
    df.to_parquet(OUT / "auditor_panel.parquet")

    firms = df["Code"].nunique()
    chg = df.groupby("Code")["auditor_changed"].any().mean()
    b2s = df.groupby("Code")["big_to_small"].any().mean()
    print(f"\n監査人パネル {len(df)}行 / {firms}社", flush=True)
    print(f"  監査人を一度でも変更した企業: {chg:.1%}", flush=True)
    print(f"  大手→中小の交代を経験: {b2s:.1%}（＝この比率が全社base）", flush=True)
    print("保存:", OUT / "auditor_panel.parquet", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
