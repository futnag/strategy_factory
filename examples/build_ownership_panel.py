r"""有報XBRL（キャッシュ済み EDINET docs）→ 所有者別状況パネルをオフライン抽出（CI互換・MCP非依存）。

設計＝docs/50 §8。手順:
  1. list ミラーから有報(docTypeCode=120)の docID→(Code=secCode, fiscal_year=periodEnd年, submit_dt) を収集。
  2. キャッシュ済み docs/{docID}_5.zip を read_xbrl_csv → parse_ownership_from_xbrl（純関数）。冪等（docID差分）。
  3. オフライン結果を data/edinet/ownership_xbrl.parquet に保存。
  4. 既存 MCP パネル（ownership_categories.parquet）と照合（正しさの裏取り・overlap 一致率）。
  5. オフライン優先＋MCP gapfill で正準 ownership_categories.parquet を生成（overlay がそのまま読む）。

実行: $env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\build_ownership_panel.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.data.sources import edinet as ed  # noqa: E402
from invest_system.equities.edinet_ownership import parse_ownership_from_xbrl  # noqa: E402

DOCS = "data/edinet/docs"
LIST = "data/edinet/list"
XBRL_PANEL = "data/edinet/ownership_xbrl.parquet"
CANON = "data/edinet/ownership_categories.parquet"
ANNUAL = "120"   # docTypeCode 有価証券報告書


def collect_meta(cached_ids: set) -> dict:
    """list ミラー → {docID: (Code, fiscal_year, submit_dt)}（有報・キャッシュ済みのみ）。"""
    meta: dict[str, tuple] = {}
    for p in sorted(glob.glob(f"{LIST}/*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        if "docID" not in df.columns or "docTypeCode" not in df.columns:
            continue
        sub = df[(df["docTypeCode"].astype(str) == ANNUAL) & df["docID"].isin(cached_ids)]
        if "secCode" not in sub.columns:
            continue
        sub = sub.dropna(subset=["secCode"])
        pe_col = "periodEnd" if "periodEnd" in sub.columns else None
        for _, r in sub.iterrows():
            did = str(r["docID"])
            if did in meta:
                continue
            code = str(r["secCode"])
            fy = None
            if pe_col and pd.notna(r[pe_col]):
                fy = pd.to_datetime(r[pe_col], errors="coerce")
                fy = int(fy.year) if pd.notna(fy) else None
            if fy is None and "submitDateTime" in sub.columns and pd.notna(r.get("submitDateTime")):
                d = pd.to_datetime(r["submitDateTime"], errors="coerce")  # fallback
                fy = int(d.year) if pd.notna(d) else None
            meta[did] = (code, fy, str(r.get("submitDateTime", "")))
    return meta


def main() -> int:
    cached = {Path(p).name.replace("_5.zip", "") for p in glob.glob(f"{DOCS}/*_5.zip")}
    print(f"=== オフライン所有抽出（キャッシュ docs/_5.zip = {len(cached)}）===")
    meta = collect_meta(cached)
    print(f"有報(docTypeCode=120) かつ キャッシュ済み = {len(meta)} 件")

    done = set()
    if Path(XBRL_PANEL).exists():
        prev = pd.read_parquet(XBRL_PANEL)
        done = set(prev["docID"].astype(str)) if "docID" in prev.columns else set()
    else:
        prev = pd.DataFrame()
    todo = [(d, m) for d, m in meta.items() if d not in done and m[1] is not None]
    print(f"既抽出={len(done)}  今回抽出={len(todo)}")

    rows = []
    for i, (did, (code, fy, submit)) in enumerate(todo, 1):
        try:
            df = ed.read_xbrl_csv(f"{DOCS}/{did}_5.zip")
            rec = parse_ownership_from_xbrl(df)
        except Exception as e:  # noqa: BLE001
            if i <= 5:
                print(f"  [warn] {did}: {str(e)[:70]}")
            continue
        if rec.get("foreign_pct") is None:
            continue
        rows.append({"docID": did, "Code": code, "fiscal_year": fy, "submit_dt": submit,
                     "foreign_pct": rec["foreign_pct"], "individual_pct": rec["individual_pct"],
                     "units_total": rec.get("units_total"), "quality_flag": rec["quality_flag"],
                     "source": "edinet_xbrl_offline"})
        if i % 250 == 0:
            print(f"  ...{i}/{len(todo)} 抽出 {len(rows)}")
    offline = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True) if rows else prev
    if offline.empty:
        print("抽出ゼロ（docs 未キャッシュ？）。終了。")
        return 0
    # 訂正等の重複は (Code, fiscal_year) で最新 submit_dt を採用
    offline = (offline.sort_values("submit_dt").drop_duplicates(["Code", "fiscal_year"], keep="last")
               .reset_index(drop=True))
    offline.to_parquet(XBRL_PANEL)
    print(f"\nオフラインパネル: {offline['Code'].nunique()} 社 / {len(offline)} 行 "
          f"(quality ok={int((offline['quality_flag']=='ok').sum())}) → {XBRL_PANEL}")

    # --- 4. MCP 照合（正しさの裏取り） ---
    existing = pd.read_parquet(CANON) if Path(CANON).exists() else pd.DataFrame()
    if not existing.empty:
        existing["Code"] = existing["Code"].astype(str)
        if "source" not in existing.columns:
            existing["source"] = "edinet_db_mcp"
    mcp = existing[existing["source"] == "edinet_db_mcp"] if not existing.empty else pd.DataFrame()
    if not mcp.empty:
        j = offline.merge(mcp[["Code", "fiscal_year", "foreign_pct"]],
                          on=["Code", "fiscal_year"], suffixes=("_off", "_mcp"))
        if len(j):
            d = (j["foreign_pct_off"] - j["foreign_pct_mcp"]).abs()
            print(f"\n--- MCP 照合（overlap {len(j)} 行）---")
            print(f"  |Δforeign_pct|: 平均 {d.mean():.2f}pt / 中央 {d.median():.2f}pt / "
                  f"≤1pt {100*(d<=1).mean():.0f}% / ≤3pt {100*(d<=3).mean():.0f}%")
            worst = j.loc[d.sort_values().index[-3:], ["Code", "fiscal_year",
                                                       "foreign_pct_off", "foreign_pct_mcp"]]
            print("  最大乖離:\n" + worst.to_string(index=False))

    # --- 5. 正準パネル（オフライン優先＋MCP gapfill） ---
    keep = ["Code", "fiscal_year", "foreign_pct", "individual_pct", "source"]
    off_k = offline.reindex(columns=keep)
    parts = [off_k] + ([existing.reindex(columns=keep)] if not existing.empty else [])
    canon = (pd.concat(parts, ignore_index=True)
             .drop_duplicates(["Code", "fiscal_year"], keep="first")   # offline 優先
             .sort_values(["Code", "fiscal_year"]).reset_index(drop=True))
    canon.to_parquet(CANON)
    nb = canon["source"].value_counts().to_dict()
    print(f"\n正準パネル → {CANON}: {canon['Code'].nunique()} 社 / {len(canon)} 行  source別 {nb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
