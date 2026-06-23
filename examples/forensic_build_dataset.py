"""ワークフロー出力 → 重複排除済み不正事例DB（cases.json / cases.csv）。

jp-accounting-fraud-dataset ワークフローの生出力（JSON）を読み、英名/和名/株式会社
表記ゆれを正規化して重複排除し、構造化ラベルDBとして保存する。K不変・オフライン。

使い方: python examples/forensic_build_dataset.py --raw <workflow_output_path>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "forensic"

FIELDS = ["company", "sec_code", "industry", "fraud_type", "mechanism",
          "fraud_period", "revelation_date", "restatement_date", "sesc_action",
          "outcome", "auditor", "auditor_change", "magnitude", "confidence"]


def _norm_name(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"[（(].*?[)）]", "", s)              # 英名・補足の括弧除去
    s = re.sub(r"株式会社|ホールディングス|グループ|\bHD\b|[\s　]|,|\.|株", "", s)
    return re.sub(r"[A-Za-z]+", "", s).lower().strip()


def _norm_code(s: str) -> str:
    s = str(s or "").strip().upper()
    if s in ("", "UNKNOWN", "NONE", "N/A"):
        return ""
    m = re.match(r"^([0-9A-Z]{4,5})", s)
    if not m:
        return ""
    c = m.group(1)
    return c[:4] if len(c) == 5 and c.endswith("0") else c   # 72030→7203, 260A0→260A


def _key(c: dict) -> str:
    return _norm_code(c.get("sec_code")) or ("NAME:" + _norm_name(c.get("company")))


def _merge(a: dict, b: dict) -> None:
    a["red_flags"] = sorted(set(a.get("red_flags", [])) | set(b.get("red_flags", [])))
    a["sources"] = sorted(set(a.get("sources", [])) | set(b.get("sources", [])))
    if b.get("confidence") == "confirmed":
        a["confidence"] = "confirmed"
    for f in FIELDS:
        if (not a.get(f) or a.get(f) in ("unknown", "none", "")) and \
                b.get(f) and b.get(f) not in ("unknown", "none", ""):
            a[f] = b[f]
    # corrections（検証の補正メモ）は長い方を残す
    if len(str(b.get("corrections", ""))) > len(str(a.get("corrections", ""))):
        a["corrections"] = b.get("corrections")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()

    raw = Path(args.raw).read_text(encoding="utf-8", errors="replace")
    obj = json.loads(raw)
    if isinstance(obj, dict) and "result" in obj and isinstance(obj["result"], dict):
        obj = obj["result"]                          # タスク出力ラッパを剥がす
    cases = obj.get("cases", obj if isinstance(obj, list) else [])
    print(f"生出力: {len(cases)} 件")

    dedup: dict = {}
    for c in cases:
        if not c.get("company"):
            continue
        c.setdefault("red_flags", [])
        c.setdefault("sources", [])
        k = _key(c)
        if k in dedup:
            _merge(dedup[k], c)
        else:
            dedup[k] = dict(c)
    uniq = list(dedup.values())
    print(f"重複排除後: {len(uniq)} 社")

    # 正規化コード列を付与
    for c in uniq:
        c["sec_code_norm"] = _norm_code(c.get("sec_code"))

    (OUT / "cases.json").write_text(
        json.dumps(uniq, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV（赤旗・出典は ; 連結）
    flat = []
    for c in uniq:
        row = {f: c.get(f, "") for f in FIELDS}
        row["sec_code_norm"] = c.get("sec_code_norm", "")
        row["n_red_flags"] = len(c.get("red_flags", []))
        row["red_flags"] = " ; ".join(c.get("red_flags", []))
        row["n_sources"] = len(c.get("sources", []))
        flat.append(row)
    df = pd.DataFrame(flat)
    df.to_csv(OUT / "cases.csv", index=False, encoding="utf-8-sig")

    # サマリ
    print("\n--- サマリ ---")
    print("confidence 別:", df["confidence"].value_counts().to_dict())
    print("コード判明:", int((df["sec_code_norm"] != "").sum()), "/", len(df))
    print("不正類型 上位:", df["fraud_type"].value_counts().head(8).to_dict())
    print("\n保存:", OUT / "cases.json", "/", OUT / "cases.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
