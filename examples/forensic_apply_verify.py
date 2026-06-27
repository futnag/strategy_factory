"""verify ワークフロー結果を cases.json に反映し、検証済みDBを作る（使い捨て・K不変）。

verify-fraud-cases ワークフローの出力（task output JSON）を読み、idx で cases.json に
逆検証結果（確度・不正期間・発覚日・上場廃止・監査法人・変更・SESC処分・訂正）をマージ。
likely_false は除外フラグ。出力: data/forensic/cases_verified.json ＋ 品質サマリ。

使い方: python examples/forensic_apply_verify.py --raw <task_output.json>
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "forensic"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()

    obj = json.loads(Path(args.raw).read_text(encoding="utf-8", errors="replace"))
    if isinstance(obj, dict) and "result" in obj:
        obj = obj["result"]
    verdicts = {v["idx"]: v for v in obj.get("verdicts", []) if "idx" in v}
    cases = json.loads((OUT / "cases.json").read_text(encoding="utf-8"))
    print(f"検証結果 {len(verdicts)} 件 / 事例 {len(cases)} 件")

    changed = collections.Counter()
    for i, c in enumerate(cases):
        v = verdicts.get(i)
        if not v:
            c["verified"] = False
            continue
        old = c.get("confidence")
        new = v.get("confidence")
        if new and new != old:
            changed[f"{old}→{new}"] += 1
        c["confidence"] = new or old
        for f_v, f_c in [("fraud_period", "fraud_period"), ("revelation_date", "revelation_date"),
                         ("delisting_date", "delisting_date"), ("auditor", "auditor"),
                         ("auditor_change", "auditor_change"), ("sesc_action", "sesc_action")]:
            val = v.get(f_v)
            if val and str(val).lower() not in ("", "unknown", "none", "n/a"):
                c[f_c] = val
        if v.get("corrections"):
            c["verify_corrections"] = v["corrections"]
        if v.get("sources"):
            c["sources"] = sorted(set((c.get("sources") or []) + v["sources"]))
        c["verified"] = True

    kept = [c for c in cases if c.get("confidence") != "likely_false"]
    dropped = [c for c in cases if c.get("confidence") == "likely_false"]
    (OUT / "cases_verified.json").write_text(
        json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n検証済み {sum(c.get('verified') for c in cases)}/{len(cases)} 件")
    print("確度の変化:", dict(changed) or "なし")
    print("確度分布(検証後):", dict(collections.Counter(c.get("confidence") for c in cases)))
    print(f"likely_false で除外: {len(dropped)} 件",
          [c.get("company") for c in dropped] if dropped else "")
    # 大手→中小の監査人交代が確認された事例
    b2s = [c for c in kept if "→" in str(c.get("auditor_change", ""))
           and any(k in str(c.get("auditor_change", "")) for k in ["中小", "小規模", "準大手", "シドー"])]
    print(f"\n大手→中小（疑い含む）監査人交代が記録された事例: {len(b2s)} 件")
    for c in b2s[:12]:
        print(f"  {c.get('company')[:24]}: {str(c.get('auditor_change'))[:50]}")
    print("\n保存:", OUT / "cases_verified.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
