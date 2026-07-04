"""ループ振り返りの集計（/loop-retro の決定論的コア）。

ledger.jsonl・registry・HEURISTICS/IDEAS/BACKLOG から、ループ自体の性能指標を集計する
（解釈と方針提案は skill=LLM の仕事）:
  - サイクル・ファネル: verdict 別件数・K 消費・所要時間・領域別内訳
  - 失敗型の分布・「事前に殺せたか」（後続 H-n が引用したサイクル数）
  - scout 効率: 調査→IDEAS→昇格→検証済みの転換率・棄却理由の分布
  - K 効率: 消費 K あたりの新規ヒューリスティクス数（=学習密度の代理）
実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\loop_retro.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

LOOP = ROOT / "research_loop"


def main() -> int:
    print(f"ループ振り返り集計  {datetime.now():%Y-%m-%d %H:%M}")

    # --- サイクル・ファネル ---
    rows = [json.loads(x) for x in (LOOP / "ledger.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    print(f"\n== サイクル（{len(rows)}件・{rows[0]['date']}〜{rows[-1]['date']}）==")
    verd = Counter(r["verdict"] for r in rows)
    k_total = sum(r.get("k_added", 0) for r in rows)
    mins = [r.get("wall_minutes") for r in rows if r.get("wall_minutes")]
    print(f"  verdict: {dict(verd)}  K消費計={k_total}  "
          f"平均所要={sum(mins) / len(mins):.0f}分" if mins else "")
    print(f"  領域別: {dict(Counter(r.get('domain', '?') for r in rows))}")
    print("  失敗型:")
    for r in rows:
        print(f"    {r['scope']:<24} DSR={r.get('best_dsr')}  {r.get('failure_type', '')[:60]}")

    # --- 自己改善の密度 ---
    amendments = [a for r in rows for a in r.get("amendments", [])]
    h_added = [a for a in amendments if a.startswith("H-")]
    print(f"\n== 自己改善 ==")
    print(f"  サイクル由来の改訂: {len(amendments)}件（うち新ヒューリスティクス {len(h_added)}: "
          f"{[a.split()[0] for a in h_added]}）")
    print(f"  学習密度: {len(h_added) / max(k_total, 1):.2f} H/K・"
          f"{len(h_added) / max(len(rows), 1):.2f} H/サイクル")
    heur = (LOOP / "HEURISTICS.md").read_text(encoding="utf-8")
    hs = re.findall(r"^## (H-\d+) .*— (\w+)", heur, re.M)
    print(f"  HEURISTICS 総数: {len(hs)}（active {sum(1 for _, s in hs if s == 'active')}）")

    # --- IDEAS/scout 転換 ---
    ideas = (LOOP / "IDEAS.md").read_text(encoding="utf-8")
    tags = Counter(m for m in re.findall(r"^#{2,4}\s*I-\d+\s*(💡|⬆|🗑|🧪)", ideas, re.M))
    n_i = sum(tags.values())
    print(f"\n== scout 転換（IDEAS {n_i}件）==")
    print(f"  💡備蓄 {tags.get('💡', 0)}  ⬆昇格 {tags.get('⬆', 0)}  "
          f"🧪検証済 {tags.get('🧪', 0)}  🗑棄却 {tags.get('🗑', 0)}")
    print(f"  昇格率 {(tags.get('⬆', 0) + tags.get('🧪', 0)) / max(n_i, 1):.0%}・"
          f"検証到達率 {tags.get('🧪', 0) / max(n_i, 1):.0%}")
    surveys = sorted((LOOP / "surveys").glob("*.md"))
    print(f"  調査回数: {len(surveys)}")

    # --- BACKLOG 状態 ---
    backlog = (LOOP / "BACKLOG.md").read_text(encoding="utf-8")
    bt = Counter(m for m in re.findall(r"^#{2,4}\s*(⬜|🔬|✅|❌|⏸)", backlog, re.M))
    print(f"\n== BACKLOG: {dict(bt)} ==")

    # --- registry 全景 ---
    try:
        from invest_system.validation.registry import default_registry
        with default_registry(str(ROOT / "data" / "research_trials.db")) as reg:
            scopes = reg.list_scopes()
        print(f"\n== registry: {len(scopes)} scope・総 K={sum(k for _, k, _ in scopes)} ==")
    except Exception as e:  # noqa: BLE001
        print(f"（registry 読み取り不可: {e}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
