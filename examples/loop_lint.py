"""ループ・ガードレールの機械検証（loop_lint）—「意志でなくコードで封じる」のプロセス版。

registry が p-hacking を構造的に封じたのと同様に、プロセス側のガードレール
（PLAYBOOK §A・SKILL 複製）をモデルの善意でなくコードで強制する。無人運用（/operator）の
必須ゲート＝違反があれば exit 1（呼び出し側は続行してはならない）。

検査:
  L1 接触禁止パスの変更検知（作業ツリー＋直近コミット）: docs/03・production/・phase2_*
  L2 PLAYBOOK §A の不変性（§A テキストの SHA-256 を baseline と照合）
  L3 事前登録の先行コミット: 各 docs/NN-*-preregistration.md の初回コミット時点で
     §5 が「（未実行）」であること（＝結果より先に登録がコミットされている）
  L4 K 予算: 直近30日に作られた scope の K ≤ 上限・直近7日の新規試行数 ≤ 週次上限
  L5 ledger.jsonl のスキーマ妥当性・IDEAS/BACKLOG のパース健全性
  L6 operator ロックの整合（stale ロックの検知）
設定: research_ops/loop_lint_baseline.json（§A ハッシュの更新は人間のみ＝--freeze-a）
実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\loop_lint.py [--freeze-a]
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

OPS = ROOT / "research_ops"
BASELINE = OPS / "loop_lint_baseline.json"
PLAYBOOK = ROOT / "research_loop" / "PLAYBOOK.md"
LOCK = OPS / "operator.lock"
FORBIDDEN = ["docs/03-research-findings.md", "invest_system/production/",
             "examples/phase2_"]
DEFAULTS = {"k_scope_limit": 8, "k_weekly_limit": 12, "lock_stale_hours": 6,
            "enforce_since": "2026-07-03"}  # ループ発足日＝K規律の適用開始
VIOLATIONS: list[str] = []      # HARD: ループ全停止
BUDGET: list[str] = []          # BUDGET: 研究サイクルのみ停止（データ/運用系は続行可）


def viol(check: str, msg: str) -> None:
    VIOLATIONS.append(f"[{check}] {msg}")
    print(f"  ✗ [HARD:{check}] {msg}")


def budget(check: str, msg: str) -> None:
    BUDGET.append(f"[{check}] {msg}")
    print(f"  ◼ [BUDGET:{check}] {msg}")


def ok(check: str, msg: str) -> None:
    print(f"  ✓ [{check}] {msg}")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8").stdout


def section_a_text() -> str:
    t = PLAYBOOK.read_text(encoding="utf-8")
    m = re.search(r"(## §A .*?)(?=\n## §B )", t, re.S)
    return m.group(1) if m else ""


def check_l1_forbidden() -> None:
    dirty = git("status", "--porcelain")
    recent = git("log", "-5", "--name-only", "--pretty=format:")
    for path in FORBIDDEN:
        if re.search(rf"^..? .*{re.escape(path)}", dirty, re.M):
            viol("L1", f"接触禁止パスに未コミット変更: {path}")
        elif path in recent:
            viol("L1", f"接触禁止パスが直近コミットで変更されている: {path}")
    if not any(v.startswith("[L1]") for v in VIOLATIONS):
        ok("L1", "接触禁止パス（docs/03・production・phase2_*）に変更なし")


def check_l2_section_a(baseline: dict, freeze: bool) -> None:
    h = hashlib.sha256(section_a_text().encode("utf-8")).hexdigest()
    if freeze or "playbook_a_sha256" not in baseline:
        baseline["playbook_a_sha256"] = h
        ok("L2", f"§A ハッシュを凍結: {h[:12]}…（人間操作）")
        return
    if h != baseline["playbook_a_sha256"]:
        viol("L2", "PLAYBOOK §A が baseline から変更されている"
                   "（意図的なら人間が --freeze-a で再凍結）")
    else:
        ok("L2", "PLAYBOOK §A 不変")


def check_l3_prereg_order() -> None:
    bad = 0
    docs = sorted(ROOT.glob("docs/[0-9][0-9]-*preregistration*.md"))
    for doc in docs:
        rel = doc.relative_to(ROOT).as_posix()
        commits = git("log", "--format=%H", "--follow", "--", rel).split()
        if not commits:
            viol("L3", f"{rel}: 未コミット（結果を出す前に登録コミットが必要）")
            bad += 1
            continue
        first = commits[-1]
        content = git("show", f"{first}:{rel}")
        if "## §5" in content and "未実行" not in content.split("## §5")[-1]:
            viol("L3", f"{rel}: 初回コミット時点で §5 に結果が入っている（先行登録違反）")
            bad += 1
    if not bad:
        ok("L3", f"事前登録の先行コミット順序 OK（{len(docs)}件検査）")


def check_l4_k_budget(baseline: dict) -> None:
    try:
        from invest_system.validation.registry import default_registry
        with default_registry(str(ROOT / "data" / "research_trials.db")) as reg:
            rows = reg.recent_trials(limit=500)
    except Exception as e:  # noqa: BLE001
        viol("L4", f"registry 読み取り不可: {e}")
        return
    now = datetime.now()
    week_ago = (now - timedelta(days=7)).isoformat()
    since = baseline.get("enforce_since", "2026-07-03")
    # 新規作成（preregistered_at）のみカウント＝冪等再実行の completed_at 更新を除外
    new_week = [r for r in rows if (r["preregistered_at"] or "") >= week_ago
                and (r["preregistered_at"] or "") >= since]
    if len(new_week) > baseline.get("k_weekly_limit", 12):
        budget("L4", f"週次 K レート超過: 新規 {len(new_week)} > "
                     f"{baseline.get('k_weekly_limit', 12)}"
                     "＝研究サイクルは今週停止（データ/運用系は続行可）")
    else:
        ok("L4", f"週次 K 消費 {len(new_week)}/{baseline.get('k_weekly_limit', 12)}")
    # scope 上限はループ発足以降に生まれた scope のみ（発足前の SBuMT 大量scan等は対象外）
    from collections import Counter
    born: dict[str, str] = {}
    for r in rows:
        s, t = r["scope"], r["preregistered_at"] or ""
        born[s] = min(born.get(s, t), t) if s in born else t
    scope_counts = Counter(r["scope"] for r in rows if born.get(r["scope"], "") >= since)
    over = {s: c for s, c in scope_counts.items()
            if c > baseline.get("k_scope_limit", 8)}
    if over:
        viol("L4", f"scope K 予算超過（{since} 以降発足分）: {over}")
    else:
        ok("L4", f"scope 別 K 予算 OK（{since} 以降 {len(scope_counts)} scope）")


def check_l5_schemas() -> None:
    led = ROOT / "research_loop" / "ledger.jsonl"
    req = {"cycle_id", "date", "scope", "verdict", "k_added"}
    n_bad = 0
    if led.exists():
        for i, line in enumerate(led.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not req.issubset(row):
                    n_bad += 1
            except json.JSONDecodeError:
                n_bad += 1
        if n_bad:
            viol("L5", f"ledger.jsonl に不正行 {n_bad}件")
        else:
            ok("L5", "ledger スキーマ OK")
    for name, pat in [("IDEAS.md", r"^#{2,4}\s*I-\d+\s*(💡|⬆|🗑|🧪)"),
                      ("BACKLOG.md", r"^#{2,4}\s*(⬜|🔬|✅|❌|⏸)")]:
        p = ROOT / "research_loop" / name
        if p.exists() and not re.search(pat, p.read_text(encoding="utf-8"), re.M):
            viol("L5", f"{name}: 状態タグ付き見出しが1件もパースできない（書式破壊疑い）")
    if not any(v.startswith("[L5]") and "書式" in v for v in VIOLATIONS):
        ok("L5", "IDEAS/BACKLOG の書式 OK")


def check_l6_lock(baseline: dict) -> None:
    if not LOCK.exists():
        ok("L6", "operator ロックなし")
        return
    try:
        info = json.loads(LOCK.read_text(encoding="utf-8"))
        age_h = (datetime.now() - datetime.fromisoformat(info["started_at"])
                 ).total_seconds() / 3600
        if age_h > baseline.get("lock_stale_hours", 6):
            viol("L6", f"stale ロック検知（{age_h:.1f}h 前・{info.get('holder', '?')}）"
                       "＝前セッションの異常終了疑い。内容確認のうえ手動削除を")
        else:
            ok("L6", f"ロック保持中（{age_h:.1f}h・{info.get('holder', '?')}）"
                     "＝並行起動は禁止")
    except Exception as e:  # noqa: BLE001
        viol("L6", f"ロックファイル破損: {e}")


def main(argv: list[str]) -> int:
    print(f"loop_lint  {datetime.now():%Y-%m-%d %H:%M}")
    baseline = (json.loads(BASELINE.read_text(encoding="utf-8"))
                if BASELINE.exists() else dict(DEFAULTS))
    check_l1_forbidden()
    check_l2_section_a(baseline, "--freeze-a" in argv)
    check_l3_prereg_order()
    check_l4_k_budget(baseline)
    check_l5_schemas()
    check_l6_lock(baseline)
    BASELINE.parent.mkdir(exist_ok=True)
    BASELINE.write_text(json.dumps(baseline, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    if VIOLATIONS:
        print(f"\n== HARD 違反 {len(VIOLATIONS)}件＝ループ全停止（人間へエスカレーション）==")
        return 1
    if BUDGET:
        print(f"\n== BUDGET 制約 {len(BUDGET)}件＝研究サイクルのみ停止・データ/運用系は続行可 ==")
        return 2
    print("\n== 全ガードレール通過 ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
