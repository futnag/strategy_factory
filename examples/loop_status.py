"""research_loop の状態ブートストラップ：新セッションが1画面で現状を把握する。

/research-cycle の Step 0 で最初に実行する。表示内容：
  1) 永続レジストリ（scope別 K・V[SR]・直近試行）
  2) research_loop/BACKLOG.md の状態別件数と未着手（⬜）一覧
  3) research_loop/ledger.jsonl 末尾5サイクル
  4) data/ 主要ストアの鮮度（既定はファイル名/mtime。--deep で parquet 実日付）

常に exit 0（欠損はセクション内に警告表示）。純 stdout・書き込みなし。
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invest_system.validation.registry import default_registry  # noqa: E402

LOOP = ROOT / "research_loop"
_GLYPHS = ("⬜", "🔬", "✅", "❌", "⏸")


def _section(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)))


def show_registry(limit: int = 15) -> None:
    _section("試行レジストリ（data/research_trials.db）")
    try:
        with default_registry(str(ROOT / "data" / "research_trials.db")) as reg:
            scopes = reg.list_scopes()
            if not scopes:
                print("（試行は未記録）")
                return
            print(f"{'scope':<32}{'K':>6}{'V[SR]':>10}")
            for scope, k, srv in scopes:
                print(f"{scope:<32}{k:>6}{srv:>10.4f}")
            print(f"-- scope数={len(scopes)}  総試行={sum(k for _, k, _ in scopes)}")
            print(f"\n直近{limit}試行（新しい順）:")
            for r in reg.recent_trials(limit=limit):
                sr = "  --  " if r["sharpe"] is None else f"{r['sharpe']:+.2f}"
                when = (r["completed_at"] or r["preregistered_at"] or "")[:10]
                print(f"  {when}  {r['scope']:<26} {sr}  "
                      f"{r['status']:<13} {r['strategy_id'] or ''}")
    except Exception as e:  # noqa: BLE001
        print(f"（レジストリ読み取り不可: {e}）")


def show_backlog() -> None:
    _section("仮説キュー（research_loop/BACKLOG.md）")
    path = LOOP / "BACKLOG.md"
    if not path.exists():
        print("（BACKLOG.md 不在）")
        return
    counts = {g: 0 for g in _GLYPHS}
    open_items: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#{2,4}\s*(⬜|🔬|✅|❌|⏸)\s*(.+)$", line.strip())
        if not m:
            continue
        counts[m.group(1)] += 1
        if m.group(1) == "⬜":
            open_items.append(m.group(2).strip())
    print("  ".join(f"{g}{n}" for g, n in counts.items()))
    if open_items:
        print("未着手（⬜）:")
        for t in open_items:
            print(f"  - {t}")
    else:
        print("未着手（⬜）なし → Step 1 で新規候補を生成する。")


def show_ideas(top: int = 5) -> None:
    _section("アイデア台帳（research_loop/IDEAS.md）")
    path = LOOP / "IDEAS.md"
    if not path.exists():
        print("（IDEAS.md 不在）")
        return
    tags = {"💡": 0, "⬆": 0, "🗑": 0, "🧪": 0}
    candidates: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#{2,4}\s*I-\d+\s*(💡|⬆|🗑|🧪)\s*(.+)$", line.strip())
        if not m:
            continue
        tags[m.group(1)] += 1
        if m.group(1) == "💡":
            candidates.append(m.group(2).strip())
    print(f"💡candidate {tags['💡']}  ⬆promoted {tags['⬆']}  "
          f"🗑rejected {tags['🗑']}  🧪tested {tags['🧪']}")
    for t in candidates[:top]:
        print(f"  💡 {t}")
    if len(candidates) > top:
        print(f"  …他 {len(candidates) - top} 件")


def show_ledger(tail: int = 5) -> None:
    _section(f"サイクル台帳（research_loop/ledger.jsonl 末尾{tail}件）")
    path = LOOP / "ledger.jsonl"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip()] if path.exists() else []
    if not lines:
        print("（サイクル実績なし）")
        return
    for ln in lines[-tail:]:
        try:
            row = json.loads(ln)
            dsr = row.get("best_dsr")
            dsr_s = "  -- " if dsr is None else f"{dsr:.2f}"
            print(f"  {row.get('date','')}  {row.get('cycle_id',''):<34} "
                  f"{row.get('verdict',''):<9} DSR={dsr_s}  "
                  f"{row.get('failure_type') or ''}")
        except json.JSONDecodeError:
            print(f"  （不正な行: {ln[:60]}…）")


def _newest(path: Path, pattern: str = "*") -> str:
    """ディレクトリ内で名前順最大のファイル名（by-date 命名なら＝最新日付）。"""
    if not path.exists():
        return "（不在）"
    files = sorted(p.name for p in path.glob(pattern) if p.is_file())
    return files[-1] if files else "（空）"


def _mtime(path: Path) -> str:
    if not path.exists():
        return "（不在）"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def show_freshness(deep: bool = False) -> None:
    _section("データ鮮度（data/）")
    d = ROOT / "data"
    rows = [
        ("株価 by-date（jquants/daily/）", _newest(d / "jquants" / "daily", "*.parquet")),
        ("財務（jquants/fins_summary/）", _newest(d / "jquants" / "fins_summary", "*.parquet")),
        ("wide ストア（adj_close.parquet mtime）",
         _mtime(d / "processed" / "equities" / "wide" / "adj_close.parquet")),
        ("外部価格（investers/ mtime）", _mtime(d / "investers")),
        ("マクロ（supplemental/ mtime）", _mtime(d / "supplemental")),
        ("EDINET 一覧（edinet/list/）", _newest(d / "edinet" / "list", "*.parquet")),
        ("TDnet（tdnet/）", _newest(d / "tdnet", "*.parquet")),
        ("ToSTNeT（jpx_tostnet/）", _newest(d / "jpx_tostnet", "*.parquet")),
        ("JSF（jsf/）", _newest(d / "jsf", "*.parquet")),
    ]
    for name, val in rows:
        print(f"  {name:<38} {val}")
    if deep:
        try:
            import pandas as pd
            wide = d / "processed" / "equities" / "wide" / "adj_close.parquet"
            if wide.exists():
                idx = pd.read_parquet(wide).index
                print(f"  [deep] wide adj_close 実日付範囲: {idx.min():%Y-%m-%d} 〜 "
                      f"{idx.max():%Y-%m-%d}")
        except Exception as e:  # noqa: BLE001
            print(f"  [deep] 読み取り失敗: {e}")


def main(argv: list[str]) -> int:
    deep = "--deep" in argv
    print("research_loop 状態サマリ  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    show_registry()
    show_backlog()
    show_ideas()
    show_ledger()
    show_freshness(deep=deep)
    print("\n※ 次に読む: research_loop/{PLAYBOOK,HEURISTICS,LESSONS,BACKLOG}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
