"""EDINET 有価証券報告書（三表明細）の全上場バックフィル — 手元ターミナルで直接実行。

GKX Phase 2 のデータ層。公式 EDINET API v2（type=5 = XBRL→CSV 変換済み）を一次ソースに、
全上場企業の有報を「最古年優先」で取得しローカルにミラーする。Claude を介さない自己完結
スクリプト（=> プラン使用量を消費しない）。Ctrl-C 中断・再実行で「続きから」再開（冪等）。

なぜ最古年優先か：公式 API の保持は前方ローリング約 10 年。最古年（2016/FY2015）は窓の縁で、
日々取得不能になっていく＝取り返しのつかない生バイトを先に確保する（handoff §1-A）。

二段構え（どちらも再開可能・冪等）:
  phase=list  書類一覧の by-date ミラーを start..until に延伸（1 日 1 リクエスト・軽い）。
              どの docID が有報かを知るための索引。空の日もマーカー保存で再取得しない。
  phase=docs  ミラー済み一覧から有報（docType 120/130・csvFlag=1）を抽出し、
              periodEnd 昇順（最古年優先）で type=5 を取得（重い・3〜5 秒間隔推奨）。
              取得済み zip はスキップ＝再実行で続きから。

レート: EDINET は数値上限非公表だが「1 リクエスト 3〜5 秒間隔」を空けないと接続を切られる。
        --interval（既定 3.5 秒）で全リクエストにスロットルを掛ける（指数バックオフは内蔵）。

保存先: data/edinet/（gitignore 済）。list/{YYYYMMDD}.parquet, docs/{docID}_5.zip。コミット不可。

使い方（まず .env に EDINET_API_KEY）:
  PowerShell:
    .venv\\Scripts\\python.exe examples\\download_edinet.py --plan
    .venv\\Scripts\\python.exe examples\\download_edinet.py --phase list --start 2016-01-01
    .venv\\Scripts\\python.exe examples\\download_edinet.py --phase docs
    .venv\\Scripts\\python.exe examples\\download_edinet.py            # list→docs を通しで
  bash / WSL / macOS:
    .venv/bin/python examples/download_edinet.py --plan
中断したら同じコマンドを再実行するだけで続きから再開します。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # Windows コンソールでも日本語・進捗行を文字化けさせない
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001  # pragma: no cover
    pass

import pandas as pd  # noqa: E402

from invest_system.config import get_env  # noqa: E402
from invest_system.data.catalog import EDINET_DATASETS  # noqa: E402
from invest_system.data.sources import edinet as ed  # noqa: E402
from invest_system.data.updater import (  # noqa: E402
    Manifest, candidate_dates, missing_dates, scan_cache_dates)

START = "2016-06-10"          # 最古年（公式 ~10 年窓の縁）。昇順で回す＝最古年優先。
DOCS_DIR = ed._CACHE / "docs"
LIST_DS = EDINET_DATASETS["edinet_docs"]
_DOC_COLS = ["docID", "secCode", "periodEnd", "submitDateTime"]


def _fmt_dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _cached_doc_ids() -> set[str]:
    """data/edinet/docs/{docID}_5.zip から取得済み docID を復元（再開の根拠）。"""
    return {p.stem.rsplit("_", 1)[0] for p in DOCS_DIR.glob("*_5.zip")}


def _list_todo(base, manifest, start, until):
    fetched = manifest.fetched(LIST_DS.name) | scan_cache_dates(LIST_DS, base)
    cands = candidate_dates("daily", start, until)        # 昇順＝最古年優先
    return cands, missing_dates(cands, fetched)


def mirror_list(base, manifest, start, until) -> int:
    """書類一覧 by-date ミラーを start..until に延伸（欠損営業日のみ・再開可）。"""
    cands, todo = _list_todo(base, manifest, start, until)
    print(f"[list] 候補 {len(cands)} 営業日 / 取得済 {len(cands) - len(todo)} / 今回 {len(todo)}")
    if not todo:
        print("[list] 差分なし。")
        return 0
    t0 = time.time()
    try:
        for i, dt in enumerate(todo, 1):
            try:
                n = len(ed.fetch_documents_list(dt))
                manifest.mark(LIST_DS.name, dt)
                el = time.time() - t0
                eta = (len(todo) - i) / (i / el) if el > 0 else 0
                print(f"\r[list] [{i:>5}/{len(todo)}] {dt}  {n:>5} 件  "
                      f"経過 {_fmt_dur(el)}  残り≈{_fmt_dur(eta)}     ", end="", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"\r[list] [{i:>5}/{len(todo)}] {dt}  [warn] {str(e)[:50]}{' ' * 16}")
            if i % 50 == 0:
                manifest.save()
    except KeyboardInterrupt:
        manifest.save()
        print("\n[list] 中断。再実行で続きから。")
        return 130
    finally:
        manifest.save()
    print(f"\n[list] 完了: {len(todo)} 日ミラー。")
    return 0


def scan_annual_docs() -> pd.DataFrame:
    """ミラー済み一覧から有報（120/130・csvFlag=1・secCode 有）を最古年優先で列挙。"""
    rows = []
    for p in sorted((ed._CACHE / "list").glob("*.parquet")):
        df = pd.read_parquet(p)
        if "_empty" in df.columns or "docTypeCode" not in df.columns:
            continue
        a = ed.annual_report_documents(df)
        if len(a):
            a = a[a["csvFlag"].astype(str) == "1"]
            if len(a):
                rows.append(a[_DOC_COLS])
    if not rows:
        return pd.DataFrame(columns=_DOC_COLS)
    docs = pd.concat(rows, ignore_index=True).dropna(subset=["docID"])
    docs = docs.drop_duplicates("docID")
    return docs.sort_values("periodEnd", na_position="last").reset_index(drop=True)


def backfill_docs(limit=None) -> int:
    """有報 type=5 を最古年優先で取得（取得済みはスキップ）。"""
    docs = scan_annual_docs()
    if docs.empty:
        print("[docs] 一覧に有報なし。先に --phase list を実行してください。")
        return 0
    cached = _cached_doc_ids()
    todo = docs[~docs["docID"].isin(cached)]
    if limit:
        todo = todo.head(limit)
    yrs = docs["periodEnd"].astype(str).str[:4]
    have = len(set(docs["docID"]) & cached)
    print(f"[docs] 有報(csvFlag=1) {len(docs)} 件 / 取得済 {have} / 今回 {len(todo)}  "
          f"期末年 {yrs.min()}〜{yrs.max()}")
    if todo.empty:
        print("[docs] 差分なし。")
        return 0
    t0 = time.time()
    ok = 0
    mb = 0.0
    try:
        for i, (_, r) in enumerate(todo.iterrows(), 1):
            did, yr = r["docID"], str(r["periodEnd"])[:4]
            try:
                p = ed.fetch_document(did, doc_type=5)
                ok += 1
                mb += p.stat().st_size / 1e6
                el = time.time() - t0
                eta = (len(todo) - i) / (i / el) if el > 0 else 0
                print(f"\r[docs] [{i:>6}/{len(todo)}] {did} FY{yr}  "
                      f"累計 {mb:>7,.0f}MB  経過 {_fmt_dur(el)}  残り≈{_fmt_dur(eta)}    ",
                      end="", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"\r[docs] [{i:>6}/{len(todo)}] {did} FY{yr}  "
                      f"[warn] {str(e)[:45]}{' ' * 12}")
    except KeyboardInterrupt:
        print(f"\n[docs] 中断。今回 {ok} 件 / {mb:,.0f}MB。再実行で続きから。")
        return 130
    print(f"\n[docs] 完了: {ok}/{len(todo)} 件取得, 累計 {mb:,.0f}MB（{DOCS_DIR}）")
    return 0


def plan(base, manifest, start, until, interval) -> int:
    cands, todo = _list_todo(base, manifest, start, until)
    print(f"[plan/list] {start}〜{until}: 候補 {len(cands)} 営業日 / 欠損 {len(todo)}"
          f"  推定 {_fmt_dur(len(todo) * interval)}")
    docs = scan_annual_docs()
    cached = _cached_doc_ids()
    n = 0 if docs.empty else int((~docs["docID"].isin(cached)).sum())
    print(f"[plan/docs] 現ミラー範囲の有報 {len(docs)} 件 / 取得済 {len(docs) - n} / 残り {n}"
          f"  推定 {_fmt_dur(n * interval)}・約 {n * 0.2:,.0f}MB（@0.2MB/件）")
    print("※ docs 件数は『現在の一覧ミラー範囲』に依存。先に list を 2016 まで延伸すると増える。")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="EDINET 有報 全上場バックフィル（最古年優先・再開可）")
    ap.add_argument("--phase", choices=["list", "docs", "all"], default="all")
    ap.add_argument("--start", default=START, help=f"一覧ミラー開始日（既定 {START}＝最古年）")
    ap.add_argument("--until", default=None, help="終端日 YYYY-MM-DD（既定 前営業日）")
    ap.add_argument("--interval", type=float, default=3.5,
                    help="リクエスト間隔秒（既定 3.5＝EDINET 推奨 3〜5 秒）")
    ap.add_argument("--limit", type=int, default=None, help="docs を先頭 N 件のみ（試走用）")
    ap.add_argument("--plan", action="store_true", help="取得予定だけ表示（取得しない）")
    args = ap.parse_args(argv)

    if not get_env("EDINET_API_KEY"):
        print("ERROR: .env に EDINET_API_KEY が設定されていません。")
        return 1

    ed._MIN_INTERVAL = args.interval                 # 全リクエストに適用（3〜5 秒推奨）
    base = ed._CACHE
    manifest = Manifest(base / "manifest.json")
    until = (args.until or (pd.Timestamp.today().normalize()
                            - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

    if args.plan:
        return plan(base, manifest, args.start, until, args.interval)

    print(f"=== EDINET 有報バックフィル  interval={args.interval}s  {args.start}〜{until} ===")
    rc = 0
    if args.phase in ("list", "all"):
        rc = mirror_list(base, manifest, args.start, until)
        if rc == 130:
            return rc
    if args.phase in ("docs", "all"):
        rc = backfill_docs(limit=args.limit) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
