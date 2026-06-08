"""J-Quants の by-date データセット単体ダウンローダ（手元のターミナルで直接実行する用）。

Claude を介さず、あなた自身のターミナルで動かすための自己完結スクリプトです
（=> Claude のプラン使用量を一切消費しません）。1営業日ずつ進捗・ETA を表示し、Ctrl-C で
中断してもキャッシュ済みの日はスキップして「続きから」再開できます（冪等）。

対象（カタログの by-date データセット）:
  fins_summary   財務サマリー全件ミラー（案A：日付別・その日の全開示企業）
  daily_quotes   全銘柄日次株価四本値（日付別・全銘柄）
  options_225    日経225オプション四本値（IV等）
  margin_alert   日々公表信用取引残高
  weekly_margin  信用取引週末残高（週次）
  short_positions 空売り残高報告
  （他、catalog.DATASETS のもの）

────────────────────────────────────────────────────────────────────────
使い方（まず .env に J_QUANTS_API_KEY を設定しておく）

  PowerShell:
    $env:J_QUANTS_MIN_INTERVAL = "0.7"        # Standard はこの間隔が安全・高速
    .venv\\Scripts\\python.exe examples\\download_jquants.py --list           # 状況一覧
    .venv\\Scripts\\python.exe examples\\download_jquants.py --dataset fins_summary
    .venv\\Scripts\\python.exe examples\\download_jquants.py --dataset daily_quotes
    .venv\\Scripts\\python.exe examples\\download_jquants.py --all             # 未完を全部

  bash / WSL / macOS:
    J_QUANTS_MIN_INTERVAL=0.7 .venv/bin/python examples/download_jquants.py --dataset fins_summary

中断したら同じコマンドを再実行するだけで続きから再開します。市場データは data/ 配下
（gitignore 済）にのみ保存され、コミットされません（J-Quants 利用規約）。
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
from invest_system.data.catalog import DATASETS  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.data.updater import (  # noqa: E402
    Manifest,
    candidate_dates,
    missing_dates,
    scan_cache_dates,
)

START = "2016-06-13"   # Standard の取得下限（10年・他データと統一）


def _fmt_dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _plan(name, base, manifest, until):
    ds = DATASETS[name]
    fetched = manifest.fetched(ds.name) | scan_cache_dates(ds, base)
    cands = candidate_dates(ds.cadence, START, until)
    return ds, cands, missing_dates(cands, fetched)


def _list(base, manifest, until) -> int:
    print(f"{'dataset':<16}{'cadence':<9}{'候補':>7}{'取得済':>8}{'欠損':>8}  maintained")
    print("-" * 64)
    for name in DATASETS:
        ds, cands, todo = _plan(name, base, manifest, until)
        print(f"{name:<16}{ds.cadence:<9}{len(cands):>7}{len(cands) - len(todo):>8}"
              f"{len(todo):>8}  {ds.maintained}")
    return 0


def _download(name, base, manifest, until) -> int:
    ds, cands, todo = _plan(name, base, manifest, until)
    print(f"[{name}] 候補 {len(cands)} / 取得済 {len(cands) - len(todo)} / 今回 {len(todo)}")
    if not todo:
        print(f"[{name}] 差分なし。")
        return 0
    t0 = time.time()
    rows = ok = 0
    try:
        for i, dt in enumerate(todo, 1):
            try:
                n = len(ds.fetch_by_date(dt))
                rows += n
                ok += 1
                manifest.mark(ds.name, dt)
                el = time.time() - t0
                eta = (len(todo) - i) / (i / el) if el > 0 else 0
                print(f"\r[{name}] [{i:>5}/{len(todo)}] {dt}  {n:>6,}行  "
                      f"累計 {rows:>10,}  経過 {_fmt_dur(el)}  残り≈{_fmt_dur(eta)}    ",
                      end="", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"\r[{name}] [{i:>5}/{len(todo)}] {dt}  [warn] {str(e)[:55]}"
                      f"{' ' * 18}")
            if i % 50 == 0:
                manifest.save()
    except KeyboardInterrupt:
        manifest.save()
        print(f"\n[{name}] 中断。今回 {ok} 日 / {rows:,} 行取得。再実行で続きから。")
        return 130
    finally:
        manifest.save()
    cache = base / ds.cache_subdir
    mb = (sum(p.stat().st_size for p in cache.glob("*.parquet")) / 1e6
          if cache.exists() else 0)
    print(f"\n[{name}] 完了: {ok}/{len(todo)} 日, 累計 {rows:,} 行, "
          f"キャッシュ {mb:,.0f} MB ({cache})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="J-Quants by-date データセットの単体ダウンローダ")
    ap.add_argument("--dataset", help=f"取得対象（{', '.join(DATASETS)}）")
    ap.add_argument("--all", action="store_true",
                    help="maintained な by-date データセットを全て最新化")
    ap.add_argument("--list", action="store_true", help="各データセットの取得状況を表示")
    ap.add_argument("--until", default=None, help="終了日 YYYY-MM-DD（既定 本日）")
    args = ap.parse_args(argv)

    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が設定されていません。")
        return 1

    base = jq._CACHE
    manifest = Manifest(base / "manifest.json")
    until = pd.Timestamp(args.until) if args.until else pd.Timestamp.today().normalize()

    if args.list or (not args.dataset and not args.all):
        rc = _list(base, manifest, until)
        if not args.dataset and not args.all:
            print("\n--dataset NAME で個別取得、--all で maintained を全件取得します。")
        return rc

    if not get_env("J_QUANTS_MIN_INTERVAL"):
        print("ヒント: Standard では J_QUANTS_MIN_INTERVAL=0.7 を設定すると安全・高速です。")

    if args.all:
        names = [n for n, d in DATASETS.items() if d.maintained]
    else:
        if args.dataset not in DATASETS:
            print(f"ERROR: 未知のデータセット '{args.dataset}'。候補: {', '.join(DATASETS)}")
            return 2
        names = [args.dataset]

    rc = 0
    for name in names:
        rc = _download(name, base, manifest, until) or rc
        if rc == 130:        # Ctrl-C は全体中断
            break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
