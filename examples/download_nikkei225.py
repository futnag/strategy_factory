"""日経225オプション四本値の単体ダウンローダ（手元のターミナルで直接実行する用）。

Claude を介さず、あなた自身のターミナルで動かすための自己完結スクリプトです。
=> Claude のプラン使用量を一切消費しません。1営業日ずつ進捗・ETA を表示し、Ctrl-C で
中断してもキャッシュ済みの日はスキップして「続きから」再開できます（冪等）。

取得対象: /derivatives/bars/daily/options/225
  各営業日の全契約（IV=インプライドボラ・Theo=理論価格・Settle=清算値・OI=建玉・
  UnderPx=原資産NK225・Strike など, 1日 ≈ 9,600 行）。
期間: Standard の 2016-06-13 〜 本日（--from / --until で変更可）。
規模: 初回フルは約2,400営業日・20〜35分・数百MB級。以降は未取得日（差分）だけ取得。

────────────────────────────────────────────────────────────────────────
使い方（まず .env に J_QUANTS_API_KEY を設定しておく）

  PowerShell:
    $env:J_QUANTS_MIN_INTERVAL = "0.7"        # Standard はこの間隔が安全・高速
    .venv\\Scripts\\python.exe examples\\download_nikkei225.py

  bash / WSL / macOS:
    J_QUANTS_MIN_INTERVAL=0.7 .venv/bin/python examples/download_nikkei225.py

  期間を絞る / 別の by-date データセットを回す例:
    ... examples\\download_nikkei225.py --from 2024-01-01 --until 2024-12-31
    ... examples\\download_nikkei225.py --dataset weekly_margin

中断したら同じコマンドを再実行するだけで続きから再開します。市場データは data/ 配下
（gitignore 済）にのみ保存され、コミットされません（J-Quants 利用規約）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows コンソールでも日本語・進捗行を文字化けさせない
try:  # pragma: no cover
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
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

DEFAULT_DATASET = "options_225"


def _fmt_dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _cache_size_mb(cache_dir: Path) -> float:
    if not cache_dir.exists():
        return 0.0
    return sum(p.stat().st_size for p in cache_dir.glob("*.parquet")) / 1e6


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="日経225オプション（既定）など by-date データセットの単体ダウンローダ")
    ap.add_argument("--from", dest="frm", default="2016-06-13",
                    help="取得開始日 YYYY-MM-DD（既定 2016-06-13 = Standard下限）")
    ap.add_argument("--until", default=None,
                    help="取得終了日 YYYY-MM-DD（既定 本日）")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"カタログ上の by-date データセット名（既定 {DEFAULT_DATASET}）")
    args = ap.parse_args(argv)

    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が設定されていません。")
        return 1

    ds = DATASETS.get(args.dataset)
    if ds is None:
        print(f"ERROR: 未知のデータセット '{args.dataset}'。"
              f" 候補: {', '.join(DATASETS)}")
        return 2

    until = (pd.Timestamp(args.until) if args.until
             else pd.Timestamp.today().normalize())
    base = jq._CACHE
    manifest = Manifest(base / "manifest.json")

    fetched = manifest.fetched(ds.name) | scan_cache_dates(ds, base)
    cands = candidate_dates(ds.cadence, args.frm, until)
    todo = missing_dates(cands, fetched)

    print(f"対象 {ds.name} / 期間 {args.frm}〜{until.date()} / "
          f"候補 {len(cands)} 営業日 / 取得済 {len(cands) - len(todo)} / "
          f"今回取得 {len(todo)}")
    if not todo:
        print("差分なし。すべて取得済みです。")
        return 0
    if not get_env("J_QUANTS_MIN_INTERVAL"):
        print("ヒント: Standard では J_QUANTS_MIN_INTERVAL=0.7 を設定すると安全・高速です。")

    t0 = time.time()
    rows_total = 0
    ok = 0
    try:
        for i, dt in enumerate(todo, 1):
            try:
                df = ds.fetch_by_date(dt)          # キャッシュ書き込み込み
                n = len(df)
                rows_total += n
                ok += 1
                manifest.mark(ds.name, dt)
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (len(todo) - i) / rate if rate > 0 else 0.0
                print(f"\r[{i:>5}/{len(todo)}] {dt}  {n:>6,}行  "
                      f"累計 {rows_total:>10,}行  経過 {_fmt_dur(elapsed)}  "
                      f"残り≈{_fmt_dur(eta)}      ", end="", flush=True)
            except Exception as e:  # noqa: BLE001
                # 失敗日は残せるよう改行して表示（次回再実行で再取得対象になる）
                print(f"\r[{i:>5}/{len(todo)}] {dt}  [warn] {str(e)[:60]}"
                      f"{' ' * 20}")
            if i % 50 == 0:                        # 定期保存（再開に安全）
                manifest.save()
    except KeyboardInterrupt:
        manifest.save()
        print(f"\n中断しました。今回 {ok} 日 / {rows_total:,} 行を取得済み。"
              f" 同じコマンドで続きから再開できます。")
        return 130
    finally:
        manifest.save()

    cache_dir = base / ds.cache_subdir
    print(f"\n完了: {ok}/{len(todo)} 日取得, 累計 {rows_total:,} 行, "
          f"キャッシュ {_cache_size_mb(cache_dir):,.0f} MB ({cache_dir})")
    print("差分更新: 後日また同じコマンドを実行すれば未取得日だけ取得します。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
