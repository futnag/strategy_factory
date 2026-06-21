"""EDINET 有報の長形式フル材化 — 手元ターミナルで進捗付き・再開可で一度回す薄いランナー。

GKX Phase 2 のデータ層。`examples/download_edinet.py` でローカルにバックフィルした有報
type=5 zip（約 44,000 件）を 1 件ずつパースし、正準フィールドの長形式
`data/edinet/fundamentals_long.parquet` を材化する（`build_edinet_long` を呼ぶだけ）。

なぜ独立ランナーか：材化を診断（`diag_edinet_fundamentals.py`）や as-of パネル構築から
切り離し、初回フル材化だけを進捗付きで一度回せるようにするため。Claude を介さない自己完結
スクリプト（=> プラン使用量を消費しない）。材化はファクタ計算・スキーマ・レジストリ（K）に
一切触れない＝データ基盤の堅牢化のみ。

初回フル材化は約 44k 件で **数十分〜1.5h 規模**。checkpoint_every（既定 1000）件ごとに
本体 parquet を原子的に上書き保存するので、**Ctrl-C 中断 → 同じコマンド再実行で続きから**
再開できる（既処理 docID は再パースしない）。材化中は **PC をスリープさせない**こと。

使い方（先に download_edinet.py で zip をバックフィル）:
  PowerShell:
    .venv\\Scripts\\python.exe examples\\build_edinet_long.py
    .venv\\Scripts\\python.exe examples\\build_edinet_long.py --checkpoint-every 2000
    .venv\\Scripts\\python.exe examples\\build_edinet_long.py --rebuild        # 全再構築
  bash / WSL / macOS:
    .venv/bin/python examples/build_edinet_long.py
中断したら同じコマンドを再実行するだけで続きから再開します。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # Windows コンソールでも日本語・進捗行を文字化けさせない
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001  # pragma: no cover
    pass

import pandas as pd  # noqa: E402

from invest_system.equities import edinet_fundamentals as ef  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="EDINET 有報 長形式フル材化（進捗付き・中断再開可）")
    ap.add_argument("--rebuild", action="store_true",
                    help="既存キャッシュを無視して全再構築（既定は増分）")
    ap.add_argument("--checkpoint-every", type=int, default=1000,
                    help="N 件ごとに本体 parquet を原子的保存（既定 1000）")
    args = ap.parse_args(argv)

    print(f"=== EDINET 長形式材化  checkpoint_every={args.checkpoint_every}"
          f"{'  rebuild' if args.rebuild else ''} ===")
    long = ef.build_edinet_long(rebuild=args.rebuild, verbose=True,
                                checkpoint_every=args.checkpoint_every)
    if long.empty:
        print("EDINET 長形式が空。先に examples/download_edinet.py を実行してください。")
        return 1
    pe = pd.to_datetime(long["period_end"], errors="coerce")
    rng = (f"{pe.min():%Y-%m} 〜 {pe.max():%Y-%m}" if pe.notna().any() else "—")
    print(f"対象開示: {len(long)} 件 / 銘柄 {long['Code'].nunique()} / 期末 {rng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
