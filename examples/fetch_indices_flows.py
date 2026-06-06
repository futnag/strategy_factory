"""指数四本値・投資部門別フローの一括取得（コア・プリフェッチ）。

投資部門別は from/to で全履歴を1回（週次・軽量）。指数は全コードを銘柄別に全履歴
取得（各1回で全期間）。いずれもローカル Parquet にキャッシュ（gitignore済）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402

START = get_env("J_EQ_START_D", "2016-06-13") or "2016-06-13"
END = get_env("J_EQ_END_D", "2026-06-06") or "2026-06-06"


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    print(f"=== コア・プリフェッチ（指数・投資部門別）{START}〜{END} ===")

    print("投資部門別売買状況（全履歴・1回）…")
    inv = jq.fetch_investor_types(frm=START, to=END)
    secs = sorted(inv["Section"].dropna().unique()) if "Section" in inv.columns else []
    print(f"  {len(inv):,} 行, 区分={secs}")

    print("指数コード一覧を取得…")
    snap = jq.fetch_index_bars(date="20260529")
    codes = sorted(snap["Code"].dropna().astype(str).unique()) \
        if "Code" in snap.columns else []
    print(f"  指数 {len(codes)} 種を銘柄別に全履歴取得…")
    rows = 0
    for i, c in enumerate(codes, 1):
        try:
            df = jq.fetch_index_bars(code=c, frm=START, to=END)
            rows += len(df)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {c}: {str(e)[:60]}")
        if i % 25 == 0:
            print(f"  {i}/{len(codes)}")
    print(f"  指数 累計 {rows:,} 行")
    print("\n完了。投資部門別＋指数をローカルにキャッシュしました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
