"""全 TOB 案件の本体を取得し、当初/最終価格・バンプ・成否を埋めた案件テーブルを作る。

build_deal_chains（構造）＋ enrich_deal（本体）を全案件に適用し
data/edinet/tob_deals.parquet に保存。冪等・再開可：既に本体取得済み(body_ok)の
deal_id はスキップし、書類本体は fetch_document がキャッシュ済みのため再実行も安価。
途中保存（既定 25 件毎）で中断しても再開できる。

usage:
  python examples/edinet_tob_backfill.py
  python examples/edinet_tob_backfill.py --limit 50      # 部分実行（動作確認）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import edinet as ed          # noqa: E402
from invest_system.equities import tob_events as te          # noqa: E402

OUT = ed._CACHE / "tob_deals.parquet"

# enrich_deal が返す列（再開時はこれだけ復元し、構造列との重複マージを避ける）
ENRICH_COLS = ["deal_id", "initial_price", "final_price", "purchase_ratio_pct",
               "period_start", "period_end_initial", "period_end_final",
               "n_bumps", "n_extensions", "result", "withdrawn", "body_ok"]


def load_listing() -> pd.DataFrame:
    frames = []
    for p in sorted((ed._CACHE / "list").glob("*.parquet")):
        df = pd.read_parquet(p)
        if "_empty" not in df.columns:
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 案件のみ（0=全件）")
    ap.add_argument("--save-every", type=int, default=25)
    args = ap.parse_args()

    listing = load_listing()
    deals = te.build_deal_chains(listing)
    if args.limit:
        deals = deals.head(args.limit)
    print(f"案件 {len(deals)} 件（第三者TOB）")

    enriched: dict[str, dict] = {}
    done: set[str] = set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        for _, r in prev.iterrows():
            enriched[r["deal_id"]] = {c: r[c] for c in ENRICH_COLS if c in prev.columns}
            if r.get("body_ok"):
                done.add(r["deal_id"])
        print(f"既存 {len(enriched)} 件（本体取得済み {len(done)}）→ 残りを取得")

    todo = [d for d in deals["deal_id"] if d not in done]
    print(f"取得対象 {len(todo)} 件")

    def flush():
        rows = [enriched[d] for d in deals["deal_id"] if d in enriched]
        merged = deals.merge(pd.DataFrame(rows), on="deal_id", how="left",
                             suffixes=("", "_e"))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(OUT)

    for i, did in enumerate(todo, 1):
        row = deals[deals["deal_id"] == did].iloc[0]
        try:
            enriched[did] = te.enrich_deal(row, listing)
        except Exception as e:                          # noqa: BLE001
            print(f"  [warn] {did}: {str(e)[:80]}")
            enriched[did] = {"deal_id": did, "body_ok": False}
        if i % args.save_every == 0:
            flush()
            print(f"  {i}/{len(todo)} … 途中保存")
    flush()

    final = pd.read_parquet(OUT)
    ok = int(final["body_ok"].fillna(False).sum()) if "body_ok" in final else 0
    res = final["result"].value_counts(dropna=True).to_dict() if "result" in final else {}
    bumped = int((final.get("n_bumps", pd.Series(dtype=float)).fillna(0) > 0).sum())
    print(f"\n完了：{len(final)} 件 / 本体OK {ok} / 成否 {res} / バンプ有 {bumped} 件")
    print(f"保存先: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
