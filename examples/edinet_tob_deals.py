"""EDINET から TOB 案件テーブルを構築するデモ（C1 リスクアーブの一次データ）。

(A) キャッシュ済み一覧（data/edinet/list）から全案件の構造テーブルを構築（ネットワーク
    不要・即時）。(B) --enrich N で直近 N 案件だけ本体 CSV を取得し、当初/最終価格・
    バンプ・延長・成否を抽出（PIT：当初は 240、結果は 250/270）。

usage:
  python examples/edinet_tob_deals.py
  python examples/edinet_tob_deals.py --enrich 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import edinet as ed          # noqa: E402
from invest_system.equities import tob_events as te          # noqa: E402


def load_listing() -> pd.DataFrame:
    frames = []
    for p in sorted((ed._CACHE / "list").glob("*.parquet")):
        df = pd.read_parquet(p)
        if "_empty" not in df.columns:
            frames.append(df)
    if not frames:
        raise SystemExit("EDINET 一覧キャッシュが空です。先に edinet_update.py を実行。")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrich", type=int, default=0,
                    help="直近 N 案件の本体を取得し価格/成否を抽出")
    args = ap.parse_args()

    deals = te.build_deal_chains(load_listing())
    deals["year"] = pd.to_datetime(deals["announce_dt"]).dt.year
    print(f"=== TOB 案件テーブル（第三者TOB・府令040）: {len(deals)} 件 ===")

    print("\n年 | 件数 | 競合 | 報告(270)あり | 撤回 | target_sec解決")
    for y in sorted(deals["year"].dropna().unique()):
        g = deals[deals["year"] == y]
        print(f"  {int(y)}: {len(g):4d}  競合{int(g['competing'].sum()):3d}  "
              f"報告{int(g['has_report'].sum()):4d}  撤回{int(g['has_withdrawal'].sum()):2d}  "
              f"target_sec {g['target_sec'].notna().mean():.0%}")

    print(f"\n訂正(250)を持つ案件: {(deals['n_amend'] > 0).sum()} / "
          f"競合案件: {int(deals['competing'].sum())} / "
          f"撤回あり: {int(deals['has_withdrawal'].sum())}")

    if args.enrich > 0:
        listing = load_listing()
        sample = (deals[deals["has_report"]]
                  .sort_values("announce_dt").tail(args.enrich))
        print(f"\n=== 本体抽出（直近 {len(sample)} 案件・報告済み） ===")
        for _, d in sample.iterrows():
            e = te.enrich_deal(d, listing)
            arrow = (f"{e['initial_price']}→{e['final_price']}"
                     if e["initial_price"] != e["final_price"]
                     else f"{e['initial_price']}")
            print(f"  {d['deal_id']} 対象{d['target_edinet']}({d['target_sec']}) "
                  f"買収者{str(d['acquirer_name'])[:16]}")
            print(f"    価格 {arrow}円  バンプ{e['n_bumps']} 延長{e['n_extensions']}  "
                  f"期間 {e['period_start']}〜{e['period_end_final']}  成否={e['result']}  "
                  f"買付割合={e['purchase_ratio_pct']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
