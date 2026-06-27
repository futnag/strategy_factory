"""月次ライブ監視：因果エッジ・不安定業種数・adverse フラグ（オプション c）。

value+PEAD 一次戦略のサイズ決定には使わない（docs/40: ハイブリッド FAIL）。
downside リスクの早期警戒用ダッシュボード。

実行:
  .venv\\Scripts\\python.exe examples\\causal_sector_monitor.py
  .venv\\Scripts\\python.exe examples\\causal_sector_monitor.py --refresh
  .venv\\Scripts\\python.exe examples\\causal_sector_monitor.py --asof 2026-05

出力:
  data/reports/causal_sector/monitor_latest.{json,md}
  data/reports/causal_sector/monitor_YYYY-MM.{json,md}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invest_system.config import get_env  # noqa: E402
from invest_system.research.causal_sector.monitor import (  # noqa: E402
    build_monitor_snapshot, write_monitor_report,
)


def main() -> int:
    p = argparse.ArgumentParser(description="因果レジーム月次ライブ監視")
    p.add_argument("--asof", default=None, help="対象月 YYYY-MM（省略=最新）")
    p.add_argument("--refresh", action="store_true", help="全33業種キャッシュを再構築")
    p.add_argument("--no-archive", action="store_true", help="月次アーカイブを書かない")
    args = p.parse_args()

    if args.refresh and not get_env("J_QUANTS_API_KEY"):
        print("ERROR: --refresh には J_QUANTS_API_KEY が必要です")
        return 1

    asof = pd.Timestamp(args.asof) if args.asof else None
    if asof is not None:
        asof = asof + pd.offsets.MonthEnd(0)

    print("=" * 72)
    print("因果レジーム月次監視（監視専用・売買トリガーではない）")
    print("=" * 72)

    snap = build_monitor_snapshot(
        asof=asof, refresh_cache=args.refresh, verbose=True,
    )
    adv = snap.adverse
    a = snap.aggregate

    print(f"\n対象月: {snap.asof:%Y-%m}")
    print(f"ステータス: {snap.status}")
    print(f"  causal_edge      = {a.get('causal_edge')}")
    print(f"  edge_financial   = {a.get('edge_financial')}")
    print(f"  n_unstable       = {a.get('n_sectors_unstable')}")
    print(f"  edge_heavy_ind   = {a.get('edge_heavy_industry')}")
    print(f"  edge_it_comm     = {a.get('edge_it_comm')}")
    print(f"\nadverse 3条件: edge<0={adv.causal_edge_neg}  "
          f"fin<0={adv.edge_financial_neg}  unstable>={adv.unstable_thresh}={adv.n_unstable_high}")
    print(f"  → adverse={adv.all_met}  (hybrid 0.5x 参考: {adv.all_met})")

    if not snap.sector_edges.empty:
        neg = snap.sector_edges[snap.sector_edges["causal_edge"] < 0]
        print(f"\nエッジ負の業種 ({len(neg)}):")
        for _, r in neg.head(8).iterrows():
            print(f"  {r['s33']} {r['sector_name']}: {r['causal_edge']:.4f}")
        if len(neg) > 8:
            print(f"  … 他 {len(neg) - 8} 業種")

    json_p, md_p = write_monitor_report(snap, archive=not args.no_archive)
    print(f"\nJSON: {json_p}")
    print(f"MD:   {md_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())