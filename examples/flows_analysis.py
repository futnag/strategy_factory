"""投資部門別フローのミニ分析：海外勢ネットフローとTOPIXの関係（PIT配慮）。

「海外勢が買った週の翌週、TOPIXは上がるか？」を素直に見る。フロー(週末EnDate)は
数営業日後に公表されるため、翌週リターンの予測に使うのが point-in-time 安全。
同時相関（その週の海外買い vs その週のTOPIX）も併記し、追随か先行かを区別する。

実行: .venv\\Scripts\\python.exe examples\\flows_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import flows  # noqa: E402


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    inv = flows.load_investor_types()
    if inv.empty:
        print("投資部門別データ未取得。examples/fetch_indices_flows.py を先に実行。")
        return 1
    print(f"投資部門別: {len(inv):,} 行, 区分={sorted(inv['Section'].dropna().unique())}")

    topix = jq.fetch_index_bars(code="0000")          # キャッシュ済み
    tclose = topix.dropna(subset=["Date"]).set_index("Date")["C"].sort_index()

    for section in ("TokyoNagoya", "TSEPrime"):
        f = flows.net_flow_intensity(inv, investor="foreign", section=section).dropna()
        if f.size < 20:
            continue
        tw = tclose.reindex(f.index, method="ffill")
        ret_same = tw / tw.shift(1) - 1.0             # その週のTOPIX
        ret_next = tw.shift(-1) / tw - 1.0            # 翌週のTOPIX（予測対象）
        d = pd.DataFrame({"flow": f, "same": ret_same, "next": ret_next}).dropna()
        if len(d) < 20:
            continue
        c_same = d["flow"].corr(d["same"])
        c_next = d["flow"].corr(d["next"])
        # 海外ネット買いが正の週の翌週リターン平均 vs 負の週
        hi = d[d["flow"] > 0]["next"].mean()
        lo = d[d["flow"] <= 0]["next"].mean()
        print(f"\n[{section}]  n={len(d)} 週")
        print(f"  海外ネットフロー vs 同週TOPIX  corr={c_same:+.2f}（追随/同時）")
        print(f"  海外ネットフロー vs 翌週TOPIX  corr={c_next:+.2f}（予測力）")
        print(f"  翌週平均: 買い越し週後 {hi:+.3%} / 売り越し週後 {lo:+.3%}  "
              f"差 {hi - lo:+.3%}")

    print("\n※ これは需給データが『分析可能な形』で揃ったことの実証。")
    print("  本格判定は research の判定器（フロー条件→TOPIX建玉の戦略）で行える。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
