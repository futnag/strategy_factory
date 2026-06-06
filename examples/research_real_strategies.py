"""実戦略を判定器にかける：(A) フロー→TOPIXタイミング, (B) value クロスセクション。

検証ファクトリ（research）の総合デモ。実データで2つの実戦略を Strategy 化し、判定器
（事前登録＋パラメータ格子デフレートDSR＋PSR/minTRL/サブ期間）で厳格に裁く。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_real_strategies.py
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
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.panel import (  # noqa: E402
    assemble_panel, fetch_month_end_snapshots,
)
from invest_system.equities.fundamentals import point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, SignalTimingStrategy, judge_grid,
)
from invest_system.validation.registry import default_registry  # noqa: E402

START, END = "2016-07", "2026-05"


def fetch_fundamentals(codes):
    fr = []
    for c in codes:
        try:
            st = jq.fetch_statements(code=c)
            if not st.empty:
                fr.append(st)
        except Exception:  # noqa: BLE001
            pass
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def strategy_a_flow_timing(reg):
    print("\n########## (A) フロー→TOPIX タイミング（週次）##########")
    inv = flows.load_investor_types()
    sub = inv[inv["Section"] == "TokyoNagoya"].dropna(subset=["PubDate"])
    sub = sub.set_index("PubDate").sort_index()
    sig = (sub["FrgnBal"] / sub["FrgnTot"].replace(0, np.nan)).dropna()
    sig = sig[~sig.index.duplicated(keep="last")]      # index=公表日（利用可能日）
    topix = jq.fetch_index_bars(code="0000")
    tc = topix.dropna(subset=["Date"]).set_index("Date")["C"].sort_index()
    tw = tc.resample("W-FRI").last().dropna().to_frame("0000")
    view = AsOfView({"close": tw})
    grid = [SignalTimingStrategy(sig, "0000", threshold=th, side=s)
            for th in (0.0, 0.05) for s in (1, -1)]
    print(f"TOPIX週次 {len(tw)}本, フロー {len(sig)}週, 格子 {len(grid)}通り")
    v = judge_grid(grid, view, scope="flow_topix_timing",
                   hypothesis="海外勢の純買いが優勢な局面で翌週TOPIXは上昇しやすい",
                   economic_rationale="海外投資家フローは日本株の主要な限界需要であり需給を主導するため",
                   registry=reg, costs_bps=10.0)
    print(v.report_md)


def strategy_b_value(reg):
    print("\n########## (B) value クロスセクション（月次・PITユニバース）##########")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    mask = point_in_time_universe(turn_c, top_n=300, lookback=12, min_obs=6)
    superset = universe_members(mask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    mask = mask.reindex(columns=superset).fillna(False)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    fund = fetch_fundamentals(superset)
    pit = point_in_time(fund, adj.index, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    ff = value_quality_size_factors(pit, raw, adj)
    bm = cross_sectional_zscore(sector_neutralize(
        apply_universe_mask(ff["book_to_market"], mask), sector))
    view = AsOfView({"close": adj})
    grid = [CrossSectionalStrategy(bm, quantile=q, name=f"value_bm(q={q})")
            for q in (0.1, 0.2, 0.3)]
    print(f"月次 {adj.shape[0]}本 × superset {adj.shape[1]}銘柄, 格子 {len(grid)}通り")
    v = judge_grid(grid, view, scope="value_xs",
                   hypothesis="割安（高 book/market）銘柄は割高銘柄を中長期で上回る",
                   economic_rationale="リスク/行動バイアスに基づくバリュー・プレミアムの持続を仮定",
                   registry=reg, costs_bps=15.0)
    print(v.report_md)


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    with default_registry() as reg:           # 永続グローバル・レジストリ（累積）
        strategy_a_flow_timing(reg)
        strategy_b_value(reg)
        print("\n=== 永続レジストリ（scope別 累計試行 K）===")
        for scope, k, srv in reg.list_scopes():
            print(f"  {scope:<22} K={k}  V[SR]={srv:.4f}")
    print("\n＝判定は永続レジストリに累積。再実行は冪等（Kを水増ししない）、")
    print("  新パラメータは K を増やし全戦略のDSRを下げる＝真のグローバル・デフレート。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
