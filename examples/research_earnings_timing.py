"""決算発表予定ベースの戦略を判定器へ：
  ① run-up プレミアム（翌月に発表見込みの銘柄をロング＝発表前ドリフト）
  ② 発表回避オーバーレイ on value（発表見込み銘柄を建玉から除外→改善するか）
  ③ 発表回避オーバーレイ on PEAD

発表月は /fins/summary の DiscDate から予測（events.expected_announcement_month, PIT）。
月次・PITユニバース・セクター中立・永続デフレートDSR・容量・HTML。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_earnings_timing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import events  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.fundamentals import point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.research import AsOfView, CrossSectionalStrategy, judge_grid, write_html  # noqa: E402
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


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=300, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    adv = turn.reindex(columns=superset)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})
    fund = fetch_fundamentals(superset)
    print(f"PITユニバース superset={len(superset)}, 月次={adj.shape[0]}本")

    # 翌月の発表見込みマスク（PIT）
    amask = events.expected_announcement_month(fund, rebal).reindex(
        index=rebal, columns=superset).fillna(False)
    print(f"発表見込み（月平均銘柄数）= {amask.sum(axis=1).mean():.0f}")

    # ファクター
    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    value = apply_universe_mask(value_quality_size_factors(pit, raw, adj)["book_to_market"],
                                umask)
    pead = apply_universe_mask(point_in_time(events.forecast_revision(fund), rebal,
                                             ["fcst_revision"], date_col="DiscDate",
                                             lag_days=1)["fcst_revision"]
                               .reindex(columns=superset), umask)

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(f, sector))

    with default_registry() as reg:
        # ① run-up（翌月発表見込みをロング）
        runup = zN(apply_universe_mask(amask.astype(float), umask))
        v1 = judge_grid([CrossSectionalStrategy(runup, quantile=q, name=f"runup(q={q})")
                         for q in (0.1, 0.2, 0.3)], view, scope="earnings_runup",
                        hypothesis="決算発表が近い銘柄は発表前に超過リターンを出す（announcement premium）",
                        economic_rationale="発表前の注目・リスクプレミアム上昇による run-up アノマリー",
                        registry=reg, costs_bps=15.0, adv=adv)
        print("\n########## ① run-up プレミアム ##########")
        print(v1.report_md)
        print("HTML:", write_html(v1, f"data/reports/{v1.scope}.html"))

        # ②③ 発表回避オーバーレイ（base vs base.where(~発表見込み)）
        for label, fac in (("value", value), ("pead", pead)):
            base = zN(fac)
            ovl = zN(fac.where(~amask))            # 発表見込み銘柄は建玉から除外
            grid = [CrossSectionalStrategy(base, 0.2, name=f"{label}_base"),
                    CrossSectionalStrategy(ovl, 0.2, name=f"{label}_avoid_earnings")]
            v = judge_grid(grid, view, scope=f"{label}_earn_overlay",
                           hypothesis=f"{label}戦略は決算をまたぐ建玉を避けるとリスク調整後が改善する",
                           economic_rationale="決算発表の個別ジャンプ・リスクを外すことでDD/分散が縮小するとの仮説",
                           registry=reg, costs_bps=15.0, adv=adv)
            print(f"\n########## オーバーレイ on {label}（base vs 発表回避）##########")
            print(v.report_md)
            print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
