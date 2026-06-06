"""新戦略種を判定器へ：①PEAD（会社予想の改訂）②信用/空売り（空売り残高）。

いずれもクロスセクション・ロングショートとして判定器にかける（PITユニバース・
セクター中立・永続デフレートDSR・容量・HTML出力）。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_pead_shortint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import events, margin  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.fundamentals import point_in_time  # noqa: E402
from invest_system.equities.factors import cross_sectional_zscore, sector_neutralize  # noqa: E402
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
    adj, turn = assemble_panel(snaps, "AdjC"), assemble_panel(snaps, "Va")
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    mask = point_in_time_universe(turn_c, top_n=300, lookback=12, min_obs=6)
    superset = universe_members(mask)
    adj = adj.reindex(columns=superset)
    mask = mask.reindex(columns=superset).fillna(False)
    adv = turn.reindex(columns=superset)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})
    print(f"PITユニバース superset={len(superset)}, 月次={adj.shape[0]}本")

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, mask),
                                                        sector))

    fund = fetch_fundamentals(superset)

    with default_registry() as reg:
        # ① PEAD（会社予想の改訂）
        rev = point_in_time(events.forecast_revision(fund), rebal, ["fcst_revision"],
                            date_col="DiscDate", lag_days=1)["fcst_revision"]
        rev_z = zN(rev.reindex(columns=superset))
        grid = [CrossSectionalStrategy(rev_z, quantile=q, name=f"pead_rev(q={q})")
                for q in (0.1, 0.2, 0.3)]
        v1 = judge_grid(grid, view, scope="pead_revision",
                        hypothesis="会社予想を上方修正した銘柄は将来アウトパフォーム（PEAD）",
                        economic_rationale="情報の漸進的織り込みとガイダンス改訂アノマリーの持続",
                        registry=reg, costs_bps=15.0, adv=adv)
        print("\n########## ① PEAD（予想改訂）##########")
        print(v1.report_md)
        print("HTML:", write_html(v1, f"data/reports/{v1.scope}.html"))

        # ② 信用/空売り（空売り残高、両符号を試行）
        si = point_in_time(margin.short_interest(margin.load_short_positions()),
                           rebal, ["short_interest"], date_col="Date",
                           lag_days=4)["short_interest"].reindex(columns=superset).fillna(0.0)
        si_z = zN(si)
        grid2 = ([CrossSectionalStrategy(si_z, quantile=q, name=f"shortint_hi(q={q})")
                  for q in (0.1, 0.2, 0.3)]
                 + [CrossSectionalStrategy(-si_z, quantile=q, name=f"shortint_lo(q={q})")
                    for q in (0.1, 0.2, 0.3)])
        v2 = judge_grid(grid2, view, scope="short_interest_xs",
                        hypothesis="空売り残高の多寡が将来リターンを予測（需給/スクイーズ）",
                        economic_rationale="空売り需給はミスプライスや将来の買い戻し圧力を示し得る",
                        registry=reg, costs_bps=15.0, adv=adv)
        print("\n########## ② 信用/空売り（空売り残高）##########")
        print(v2.report_md)
        print("HTML:", write_html(v2, f"data/reports/{v2.scope}.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
