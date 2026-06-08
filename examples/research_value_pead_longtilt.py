"""仮説検証：ロングティルトPEAD（上方修正ロング×市場ヘッジ）＋value 合成。

PEAD診断＝「エッジは上方修正ロング側、下方修正ショートが value と対立し OOS で崩れた」。
→ PEAD をロングオンリー（市場ヘッジ）にし、value(ロングショート)とウェイト合成すれば
OOS が改善するはず、を判定器＋IS/OOS で検証する。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_value_pead_longtilt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
from invest_system.equities.fundamentals import load_fundamentals, point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, CompositeStrategy, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"


def fetch_fundamentals(codes):
    # fins_summary/ 全件 by-date ミラーから長形式取得（旧 by-code statements/ も併合・重複除去）
    return load_fundamentals(codes)


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

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, umask),
                                                        sector))
    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    value = zN(value_quality_size_factors(pit, raw, adj)["book_to_market"])
    pead = zN(point_in_time(events.forecast_revision(fund), rebal, ["fcst_revision"],
                            date_col="DiscDate", lag_days=1)["fcst_revision"]
              .reindex(columns=superset))

    value_ls = CrossSectionalStrategy(value, 0.2, name="value")
    pead_lt = CrossSectionalStrategy(pead, 0.2, name="pead_longtilt", long_only=True)
    combo = CompositeStrategy([value_ls, pead_lt], [0.5, 0.5], name="value+pead_lt")

    with default_registry() as reg:
        v = judge_grid([value_ls, pead_lt, combo], view, scope="value_pead_longtilt",
                       hypothesis="ロングティルトPEAD（上方修正ロング）はvalueと対立せず合成のOOSを改善する",
                       economic_rationale="OOS失敗は下方修正ショートがvalueロングと衝突した為。ショート脚を外せば持続",
                       registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_ = ls[ls.index < pd.Timestamp(OOS)]
        oos = ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        (_, pre), (_, post) = pre_post_sharpe(ls, "2020-01-01")
        print(f"  {r.name:<14} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={si:+.2f} OOS={so:+.2f} | 前2020={pre:+.2f} 後={post:+.2f}")

    print("\n--- 判定（旧合成 value+pead[LS] は OOS +0.06 だった）---")
    c = next((r for r in v.results if r.name == "value+pead_lt"), None)
    val = next((r for r in v.results if r.name == "value"), None)
    if c and val:
        clo = v.series["value+pead_lt"].dropna()
        co_oos = sharpe_ratio(clo[clo.index >= pd.Timestamp(OOS)]) * np.sqrt(12)
        vlo = v.series["value"].dropna()
        v_oos = sharpe_ratio(vlo[vlo.index >= pd.Timestamp(OOS)]) * np.sqrt(12)
        print(f"  合成(longtilt) 全DSR={c.dsr:.2f}, OOS SR={co_oos:+.2f} / "
              f"value単独 OOS SR={v_oos:+.2f}")
        if co_oos >= v_oos - 0.05 and c.dsr >= val.dsr:
            print("  ◎ ロングティルト化でOOSの失速を解消＋全DSRも単独value以上＝仮説成立。")
        elif co_oos > 0.06 + 0.10:
            print("  △ OOSは旧合成(+0.06)より改善したが、value単独OOSには届かず。")
        else:
            print("  ・期待した改善は限定的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
