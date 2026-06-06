"""二大候補 value＋PEAD の深掘り：相関→合成→判定器→IS/OOS。

value(book/market) と PEAD(予想改訂) は別系統。低相関なら等加重zコンポジットで
Sharpe 向上＝合成DSR>単独DSR を狙える。判定器(永続デフレートDSR・容量)で検証し、
保留OOS(2024-)でも持続するか確認する。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_value_pead_combo.py
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
from invest_system.equities.fundamentals import point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import AsOfView, CrossSectionalStrategy, judge_grid, write_html  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"


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


def _nanmean2(a, b):
    s = np.array([a.values, b.values])
    cnt = np.sum(~np.isnan(s), axis=0)
    tot = np.nansum(s, axis=0)
    return pd.DataFrame(np.where(cnt > 0, tot / np.where(cnt == 0, 1, cnt), np.nan),
                        index=a.index, columns=a.columns)


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
    combo = _nanmean2(value, pead)

    cs_corr = float(np.nanmean([value.loc[t].corr(pead.loc[t]) for t in rebal]))
    print(f"PITユニバース superset={len(superset)}, 月次={adj.shape[0]}本")
    print(f"value と PEAD の平均クロスセクション相関 = {cs_corr:+.2f}"
          f"（低いほど合成の分散効果が大）")

    grid = [CrossSectionalStrategy(value, 0.2, name="value"),
            CrossSectionalStrategy(pead, 0.2, name="pead"),
            CrossSectionalStrategy(combo, 0.2, name="value+pead")]
    with default_registry() as reg:
        v = judge_grid(grid, view, scope="value_pead_combo",
                       hypothesis="value と PEAD は別系統で低相関のため合成でSharpeが向上する",
                       economic_rationale="割安(リスク/行動)と予想改訂(情報拡散)は独立要因で分散効果を持つ",
                       registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_ = ls[ls.index < pd.Timestamp(OOS)]
        oos = ls[ls.index >= pd.Timestamp(OOS)]
        sr_is = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        sr_oos = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        (_, pre), (_, post) = pre_post_sharpe(ls, "2020-01-01")
        print(f"  {r.name:<12} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={sr_is:+.2f} OOS={sr_oos:+.2f} | 前2020={pre:+.2f} 後={post:+.2f}")

    best = max(v.results, key=lambda r: r.dsr)
    singles = [r.dsr for r in v.results if r.name in ("value", "pead")]
    comb = next((r.dsr for r in v.results if r.name == "value+pead"), float("nan"))
    print("\n--- 判定 ---")
    if comb >= max(singles) + 0.02:
        print(f"  ◎ 合成DSR {comb:.2f} > 単独最良 {max(singles):.2f}＝分散効果あり。"
              "厳密OOS継続の価値。")
    else:
        print(f"  ・合成DSR {comb:.2f} は単独最良 {max(singles):.2f} を超えず"
              f"（相関{cs_corr:+.2f}）。合成の上積みは限定的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
