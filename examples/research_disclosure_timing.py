"""仮説検証③：決算発表タイミング（前倒し/遅延）シグナル【柱C 拡張・開示行動】。

経済的根拠：開示行動の古典仮説 "good news early, bad news late"。悪い決算ほど社内
確定・監査・文言調整に時間がかかり発表が遅れる（経営者の開示インセンティブ）。発表の
**前年同期比の遅延**は、内容を見る前から決算の質を予告する無料のシグナルであるはず。
反対側＝発表日程の変化を見ない参加者。日本は発表日慣行が安定的（同一銘柄は毎年ほぼ
同じ営業日に発表）なので、ズレの情報量が相対的に高いという仮説。

シグナル（PIT）：`events.announcement_delay`＝同一 CurPerType（1Q/2Q/3Q/FY）の前年
開示日＋365日 を基準とした遅延日数（正＝遅延）。決算短信のみ対象（臨時開示は除外）。
ファクター＝**−delay**（前倒しロング/遅延ショート方向）。開示日に更新・as-of 月次。

事前登録（K=2）：timing_lt（ロングティルト）/ timing_ls（ロングショート）。
現実性は §6.9-6.11 と同一（月次・15bps・容量）。

実行: .venv\\Scripts\\python.exe examples\\research_disclosure_timing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

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
    cross_sectional_zscore, sector_neutralize,
)
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
SCOPE = "disclosure_timing"


def _sr(x: pd.Series, lo=None, hi=None) -> float:
    r = x.dropna()
    if lo is not None:
        r = r[r.index >= pd.Timestamp(lo)]
    if hi is not None:
        r = r[r.index < pd.Timestamp(hi)]
    return float(sharpe_ratio(r) * np.sqrt(12)) if r.size >= 8 else float("nan")


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
    adj = adj.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    adv = turn.reindex(columns=superset)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})
    fund = load_fundamentals(superset)

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, umask),
                                                        sector))

    delay_l = events.announcement_delay(fund)
    print(f"=== 決算発表タイミング（{START}〜{END}・月末{len(rebal)}回・scope={SCOPE}）===")
    dd = delay_l["delay_days"]
    print(f"遅延イベント: {len(delay_l)}件 / 分布 p10={dd.quantile(.1):+.0f} "
          f"中央値={dd.median():+.0f} p90={dd.quantile(.9):+.0f} 日")
    timing = zN(-point_in_time(delay_l, rebal, ["delay_days"], date_col="DiscDate",
                               lag_days=1)["delay_days"].reindex(columns=superset))

    strategies = [
        CrossSectionalStrategy(timing, 0.2, name="timing_lt", long_only=True),
        CrossSectionalStrategy(timing, 0.2, name="timing_ls"),
    ]
    with default_registry() as reg:
        v = judge_grid(
            strategies, view, scope=SCOPE,
            hypothesis=("決算発表の前年同期比の遅延は悪材料を、前倒しは好材料を予告する"
                        "（good news early, bad news late）＝発表タイミング自体が"
                        "クロスセクションの予測力を持つか"),
            economic_rationale=("悪い決算ほど確定・監査・文言調整が長引く開示行動は国際的に"
                                "頑健な実証。日本は発表日慣行が安定しズレの情報量が高いはず。"
                                "反対側は日程変化を見ない参加者。"),
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        s = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        (_, pre), (_, post) = pre_post_sharpe(s, "2020-01-01")
        print(f"  {r.name:<10} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={_sr(s, hi=OOS):+.2f} OOS={_sr(s, lo=OOS):+.2f} | "
              f"前/後2020={pre:+.2f}/{post:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
