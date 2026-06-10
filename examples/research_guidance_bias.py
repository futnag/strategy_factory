"""仮説検証②：期初ガイダンス保守バイアス（上方修正の「先回り」）【柱C 拡張】。

経済的根拠：日本企業は期初予想を系統的に低く出し期中に上方修正する行動が学術・実務の
双方で確認されている（保守的開示の常習性）。既存 PEAD（§7）は改訂が**起きた後**の
ドリフトを取る。本研究はその**前**＝「常習的ビーター」を期初に仕込み、予見可能な
上方修正（中間決算 10-11 月に集中）を先回りで収穫できるかを裁く。反対側＝期初予想を
額面どおり織り込む参加者。常習性が既に株価に織り込まれていれば期待値ゼロ＝それを
判定器が決める。

シグナル（PIT）：`events.guidance_conservatism`＝銘柄ごとに
  surprise(年度) = (実績EPS − 前年FY行の来期予想 NxFEPS) / |NxFEPS|（年度厳密整合）
の直近3年平均。FY 本決算開示日に更新・as-of で月次へ。

事前登録（K=2）：cons_lt（ロングティルト・主形）/ cons_ls（ロングショート＝楽観常習
未達銘柄ショートの対称仮説も同時に裁く）。現実性は §6.9-6.11 と同一（月次・15bps・容量）。

実行: .venv\\Scripts\\python.exe examples\\research_guidance_bias.py
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
SCOPE = "guidance_bias"


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

    cons_l = events.guidance_conservatism(fund, n_years=3, min_years=2)
    cons = zN(point_in_time(cons_l, rebal, ["cons_score"], date_col="DiscDate",
                            lag_days=1)["cons_score"].reindex(columns=superset))
    cover = cons.notna().mean(axis=1)
    print(f"=== 期初ガイダンス保守バイアス（{START}〜{END}・月末{len(rebal)}回・"
          f"scope={SCOPE}）===")
    print(f"スコア被覆率（ユニバース superset 比）: 平均 {cover.mean():.0%} / "
          f"2019以降 {cover[cover.index >= '2019-01'].mean():.0%}"
          f"（3年分の FY 蓄積が要るため序盤は薄い）")

    strategies = [
        CrossSectionalStrategy(cons, 0.2, name="cons_lt", long_only=True),
        CrossSectionalStrategy(cons, 0.2, name="cons_ls"),
    ]
    with default_registry() as reg:
        v = judge_grid(
            strategies, view, scope=SCOPE,
            hypothesis=("期初予想を常習的に低く出す銘柄（保守バイアス）は、予見可能な上方修正を"
                        "通じて超過リターンを生むか＝PEAD の先回りが可能か"),
            economic_rationale=("保守的期初予想→期中上方修正は日本の開示行動として学術・実務で"
                                "確認済み。常習性は銘柄属性として持続的で、期初に観測可能。"
                                "反対側は予想を額面で織り込む参加者。織り込み済みなら期待値ゼロ"
                                "＝判定器が裁く。"),
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜）・暦月別の平均ネット（収穫期＝10-11月仮説の診断）---")
    for r in v.results:
        s = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        (_, pre), (_, post) = pre_post_sharpe(s, "2020-01-01")
        print(f"  {r.name:<10} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={_sr(s, hi=OOS):+.2f} OOS={_sr(s, lo=OOS):+.2f} | "
              f"前/後2020={pre:+.2f}/{post:+.2f}")
        bym = s.groupby(s.index.month).mean() * 1e4
        top = bym.sort_values(ascending=False).head(3)
        print(f"      暦月別平均(bp): " +
              " ".join(f"{m}月:{x:+.0f}" for m, x in bym.items()) +
              f"  ← 上位 {list(top.index)} 月")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
