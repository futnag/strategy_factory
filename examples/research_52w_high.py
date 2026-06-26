r"""仮説検証 ⑧：52週高値プロキシミティ・モメンタム（George-Hwang 2004）。

「現在値 / 過去52週高値」が1に近い銘柄（高値圏）は将来アウトパフォームする——アンカリング/
アンダーリアクション仮説。George-Hwang は **52週高値が伝統的モメンタムを subsume する**と報告。
日本のモメンタムは局面依存（既試）だが、高値プロキシミティ版は未テスト＝新規 scope。

シグナル: proximity = AdjC / (過去12ヶ月 AdjC 最大)（∈(0,1]、高い＝高値圏＝ロング）。PIT・先読みなし。
セクター中立・winsorize・z。分位 q∈{0.1,0.2,0.3}（K=3）。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\Scripts\python.exe examples\research_52w_high.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe, universe_members,
)
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, winsorize_cross_sectional,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
TOP_N = 500
QS = [0.1, 0.2, 0.3]


def _avg_xs_corr(a: pd.DataFrame, b: pd.DataFrame, min_names: int = 20) -> float:
    vals = []
    for t in a.index:
        x, y = a.loc[t], b.reindex(columns=a.columns).loc[t]
        m = x.notna() & y.notna()
        if int(m.sum()) >= min_names and x[m].std() > 0 and y[m].std() > 0:
            vals.append(float(np.corrcoef(x[m].to_numpy(float), y[m].to_numpy(float))[0, 1]))
    return float(np.nanmean(vals)) if vals else float("nan")


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1

    print(f"=== ⑧ 52週高値プロキシミティ {START}〜{END} 上位{TOP_N} ===")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, turn = assemble_panel(snaps, "AdjC"), assemble_panel(snaps, "Va")
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=TOP_N, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj = adj.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    adv = turn.reindex(columns=superset)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})
    print(f"パネル {adj.shape[0]}ヶ月 × ユニバース{len(superset)}銘柄")

    def prep(f: pd.DataFrame) -> pd.DataFrame:
        fm = apply_universe_mask(f.reindex(columns=superset), umask)
        return cross_sectional_zscore(sector_neutralize(
            winsorize_cross_sectional(fm), sector))

    # 52週高値プロキシミティ＝ 現値 / 過去12ヶ月最大（PIT・≤t）
    high52 = adj.rolling(12, min_periods=12).max()
    proximity = adj / high52
    signal = prep(proximity)                       # 高値圏=高シグナル=ロング
    cov = int((~proximity.isna()).sum().sum())
    print(f"proximity 有効セル {cov:,}")

    strats = [CrossSectionalStrategy(signal, q, name=f"high_52w(q={q})") for q in QS]

    with default_registry() as reg:
        v = judge_grid(
            strats, view, scope="high_52w",
            hypothesis="52週高値に近い（現値/52週高値が1に近い）銘柄は将来アウトパフォームする"
                       "（George-Hwang・アンカリング/アンダーリアクション）",
            economic_rationale="投資家は52週高値をアンカーにし、好材料を高値圏で過小評価＝徐々に織り込む。"
                               "George-Hwang は高値プロキシミティが伝統的モメンタムを subsume すると報告。"
                               "反対側はアンカーに縛られる参加者。",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    # 独立性：伝統的 12-1 モメンタムとの相関（George-Hwang の主張＝別物か）
    mom = prep(adj.shift(1) / adj.shift(12) - 1.0)
    print("\n--- 独立性（proximity と 12-1 モメンタムの月平均XS相関）---")
    print(f"  vs momentum(12-1)  ρ̄ = {_avg_xs_corr(signal, mom):+.2f}")

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {r.name:<16} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
