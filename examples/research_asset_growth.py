r"""仮説検証 ④：アセットグロース／投資ファクター（FF5 CMA）。

総資産を急拡大（過剰投資・希薄化）した銘柄は将来劣後し、資産を絞る銘柄が上回る——という
投資(investment / asset growth)ファクターを、検証ファクトリ（PIT・セクター中立・大域デフレートDSR・
執行ラグ/コスト/容量・永続レジストリ）で一回勝負する。レジストリ未登録の新規 scope。

定義: 各月末の as-of 総資産 TA（DiscDate≤t−1 の最新開示）の **12ヶ月 YoY 成長率**。
シグナル = −成長率（低成長＝ロング側＝CMA: Conservative-Minus-Aggressive）。比率因子なので
winsorize→セクター中立→z 化。分位 q∈{0.1,0.2,0.3} の 3 点グリッド（K=3）。

注: fins_summary / daily_quotes の全件ミラー完了後に実行（download_jquants.py）。
実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\Scripts\python.exe examples\research_asset_growth.py
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
from invest_system.equities.fundamentals import fundamentals_panel  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
    winsorize_cross_sectional,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
TOP_N = 500
QS = [0.1, 0.2, 0.3]
FIELDS = ["ShOutFY", "TrShFY", "Eq", "TA", "EqAR", "FEPS", "FNP", "FOP", "FSales",
          "FDivAnn", "CFO", "NP"]


def _avg_xs_corr(a: pd.DataFrame, b: pd.DataFrame, min_names: int = 20) -> float:
    """各月の銘柄横断ピアソン相関を取り、月平均（独立性の診断）。"""
    vals = []
    for t in a.index:
        x, y = a.loc[t], b.reindex(columns=a.columns).loc[t]
        m = x.notna() & y.notna()
        if int(m.sum()) >= min_names:
            xx, yy = x[m].to_numpy(float), y[m].to_numpy(float)
            if xx.std() > 0 and yy.std() > 0:
                vals.append(float(np.corrcoef(xx, yy)[0, 1]))
    return float(np.nanmean(vals)) if vals else float("nan")


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1

    print(f"=== ④ アセットグロース（投資/CMA） {START}〜{END} 上位{TOP_N} ===")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=TOP_N, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    adv = turn.reindex(columns=superset)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})
    print(f"パネル {adj.shape[0]}ヶ月 × ユニバース{len(superset)}銘柄")

    pit = fundamentals_panel(rebal, FIELDS, codes=superset, lag_days=1)

    def prep(f: pd.DataFrame) -> pd.DataFrame:
        """ユニバースmask → winsorize(比率外れ値) → セクター中立 → z 化（事前固定）。"""
        fm = apply_universe_mask(f.reindex(columns=superset), umask)
        return cross_sectional_zscore(sector_neutralize(
            winsorize_cross_sectional(fm), sector))

    # --- アセットグロース：12ヶ月 YoY 総資産成長率、低成長をロング（CMA）---
    ta = pit["TA"].reindex(columns=superset)
    ta = ta.where(ta > 0)
    agr = ta / ta.shift(12) - 1.0                      # YoY 成長率（PIT・先読みなし）
    signal = prep(-agr)                                # 低成長=高シグナル=ロング
    cover = int((~agr.isna()).sum().sum())
    print(f"asset_growth 有効セル {cover:,}（12ヶ月 YoY）")

    strats = [CrossSectionalStrategy(signal, q, name=f"asset_growth(q={q})") for q in QS]

    with default_registry() as reg:
        v = judge_grid(
            strats, view, scope="asset_growth",
            hypothesis="総資産を急拡大した銘柄は過剰投資・希薄化で将来劣後し、資産を絞る銘柄が上回る"
                       "（投資/アセットグロース・ファクター, FF5 CMA）",
            economic_rationale="経営者の過剰投資・empire building と市場の成長の過剰外挿。FF5 の"
                               "investment(CMA)因子は国際的に頑健（日本含む）。実物投資の規律＝"
                               "需給/モメンタムと別系統。反対側は成長を額面で買う参加者。",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    # --- 独立性診断：既知ファクターとの月平均クロスセクション相関 ---
    vqs = value_quality_size_factors(pit, raw, adj)
    bm, mom, sz = prep(vqs["book_to_market"]), prep(vqs["momentum"]), prep(vqs["size"])
    print("\n--- 独立性（asset_growth シグナルと既知factorの月平均XS相関）---")
    for nm, other in (("value(B/M)", bm), ("momentum", mom), ("size", sz)):
        print(f"  vs {nm:<12} ρ̄ = {_avg_xs_corr(signal, other):+.2f}")
    print("  |ρ̄| が小さいほど独立（既知factorの代理でない）。")

    # --- IS/OOS（保留 OOS〜）---
    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {r.name:<20} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
