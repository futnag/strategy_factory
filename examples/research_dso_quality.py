r"""仮説検証 ②：受取債権／DSO 動学＝収益の質ファクター（フォレンジック由来）。

会計不正フォレンジック（docs/29）の結論：素朴アクルーアル(lift0.8)は効かないが、**ΔDSO・受取債権が
売上より速く伸びる(recv_minus_sales_growth) は lift2.0〜2.4** で収益偽装型を判別する。これは既存の
ファクター集合(value/quality/accruals…)に**無い**。本検証は「フォレンジックのレア事象スクリーンを、
全銘柄の収益の質プレミアムへ一般化したら、横断リターンを予測するか」を判定器で裁く。

シグナル: EDINET 有報明細由来の DSO パネル（`data/forensic/dso_panel.parquet`、**実 DiscDate でPIT**）の
`dso_change`（ΔDSO YoY）＋ `recv_minus_sales_growth`。高い＝受取債権膨張＝低品質。**低品質をショート/
高品質をロング**。比率因子なので winsorize→セクター中立→z。分位 q∈{0.1,0.2,0.3}（K=3）。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\Scripts\python.exe examples\research_dso_quality.py
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
from invest_system.equities.fundamentals import fundamentals_panel, point_in_time  # noqa: E402
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
DSO_PATH = Path("data/forensic/dso_panel.parquet")
VQ_FIELDS = ["ShOutFY", "TrShFY", "Eq", "TA", "FEPS", "FNP", "FOP", "FSales",
             "FDivAnn", "CFO", "NP"]


def _avg_xs_corr(a: pd.DataFrame, b: pd.DataFrame, min_names: int = 20) -> float:
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
    if not DSO_PATH.exists():
        print(f"ERROR: {DSO_PATH} が必要です（examples/forensic_dso_panel.py）。")
        return 1

    print(f"=== ② DSO/収益の質ファクター {START}〜{END} 上位{TOP_N} ===")
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

    # DSO 明細を実 DiscDate で as-of（PIT・先読みなし）
    dso = pd.read_parquet(DSO_PATH)
    dso["Code"] = dso["Code"].astype(str)
    dso = dso.dropna(subset=["DiscDate"])
    pit_dso = point_in_time(dso, rebal, ["dso_change", "recv_minus_sales_growth"],
                            date_col="DiscDate", code_col="Code", lag_days=1)

    def prep(f: pd.DataFrame) -> pd.DataFrame:
        fm = apply_universe_mask(f.reindex(columns=superset), umask)
        return cross_sectional_zscore(sector_neutralize(
            winsorize_cross_sectional(fm), sector))

    dchg = pit_dso["dso_change"].reindex(columns=superset)
    rms = pit_dso["recv_minus_sales_growth"].reindex(columns=superset)
    cov = int((~dchg.isna()).sum().sum())
    print(f"DSO as-of 有効セル {cov:,}（dso_change）")
    # 高 ΔDSO/受取債権膨張＝低品質。高品質(低劣化)をロング＝ −risk
    risk = (prep(dchg) + prep(rms)) / 2.0
    quality = -risk

    strats = [CrossSectionalStrategy(quality, q, name=f"dso_quality(q={q})") for q in QS]

    with default_registry() as reg:
        v = judge_grid(
            strats, view, scope="dso_quality",
            hypothesis="受取債権が売上より速く膨張する（高ΔDSO・高recv_minus_sales_growth）銘柄は"
                       "収益の質が低く将来アンダーパフォームする（収益の質プレミアムへ一般化）",
            economic_rationale="受取債権の質劣化は収益認識に直結するアクルーアル成分で、過大計上/回収難/"
                               "循環取引の先行指標。報告利益でなくキャッシュ化の質を見る。docs/29 で ΔDSO は"
                               "不正型 lift2.4＝素朴アクルーアル(0.8)を上回る。反対側は報告利益を額面で買う参加者。",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    # 独立性：既知factor（accruals=既試で失敗・value）との相関
    pit_vq = fundamentals_panel(rebal, VQ_FIELDS, codes=superset, lag_days=1)
    vqs = value_quality_size_factors(pit_vq, raw, adj)
    acc, bm = prep(vqs["accruals"]), prep(vqs["book_to_market"])
    print("\n--- 独立性（dso_quality シグナルと既知factorの月平均XS相関）---")
    for nm, other in (("accruals", acc), ("value(B/M)", bm)):
        print(f"  vs {nm:<12} ρ̄ = {_avg_xs_corr(quality, other):+.2f}")
    print("  |ρ̄| が小さいほど独立（既試 accruals の焼き直しでない）。")

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {r.name:<18} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
