r"""仮説検証 ⑨：低リスク・アノマリー（BAB＝低ベータ / IVOL＝低特異ボラ）。

高ベータ・高特異ボラ株が長期にリスク調整後アンダーパフォームする（Frazzini-Pedersen BAB /
Ang et al. IVOL）。既試の low_vol（総ボラ）に**隣接**するが、ベータ分解（系統 vs 特異）で別構成。
独立性は low_vol との相関で確認する。市場モデル＝ユニバース等加重リターンへの過去36ヶ月回帰。

シグナル（PIT・先読みなし）:
- bab  = −beta      （低ベータをロング）
- ivol = −特異ボラ  （単一ファクター残差の標準偏差。低 IVOL をロング）
セクター中立・winsorize・z。分位 q∈{0.1,0.2,0.3}×2signal（K=6）。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\Scripts\python.exe examples\research_low_risk.py
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
    cross_sectional_zscore, low_volatility, sector_neutralize, winsorize_cross_sectional,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
TOP_N = 500
QS = [0.1, 0.2, 0.3]


def _rolling_beta(Y: pd.DataFrame, x: pd.Series, w: int = 36, mp: int = 24) -> pd.DataFrame:
    x = x.reindex(Y.index)
    ex, ey = x.rolling(w, min_periods=mp).mean(), Y.rolling(w, min_periods=mp).mean()
    exy = Y.mul(x, axis=0).rolling(w, min_periods=mp).mean()
    cov = exy.sub(ey.mul(ex, axis=0))
    var = x.rolling(w, min_periods=mp).var(ddof=0)
    return cov.div(var.replace(0.0, np.nan), axis=0)


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

    print(f"=== ⑨ 低リスク・アノマリー（BAB/IVOL） {START}〜{END} 上位{TOP_N} ===")
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

    ret = adj.pct_change()
    mkt = ret.where(umask).mean(axis=1)                     # ユニバース等加重市場
    beta = _rolling_beta(ret, mkt)
    var_i = ret.rolling(36, min_periods=24).var()
    var_m = mkt.rolling(36, min_periods=24).var()
    ivol = np.sqrt((var_i.sub(beta.pow(2).mul(var_m, axis=0))).clip(lower=0.0))

    bab, ivol_sig = prep(-beta), prep(-ivol)
    strats = ([CrossSectionalStrategy(bab, q, name=f"bab(q={q})") for q in QS]
              + [CrossSectionalStrategy(ivol_sig, q, name=f"ivol(q={q})") for q in QS])

    with default_registry() as reg:
        v = judge_grid(
            strats, view, scope="low_risk_anomaly",
            hypothesis="高ベータ・高特異ボラ株は長期にリスク調整後アンダーパフォームする"
                       "（BAB / IVOL アノマリー）。低ベータ/低IVOLをロング",
            economic_rationale="レバレッジ制約のある投資家が高ベータを選好し過大評価（Frazzini-Pedersen）、"
                               "宝くじ選好で高IVOLが過大評価（Ang et al.）。反対側は制約/選好に縛られる参加者。",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    # 独立性：既試 low_vol（総ボラ）との相関＝焼き直しでないか
    lowvol = prep(low_volatility(adj, window=12))
    print("\n--- 独立性（既試 low_vol[総ボラ] との月平均XS相関）---")
    print(f"  bab  vs low_vol  ρ̄ = {_avg_xs_corr(bab, lowvol):+.2f}")
    print(f"  ivol vs low_vol  ρ̄ = {_avg_xs_corr(ivol_sig, lowvol):+.2f}")

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {r.name:<14} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
