"""旗艦の頑健性診断（throwaway・K不変・レジストリ書込なし）= docs/03 §6.27。

「厳しい目で再評価したら結論は変わるか」への定量的回答を2本立てで出す：

① グローバルKデフレート・ラダー — 報告 DSR は scope 局所（K=4＝switch 変種のみ）。
   多重検定族を広げると DSR がどこまで落ちるかを3段で測る：
     A: scope 局所（K=4・per-period 分散）＝報告値の再現
     B: 株式スリーブの月次ファクター探索族（実際に競合した候補・per-period 月次）＝公正な中間
     C: 全 scope（暗号方向・ペア裁定・指数イベント等の無関係研究線も含む全715）＝過剰に保守
   ＋ 参考: B を「帰無推定ノイズ分散（≈12/n 年率）」で計算した版（観測分散は本物の戦略間
   分散を拾い過剰デフレートしうるため、下限ではなく上側の感応として併記）。

② ローリング36ヶ月 年率Sharpe — switch とその2脚（value/pead_lt）が時系列で上向きか
   減衰かを点検。switch の安定性は「脚のローテーション」由来か、脚自身は減衰していないかを見る。

実行: .venv\\Scripts\\python.exe examples\\research_flagship_robustness_diag.py
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from invest_system.config import get_env
from invest_system.data.sources import jquants as jq
from invest_system.equities import events
from invest_system.equities.universe import (
    apply_universe_mask, filter_common_stocks, point_in_time_universe, universe_members)
from invest_system.equities.panel import (
    assemble_panel, fetch_month_end_snapshots, load_daily_panel)
from invest_system.equities.fundamentals import load_fundamentals, point_in_time
from invest_system.equities.factors import (
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors)
from invest_system.research import (
    AsOfView, CrossSectionalStrategy, CompositeStrategy, RegimeSwitch, backtest)
from invest_system.timeseries import vol_regime
from invest_system.validation.dsr import expected_max_sharpe, probabilistic_sharpe_ratio

START, END = "2016-07", "2026-05"
DB = "data/research_trials.db"
# 株式スリーブの月次ファクター候補として実際に競合した scope（同一探索族）
EQUITY_FAMILY = [
    "value_xs", "value_pead_combo", "value_pead_longtilt", "value_pead_regime",
    "value_pead_switch", "value_pead_wfswitch", "pead_revision", "pead_earn_overlay",
    "value_earn_overlay", "breadth_factors", "price_factors_gold", "residual_momentum",
    "short_interest_xs", "shareholder_return", "guidance_bias", "disclosure_timing",
    "earnings_runup"]


def global_k_ladder() -> None:
    """① グローバルKデフレート・ラダー（レジストリは読むだけ）。"""
    if not Path(DB).exists():
        print("（レジストリ未検出：① スキップ）"); return
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    rows = c.execute("SELECT scope,sharpe,n_obs,skew,kurt FROM trials "
                     "WHERE status='completed'").fetchall()
    n_all = len(rows)
    # 月次族 per-period 分散（全月次・n_obs<=250）
    fam = [r["sharpe"] for r in rows if r["sharpe"] is not None and r["scope"] in EQUITY_FAMILY]
    Nf, Vf = len(fam), float(np.var(fam, ddof=1))
    # 全 scope を年率換算（頻度を揃える）
    def ppy(n): return 12.0 if (n is None or n <= 250) else (52.0 if n <= 800 else 252.0)
    ann = [r["sharpe"] * np.sqrt(ppy(r["n_obs"])) for r in rows if r["sharpe"] is not None]
    Va = float(np.var(ann, ddof=1)); sq12 = np.sqrt(12.0)
    ft = {"switch": dict(sr=0.279378, n=118, sk=1.04909, ku=5.73495, kA=4, vA=0.0076),
          "combo_eqcap": dict(sr=0.302290, n=118, sk=1.04918, ku=4.67339, kA=4, vA=0.0097)}
    # 帰無ノイズ分散（per-period 月次・SR≈0 の推定分散 ≈ 1/n）
    V_null = 1.0 / 118.0
    print("=== ① グローバルKデフレート・ラダー（旗艦 DSR） ===")
    print(f"  族 N={Nf}（月次per-period）V={Vf:.4f} / 全 N={n_all} V[年率]={Va:.3f}")
    print(f"  {'戦略':12s} {'年率SR':>6s} | {'A:scope':>8s} | {'B:族(観測V)':>10s} | "
          f"{'B:族(帰無V)':>10s} | {'C:全715':>8s}")
    for nm, d in ft.items():
        srA = expected_max_sharpe(d["kA"], d["vA"])
        dA = probabilistic_sharpe_ratio(d["sr"], srA, d["n"], d["sk"], d["ku"])
        srB = expected_max_sharpe(Nf, Vf)
        dB = probabilistic_sharpe_ratio(d["sr"], srB, d["n"], d["sk"], d["ku"])
        srBn = expected_max_sharpe(Nf, V_null)
        dBn = probabilistic_sharpe_ratio(d["sr"], srBn, d["n"], d["sk"], d["ku"])
        srC = expected_max_sharpe(n_all, Va) / sq12
        dC = probabilistic_sharpe_ratio(d["sr"], srC, d["n"], d["sk"], d["ku"])
        print(f"  {nm:12s} {d['sr']*sq12:+6.2f} | {dA:8.3f} | {dB:10.3f} | "
              f"{dBn:10.3f} | {dC:8.3f}")
    c.close()
    print()


def roll_sharpe(r: pd.Series, win: int = 36) -> pd.Series:
    return (r.rolling(win).mean() / r.rolling(win).std(ddof=1)) * np.sqrt(12.0)


def rolling_decay() -> None:
    """② ローリング36ヶ月 年率Sharpe（backtest 直叩き・レジストリ不使用）。"""
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=300, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})
    fund = load_fundamentals(superset)

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, umask), sector))
    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    value = zN(value_quality_size_factors(pit, raw, adj)["book_to_market"])
    pead = zN(point_in_time(events.forecast_revision(fund), rebal, ["fcst_revision"],
                            date_col="DiscDate", lag_days=1)["fcst_revision"].reindex(columns=superset))
    daily = load_daily_panel(field="AdjC")
    vol_m = vol_regime(daily).reindex(rebal, method="ffill")
    value_ls = CrossSectionalStrategy(value, 0.2, name="value")
    pead_lt = CrossSectionalStrategy(pead, 0.2, name="pead_longtilt", long_only=True)
    combo = CompositeStrategy([value_ls, pead_lt], [0.5, 0.5], name="value+pead_lt")
    switch = RegimeSwitch(vol_m, {0: value_ls, 1: value_ls, 2: pead_lt}, name="switch")
    series = {s.name: backtest(s, view, costs_bps=15.0).returns.dropna()
              for s in (value_ls, pead_lt, combo, switch)}
    rs = pd.DataFrame({k: roll_sharpe(v) for k, v in series.items()}).dropna(how="all")
    print("=== ② ローリング36ヶ月 年率Sharpe（半年ごと） ===")
    print(f"  {'窓終端':>9s} {'value':>7s} {'pead_lt':>7s} {'combo':>7s} {'switch':>7s}")
    for t in rs.index[::6]:
        r = rs.loc[t]
        print(f"  {str(pd.Period(t, freq='M')):>9s} {r['value']:+7.2f} "
              f"{r['pead_longtilt']:+7.2f} {r['value+pead_lt']:+7.2f} {r['switch']:+7.2f}")
    sw = rs["switch"].dropna()
    slope = np.polyfit(np.arange(len(sw)), sw.values, 1)[0] * 12
    print(f"  switch: 最初{sw.iloc[0]:+.2f}→最新{sw.iloc[-1]:+.2f} / 線形トレンド{slope:+.3f}/年 "
          f"/ 直近12窓{sw.iloc[-12:].mean():+.2f} vs それ以前{sw.iloc[:-12].mean():+.2f}")


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要"); return 1
    global_k_ladder()
    rolling_decay()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
