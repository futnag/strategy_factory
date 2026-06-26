r"""仮説検証 ⑥：FX/金利でレジーム条件付けしたセクター・ローテーション。

USD/JPY と JGB 金利がセクター主導を駆動する（円安→輸出株、金利↑→銀行/value）という日本市場の
強いマクロ・ベータを、**各セクターの推定 FX/金利ベータ × マクロ・トレンドで条件付けたセクター L/S**
として判定器で裁く。マクロは GKX の特徴量や causal_sector の診断では使ったが、**明示的・経済動機の
ローテーション"規則"**は未テスト。

シグナル（各月・各S33業種、PIT）:
- beta_fx[t,s]   = 過去36ヶ月の業種リターン回帰 on ΔlnUSDJPY（円安感応度）
- beta_rate[t,s] = 同 on Δ(JGB10y)
- signal_fx   = beta_fx × fx_mom6   （円安トレンドなら高FXベータ＝輸出をロング）
- signal_rate = beta_rate × rate_mom6
- signal_combo = z(fx) + z(rate)
業種 EW リターンは PIT ユニバースの構成銘柄から組成（合成セクター指数）。q=0.2 で上位/下位を L/S。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\Scripts\python.exe examples\research_macro_sector_rotation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.data.external import asof_align, load_external_prices, load_macro  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    filter_common_stocks, point_in_time_universe, universe_members,
)
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, winsorize_cross_sectional,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
TOP_N = 500
QS = [0.2, 0.3]            # 33業種なので 0.2≈6-7 / 0.3≈10 を L/S


def _rolling_beta(Y: pd.DataFrame, x: pd.Series, w: int = 36, mp: int = 24) -> pd.DataFrame:
    """各列(業種)の x への過去 w ヶ月ローリング単回帰ベータ（PIT・先読みなし）。"""
    x = x.reindex(Y.index)
    ex = x.rolling(w, min_periods=mp).mean()
    ey = Y.rolling(w, min_periods=mp).mean()
    exy = Y.mul(x, axis=0).rolling(w, min_periods=mp).mean()
    cov = exy.sub(ey.mul(ex, axis=0))
    var = x.rolling(w, min_periods=mp).var(ddof=0)
    return cov.div(var.replace(0.0, np.nan), axis=0)


def _avg_xs_corr(a: pd.DataFrame, b: pd.DataFrame, min_n: int = 8) -> float:
    vals = []
    for t in a.index:
        x, y = a.loc[t], b.reindex(columns=a.columns).loc[t]
        m = x.notna() & y.notna()
        if int(m.sum()) >= min_n and x[m].std() > 0 and y[m].std() > 0:
            vals.append(float(np.corrcoef(x[m], y[m])[0, 1]))
    return float(np.nanmean(vals)) if vals else float("nan")


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1

    print(f"=== ⑥ FX/金利 × セクター・ローテーション {START}〜{END} ===")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, turn = assemble_panel(snaps, "AdjC"), assemble_panel(snaps, "Va")
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=TOP_N, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj = adj.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    rebal = adj.index

    # --- 33業種 EW リターン → 合成セクター指数（PITユニバース構成銘柄） ---
    smap = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    sec_of_col = smap.reindex([str(c) for c in adj.columns])
    sec_of_col.index = adj.columns
    valid = sec_of_col.dropna()
    stock_ret = adj.pct_change().where(umask)
    sector_ret = stock_ret[valid.index].T.groupby(valid.values).mean().T  # Date×S33
    sector_px = (1.0 + sector_ret.fillna(0.0)).cumprod() * 100.0
    view = AsOfView({"close": sector_px})
    print(f"合成セクター指数 {sector_px.shape[1]}業種 × {sector_px.shape[0]}ヶ月")

    # --- マクロ（PIT・as-of, ≤t-1） ---
    usdjpy = asof_align(load_external_prices(["usdjpy"], field="close")["usdjpy"],
                        rebal, lag_days=1)["usdjpy"]
    jp10 = asof_align(load_macro(["jp_10y"])["jp_10y"], rebal, lag_days=1)["jp_10y"]
    lnfx = np.log(usdjpy)
    dfx, drate = lnfx.diff(), jp10.diff()
    fx_mom, rate_mom = lnfx.diff(6), jp10.diff(6)            # 6ヶ月トレンド
    print(f"USDJPY {usdjpy.iloc[-13]:.1f}→{usdjpy.iloc[-1]:.1f}（直近13ヶ月）, "
          f"JGB10y {jp10.iloc[-1]:.2f}%")

    beta_fx = _rolling_beta(sector_ret, dfx)
    beta_rate = _rolling_beta(sector_ret, drate)

    def zwin(f: pd.DataFrame) -> pd.DataFrame:
        return cross_sectional_zscore(winsorize_cross_sectional(f))

    sig_fx = zwin(beta_fx.mul(fx_mom, axis=0))
    sig_rate = zwin(beta_rate.mul(rate_mom, axis=0))
    sig_combo = (sig_fx.add(sig_rate, fill_value=0.0))

    strats = []
    for nm, s in (("fx", sig_fx), ("rate", sig_rate), ("combo", sig_combo)):
        strats.append(CrossSectionalStrategy(s, 0.2, name=f"sector_rot_{nm}(q=0.2)"))

    with default_registry() as reg:
        v = judge_grid(
            strats, view, scope="macro_sector_rotation",
            hypothesis="USD/JPYとJGB金利のトレンドで、各業種の推定マクロ・ベータに沿ってセクター露出を"
                       "回す（円安→高FXベータ＝輸出、金利↑→金利感応＝銀行）と超過リターンを生む",
            economic_rationale="日本市場は業種レベルで強く持続的なFX/金利ベータを持つ（輸出の換算益・銀行の"
                               "利鞘）。マクロは数ヶ月かけて織り込まれ反対側に価格非感応の主体（実需・政策）。"
                               "GKX特徴量や因果診断と違い明示的な経済動機のローテーション規則。",
            registry=reg, costs_bps=10.0)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    # 独立性：素のセクター・モメンタム（業種自身の過去6ヶ月リターン）と相関
    sec_mom = zwin(sector_px.pct_change(6))
    print("\n--- 独立性（signal と素のセクター・モメンタムの月平均XS相関）---")
    for nm, s in (("fx", sig_fx), ("rate", sig_rate), ("combo", sig_combo)):
        print(f"  {nm:<6} vs sector_momentum  ρ̄ = {_avg_xs_corr(s, sec_mom):+.2f}")

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {r.name:<22} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
