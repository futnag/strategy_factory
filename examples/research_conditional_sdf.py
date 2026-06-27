r"""仮説検証 候補② Stage 1：条件付き no-arbitrage SDF（regime-conditioning の"正解"）。

docs/46 事前登録。特性管理ポートフォリオ → KNS 縮小接線で SDF を推定し、(1) 無条件SDF、
(2) 潜在マクロ状態で条件付けたSDF、(3) 最新マクロ水準で条件付け（**負のコントロール**）、
(4) GKX型予測L/S、(5) 単純合成 を walk-forward OOS で対比し判定器で裁く。
H1=SDFが予測ML/合成を上回る、H2=動的状態は効く・最新水準は崩壊、H3=容量（大型可）。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\Scripts\python.exe examples\research_conditional_sdf.py
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
    apply_universe_mask, filter_common_stocks, point_in_time_universe, universe_members,
)
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.fundamentals import fundamentals_panel  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, low_volatility, sector_neutralize, value_quality_size_factors,
    winsorize_cross_sectional,
)
from invest_system.portfolio import sdf  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, PrecomputedWeights, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
TOP_N, MIN_TRAIN = 500, 36
FIELDS = ["ShOutFY", "TrShFY", "Eq", "FEPS", "FNP", "FOP", "FSales", "FDivAnn",
          "CFO", "NP", "TA"]


def _to_weights(sig: pd.DataFrame) -> pd.DataFrame:
    """連続シグナル → ダラーニュートラル・グロス1 のウェイト panel。"""
    w = sig.sub(sig.mean(axis=1), axis=0)
    g = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(g, axis=0)


def _gkx_weights(zdict, fwd, min_train=MIN_TRAIN, alpha=10.0, refit=12) -> pd.DataFrame:
    """GKX 型：walk-forward Ridge で次期リターンを予測→標準化ウェイト（PIT・年次refit）。"""
    from sklearn.linear_model import Ridge
    chars, dates, codes = list(zdict), fwd.index, fwd.columns
    Z3 = np.stack([zdict[c].reindex(index=dates, columns=codes).to_numpy()
                   for c in chars], axis=-1)            # T×N×K
    R = fwd.reindex(columns=codes).to_numpy()
    rows, model = {}, None
    for i in range(len(dates)):
        if i < min_train:
            continue
        if model is None or (i % refit == 0):
            Xs, ys = [], []
            for s in range(i):
                zr, rr = Z3[s], R[s]
                m = np.isfinite(rr) & np.all(np.isfinite(zr), axis=1)
                if int(m.sum()) > 5:
                    Xs.append(zr[m]); ys.append(rr[m])
            if Xs:
                model = Ridge(alpha=alpha).fit(np.vstack(Xs), np.concatenate(ys))
        if model is None:
            continue
        zr = Z3[i]; m = np.all(np.isfinite(zr), axis=1)
        pred = np.full(len(codes), np.nan)
        pred[m] = model.predict(zr[m])
        s = pd.Series(pred, index=codes)
        w = s - s.mean(); g = float(w.abs().sum())
        if g > 0:
            rows[dates[i]] = w / g
    return pd.DataFrame(rows).T.reindex(columns=codes)


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    print(f"=== ② 条件付きSDF Stage1 {START}〜{END} 上位{TOP_N} ===")
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

    def prep(f):
        fm = apply_universe_mask(f.reindex(columns=superset), umask)
        return cross_sectional_zscore(sector_neutralize(winsorize_cross_sectional(fm), sector))

    pit_f = fundamentals_panel(rebal, FIELDS, codes=superset, lag_days=1)
    vqs = value_quality_size_factors(pit_f, raw, adj)
    zdict = {
        "value": prep(vqs["book_to_market"]),
        "ep": prep(vqs["earnings_yield"]),
        "roe": prep(vqs["roe"]),
        "mom": prep(vqs["momentum"]),
        "size": prep(vqs["size"]),
        "lowvol": prep(low_volatility(adj, window=12)),
    }
    fwd = adj.pct_change().shift(-1)
    F = sdf.managed_portfolios(zdict, fwd)
    print(f"特性 {len(zdict)} / 管理ポートフォリオ F {F.shape}")

    # --- マクロ状態（動的＝trailing変化 / 負コントロール＝最新水準）---
    def al(s):
        return asof_align(s, rebal, lag_days=1).iloc[:, 0]

    mac = load_macro(["jp_10y", "us_10y", "vix"])
    jp10, us10, vix = al(mac["jp_10y"]), al(mac["us_10y"]), al(mac["vix"])
    lnfx = np.log(al(load_external_prices(["usdjpy"], field="close")["usdjpy"]))
    spread = jp10 - us10

    def rz(s, w=36, mp=12):
        return (s - s.rolling(w, min_periods=mp).mean()) / s.rolling(w, min_periods=mp).std()

    S_dyn = pd.DataFrame({"d_spread6": rz(spread.diff(6)), "d_fx6": rz(lnfx.diff(6)),
                          "d_vix3": rz(vix.diff(3))}, index=rebal)
    S_naive = pd.DataFrame({"spread": rz(spread), "fx": rz(lnfx), "vix": rz(vix)},
                           index=rebal)

    # --- ウェイト panel（walk-forward・PIT）---
    print("walk-forward SDF 構築中…")
    w_uncond = sdf.walk_forward_weights(F, zdict, state=None, min_train=MIN_TRAIN)
    w_cond = sdf.walk_forward_weights(F, zdict, state=S_dyn, min_train=MIN_TRAIN)
    w_naive = sdf.walk_forward_weights(F, zdict, state=S_naive, min_train=MIN_TRAIN)
    w_comp = _to_weights(sum(zdict.values()) / len(zdict))
    w_gkx = _gkx_weights(zdict, fwd)

    strats = [
        PrecomputedWeights(w_uncond, name="sdf_uncond"),
        PrecomputedWeights(w_cond, name="sdf_cond(state)"),
        PrecomputedWeights(w_naive, name="sdf_naive(latest=control)"),
        PrecomputedWeights(w_gkx, name="gkx_forecast"),
        PrecomputedWeights(w_comp, name="composite_ew"),
    ]
    with default_registry() as reg:
        v = judge_grid(
            strats, view, scope="conditional_sdf",
            hypothesis="特性管理ポートフォリオのKNS縮小接線（no-arbitrage SDF）に潜在マクロ状態を"
                       "目的関数で条件付けると、予測ML/単純合成を上回り、最新マクロ水準条件付け(負制御)は崩壊する",
            economic_rationale="no-arbitrage は予測でなく価格付け＝小型アーティファクトに頑健。動的状態は"
                               "景気循環を表すが最新増分は表せない（Chen-Pelger-Zhu）。私のボラ・ゲート全敗の正面修正。",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1, execution_lag=0)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    res = {r.name: r for r in v.results}
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {r.name:<26} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")

    def sr(n):
        return res[n].sr_ann if n in res else float("nan")
    print("\n--- 事前登録仮説の対比 ---")
    print(f"  H1 SDF vs 予測/合成: sdf_cond {sr('sdf_cond(state)'):+.2f} / sdf_uncond "
          f"{sr('sdf_uncond'):+.2f} vs gkx {sr('gkx_forecast'):+.2f} / composite {sr('composite_ew'):+.2f}")
    print(f"  H2 条件付けの作法: 動的状態 {sr('sdf_cond(state)'):+.2f} vs 最新水準(負制御) "
          f"{sr('sdf_naive(latest=control)'):+.2f}（動的>最新 が仮説・最新は崩壊すべき）")
    print(f"  H3 容量（大型可）: " + " / ".join(
        f"{r.name}=¥{res[r.name].capacity_jpy / 1e8:.0f}億"
        for r in v.results if np.isfinite(res[r.name].capacity_jpy)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
