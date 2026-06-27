r"""仮説検証 候補② Stage 2：深層 条件付き SDF（GRU 潜在マクロ状態）。

docs/46。numpy 版（Stage1）では H2（動的状態 vs 最新値）が再現できなかった＝numpy proxy が
LSTM 潜在状態を表せていない、が原因と推定。本版は **GRU がマクロ水準の系列から潜在状態を学習**し、
no-arbitrage GMM 損失で SDF を訓練。**負コントロール＝Linear（最新水準のみ・recurrence無し）**と対比し、
「動的状態が要る／最新値では崩壊」を本来の形で検定する。要 torch（[dl]）。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\Scripts\python.exe examples\research_deep_sdf.py
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
from invest_system.portfolio import deep_sdf, sdf  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, PrecomputedWeights, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
TOP_N, MIN_TRAIN = 500, 36
FIELDS = ["ShOutFY", "TrShFY", "Eq", "FEPS", "FNP", "FOP", "FSales", "FDivAnn",
          "CFO", "NP", "TA"]


def _to_weights(sig):
    w = sig.sub(sig.mean(axis=1), axis=0)
    g = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(g, axis=0)


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    print(f"=== ② 深層SDF Stage2（GRU潜在状態） {START}〜{END} 上位{TOP_N} ===")
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
    zdict = {"value": prep(vqs["book_to_market"]), "ep": prep(vqs["earnings_yield"]),
             "roe": prep(vqs["roe"]), "mom": prep(vqs["momentum"]),
             "size": prep(vqs["size"]), "lowvol": prep(low_volatility(adj, window=12))}
    fwd = adj.pct_change().shift(-1)
    F = sdf.managed_portfolios(zdict, fwd)

    # マクロ水準（標準化）の系列＝GRU入力。負コントロールは同じ列を Linear で最新値のみ参照
    def al(s):
        return asof_align(s, rebal, lag_days=1).iloc[:, 0]
    mac = load_macro(["jp_10y", "us_10y", "vix"])
    jp10, us10, vix = al(mac["jp_10y"]), al(mac["us_10y"]), al(mac["vix"])
    lnfx = np.log(al(load_external_prices(["usdjpy"], field="close")["usdjpy"]))

    def rz(s, w=36, mp=12):
        return (s - s.rolling(w, min_periods=mp).mean()) / s.rolling(w, min_periods=mp).std()
    macro_df = pd.DataFrame({"jp10": rz(jp10), "us10": rz(us10), "spread": rz(jp10 - us10),
                             "fx": rz(lnfx), "vix": rz(vix)}, index=rebal)

    print("walk-forward 学習中（深層GRU / 負コントロール / numpy）…")
    kw = dict(epochs=400, lr=0.01, weight_decay=0.1, state_dim=4, hidden=8)
    w_gru = deep_sdf.walk_forward_deep_weights(F, zdict, macro_df, use_gru=True,
                                               refit=12, min_train=MIN_TRAIN, **kw)
    w_lin = deep_sdf.walk_forward_deep_weights(F, zdict, macro_df, use_gru=False,
                                               refit=12, min_train=MIN_TRAIN, **kw)
    # numpy アンカー（Stage1 と同設定）
    S_dyn = pd.DataFrame({"d_spread6": rz((jp10 - us10).diff(6)), "d_fx6": rz(lnfx.diff(6)),
                          "d_vix3": rz(vix.diff(3))}, index=rebal)
    w_np = sdf.walk_forward_weights(F, zdict, state=S_dyn, min_train=MIN_TRAIN)
    w_comp = _to_weights(sum(zdict.values()) / len(zdict))

    strats = [
        PrecomputedWeights(w_gru, name="deep_sdf_gru(state)"),
        PrecomputedWeights(w_lin, name="deep_sdf_linear(latest=control)"),
        PrecomputedWeights(w_np, name="sdf_cond(state)"),          # 既登録（冪等）
        PrecomputedWeights(w_comp, name="composite_ew"),           # 既登録（冪等）
    ]
    with default_registry() as reg:
        v = judge_grid(
            strats, view, scope="conditional_sdf",
            hypothesis="GRUがマクロ水準系列から潜在経済状態を抽出し no-arbitrage SDF を条件付けると、"
                       "最新水準のみ(Linear・負制御)を上回り、numpy proxy で再現できなかったH2を実現する",
            economic_rationale="動的状態は景気循環を表すが最新増分は表せない（Chen-Pelger-Zhu）。recurrent な"
                               "状態抽出で de-risk でなく価格付けの α を取る。私のボラ・ゲート全敗の正面修正。",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1, execution_lag=0)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}_deep.html"))

    res = {r.name: r for r in v.results}
    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {r.name:<30} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")

    def s(n):
        return res[n].sr_ann if n in res else float("nan")
    print("\n--- H2 本検定（深層）---")
    print(f"  GRU潜在状態 {s('deep_sdf_gru(state)'):+.2f} vs Linear最新値(負制御) "
          f"{s('deep_sdf_linear(latest=control)'):+.2f}（GRU>Linear かつ Linear崩壊 が仮説）")
    print(f"  参考: numpy sdf_cond {s('sdf_cond(state)'):+.2f} / composite {s('composite_ew'):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
