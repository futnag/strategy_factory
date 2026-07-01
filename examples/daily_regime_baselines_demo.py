"""step 4 配管検証：§6.3 ベースライン5本を完全統制で回す（value 戦略・スタブ二層）。

B&H／無フィルタ／週足のみ／日足のみ／二層(スタブ) を同一コスト（レジーム依存・Amihud→impact）・
同一執行ラグ(T+1)・同一評価窓で比較。日足のみ＝方式A λ=0（統制不変条件は単体テスト済）。

⚠ 週足アンカーは momentum スタブ → 本数字は**配管検証**（勝ち負けでなく「ベースラインが妥当値域・
   二層と日足のみが極端乖離しない・コスト依存が効く」が合格）。中核判定は step 5。

実行：python examples/daily_regime_baselines_demo.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from invest_system.research.daily_regime import backtest as drbt
from invest_system.research.daily_regime import data as drdata
from invest_system.research.daily_regime import multiscale, tiers, weekly_anchor
from invest_system.research.daily_regime.asof_bridge import anchor_daily_panel

START, LAM = "2023-01-01", 2.0
KW = dict(refit_every=20, window=378, warmup=252, n_init=2, align_col=0,
          min_dwell=3, random_state=7, max_pool_codes=30)


def build_feat(adj, va, code, vol_w=20, amh_w=10):
    ret = adj[code].pct_change(fill_method=None)
    rv = ret.rolling(vol_w).std()
    amh = (ret.abs() / va[code].where(va[code] > 0)).rolling(amh_w).mean()
    return pd.DataFrame({"rv": rv, "lamihud": np.log(amh.replace(0.0, np.nan))}).dropna()


def main():
    t0 = time.perf_counter()
    panels = drdata.load_daily_panels(start=START)
    adj, va = panels.adj_close, panels.turnover
    s33 = drdata.load_s33_map()
    anchor = weekly_anchor.build_weekly_anchor(adj, s33)
    mem = tiers.membership_panel(
        tiers.assign_tiers(adj, va, n_tiers=3, adv_floor=5e7, smooth_window=60, refit_every=20))
    asof = mem.loc[:adj.index[-1]].iloc[-1]
    anchor_ids = set(anchor["anchor_id"].unique())
    cand = [c for c in mem.columns if asof.get(c) == "T2"
            and adj[c].notna().sum() > 412 and str(s33.get(c)) in anchor_ids]
    target = cand[0]
    sector = str(s33.get(target))
    pool = [c for c in mem.columns if asof.get(c) == "T2" and adj[c].notna().sum() > 412][:30]
    if target not in pool:
        pool = [target] + pool[:29]
    feat = {c: build_feat(adj, va, c) for c in pool}
    feat = {c: f for c, f in feat.items() if len(f) > 412}
    target = target if target in feat else next(iter(feat))
    fidx = feat[target].index
    print(f"[pick] target={target} sector={sector} pool={len(feat)} days={len(fidx)} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    ms0 = multiscale.walk_forward_multiscale(target, feat, mem, anchor, sector, lam_inject=0.0, **KW)
    msX = multiscale.walk_forward_multiscale(target, feat, mem, anchor, sector, lam_inject=LAM, **KW)
    ret = adj[target].pct_change(fill_method=None)
    amh = (ret.abs() / va[target].where(va[target] > 0)).rolling(20).mean().reindex(fidx)
    wk = anchor_daily_panel(anchor, fidx, "p_bear", sector).reindex(fidx)

    res = drbt.run_value_baselines(
        target, adj[[target]].reindex(fidx), ms0["prob_stressed"], msX["prob_stressed"],
        wk, amh, lookback=60, filter_thresh=0.6, size_floor=0.2,
        base_bps=10.0, impact_coef=20.0, execution_lag=1, warmup=252)

    print("\n===== §6.3 ベースライン（value 戦略・同一コスト/執行ラグ/窓・レジーム依存コスト） =====")
    rows = []
    for name in ["B&H", "no-filter", "weekly-only", "daily-only", "two-layer"]:
        _, s = res[name]
        rows.append({"baseline": name, "ann_ret": s["ann_ret"], "sharpe": s["sharpe"],
                     "maxdd": s["maxdd"], "turnover": s["turnover"], "n": s["n"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)

    # 配管検証の合格条件（勝ち負けでなく「壊れていない」）
    sh = {r["baseline"]: r["sharpe"] for r in rows}
    print("\n--- 配管検証（スタブゆえ数字は証拠でない） ---", flush=True)
    print(f"全 Sharpe 有限: {all(np.isfinite(v) for v in sh.values())}", flush=True)
    print(f"日足のみ vs 二層(スタブ) Sharpe 乖離 = {abs(sh['two-layer']-sh['daily-only']):.3f} "
          f"（極端なら スタブが非自明効果＝要調査）", flush=True)
    print(f"平均ターンオーバー: no-filter {rows[1]['turnover']:.3f} / 二層 {rows[4]['turnover']:.3f} "
          f"（レジームフィルタで建玉が間引かれコスト/回転が変わる）", flush=True)
    print(f"\n[done] {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
