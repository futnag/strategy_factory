"""中核判定の素の数字（1〜2銘柄）＋アンカー filtered 系列（スタブ→本物）— step 5 初回提示。

本物の週足方向性HMM（build_weekly_anchor_hmm）を方式A で注入し、§6.3 ベースライン5本を回す。
事前登録（regime_detection_core_judgment_preregistration.md）どおり：
  - 主指標＝コスト後 Sharpe（記述統計併記）。本丸＝二層 vs 日足のみ／二層 vs 週足のみ。
  - **1〜2銘柄は sample 1-2＝verdict でない。** 多銘柄展開に進むか構成見直すかの材料。
  - λ 固定・状態数は分離仮説（ここでは週足=3状態既定）。

実行：python examples/daily_regime_core_judgment_demo.py
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

START, LAM, N_TARGETS = "2023-01-01", 2.0, 2
KW = dict(refit_every=20, window=378, warmup=252, n_init=2, align_col=0,
          min_dwell=3, random_state=7, max_pool_codes=30)


def build_feat(adj, va, code, vol_w=20, amh_w=10):
    ret = adj[code].pct_change(fill_method=None)
    rv = ret.rolling(vol_w).std()
    amh = (ret.abs() / va[code].where(va[code] > 0)).rolling(amh_w).mean()
    return pd.DataFrame({"rv": rv, "lamihud": np.log(amh.replace(0.0, np.nan))}).dropna()


def sharpe_table(res):
    return {k: v[1]["sharpe"] for k, v in res.items()}


def main():
    t0 = time.perf_counter()
    panels = drdata.load_daily_panels(start=START)
    adj, va = panels.adj_close, panels.turnover
    s33 = drdata.load_s33_map()
    mem = tiers.membership_panel(
        tiers.assign_tiers(adj, va, n_tiers=3, adv_floor=5e7, smooth_window=60, refit_every=20))
    asof = mem.loc[:adj.index[-1]].iloc[-1]

    # ターゲット2銘柄（T2・履歴十分・所属セクターが構成十分）→ そのセクターだけアンカー構築
    s33_counts = s33.value_counts()
    cand = [c for c in mem.columns if asof.get(c) == "T2" and adj[c].notna().sum() > 412
            and s33_counts.get(str(s33.get(c)), 0) >= 15]
    targets = cand[:N_TARGETS]
    sectors = sorted({str(s33.get(c)) for c in targets})
    sec_codes = [c for c in adj.columns if str(s33.get(c)) in sectors]
    adj_sec, s33_sec = adj[sec_codes], s33[[c for c in s33.index if c in sec_codes]]
    print(f"[setup] targets={targets} sectors={sectors} ({time.perf_counter()-t0:.0f}s)", flush=True)

    stub = weekly_anchor.build_weekly_anchor(adj_sec, s33_sec)                 # 旧スタブ
    real = weekly_anchor.build_weekly_anchor_hmm(adj_sec, s33_sec, n_states=3)  # 本物 HMM
    print(f"[anchor] stub rows={len(stub)} real rows={len(real)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    results = {}
    for code in targets:
        sector = str(s33.get(code))
        pool = [c for c in mem.columns if asof.get(c) == "T2" and adj[c].notna().sum() > 412][:30]
        if code not in pool:
            pool = [code] + pool[:29]
        feat = {c: build_feat(adj, va, c) for c in pool}
        feat = {c: f for c, f in feat.items() if len(f) > 412}
        if code not in feat:
            continue
        fidx = feat[code].index
        ms0 = multiscale.walk_forward_multiscale(code, feat, mem, real, sector, lam_inject=0.0, **KW)
        msX = multiscale.walk_forward_multiscale(code, feat, mem, real, sector, lam_inject=LAM, **KW)
        ret = adj[code].pct_change(fill_method=None)
        amh = (ret.abs() / va[code].where(va[code] > 0)).rolling(20).mean().reindex(fidx)
        wk = anchor_daily_panel(real, fidx, "p_bear", sector).reindex(fidx)
        res = drbt.run_value_baselines(
            code, adj[[code]].reindex(fidx), ms0["prob_stressed"], msX["prob_stressed"],
            wk, amh, lookback=60, filter_thresh=0.6, size_floor=0.2,
            base_bps=10.0, impact_coef=20.0, execution_lag=1, warmup=252)
        sh = sharpe_table(res)
        # アンカー filtered 系列：スタブ vs 本物（p_bear）
        sb = anchor_daily_panel(stub, fidx, "p_bear", sector).reindex(fidx)
        rb = wk
        results[code] = {"sector": sector, "sharpe": sh,
                         "anchor_corr": float(sb.corr(rb)), "stub_pbear": float(sb.mean()),
                         "real_pbear": float(rb.mean()),
                         "real_pbear_std": float(rb.std())}
        print(f"  [{code}/{sector}] done ({time.perf_counter()-t0:.0f}s)", flush=True)

    print("\n===== 中核判定の素の数字（sample 1-2・verdict でない） =====")
    for code, d in results.items():
        sh = d["sharpe"]
        print(f"\n--- {code} (sector {d['sector']}) コスト後 Sharpe ---", flush=True)
        for k in ["B&H", "no-filter", "weekly-only", "daily-only", "two-layer"]:
            print(f"    {k:12s}: {sh[k]:+.3f}", flush=True)
        m1 = sh["two-layer"] - sh["daily-only"]
        m2 = sh["two-layer"] - sh["weekly-only"]
        print(f"    本丸① 二層−日足のみ(週足層の限界) = {m1:+.3f}", flush=True)
        print(f"    本丸② 二層−週足のみ(日足層の限界) = {m2:+.3f}", flush=True)
        print(f"    アンカー p_bear: スタブ平均 {d['stub_pbear']:.3f} → 本物平均 {d['real_pbear']:.3f} "
              f"(std {d['real_pbear_std']:.3f}, corr(スタブ,本物)={d['anchor_corr']:+.2f})", flush=True)

    print("\n[note] 事前登録どおり：これは sample 1-2。符号一貫性は多銘柄(≥2/3)で初めて判定。", flush=True)
    print(f"[done] {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
