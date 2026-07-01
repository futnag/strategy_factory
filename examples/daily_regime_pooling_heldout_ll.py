"""保留予測LL裁定：per-stock vs pooled を各銘柄の窓外未来でスコア（T2/T3）＋ T3 churn 監査。

gap-CV（分散のみ）と filtered 出力（分散＋バイアス）の対立を、上位指標＝**窓外保留 LL**
（分散とバイアスを同時に含む）で裁定する。判定基準は事前固定：ティアごとに per-stock と
pooled を各銘柄自身の窓外未来でスコアし、**ティア内平均/中央値 LL が高い方**を採る。

着地は3通り：(i) T2 pooled・T3 per-stock（gap-CV反転をLLが追認）、(ii) どのティアも per-stock
（バイアスが分散低減を上回る＝pooling 不採用寄り）、(iii) 混在。LL の符号で確定。

釘：λ 固定（LL を見てλを動かさない）／T3 churn 併走（incoherence が入替過多由来かの切り分け）／
held-out は assert_future_block で PIT 厳守。

実行：python examples/daily_regime_pooling_heldout_ll.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from invest_system.research.daily_regime import data as drdata
from invest_system.research.daily_regime import pooling_eval as pe
from invest_system.research.daily_regime import tiers

START = "2023-01-01"
TIERS = ["T2", "T3"]
N_TARGETS, POOL_CAP = 8, 30
N_STATES, DOF, N_INIT = 2, 4.0, 2
REFIT_EVERY, WINDOW, WARMUP, HORIZON = 20, 378, 252, 20


def build_feat(adj, va, code, vol_w=20, amh_w=10):
    ret = adj[code].pct_change(fill_method=None)
    rv = ret.rolling(vol_w).std()
    amh = (ret.abs() / va[code].where(va[code] > 0)).rolling(amh_w).mean()
    return pd.DataFrame({"rv": rv, "lamihud": np.log(amh.replace(0.0, np.nan))}).dropna()


def tier_exit_rate(mem, tier, refit_dates):
    """refit 境界ごとの「ティア離脱率」平均（churn 監査）。"""
    rates, prev = [], None
    for r in refit_dates:
        row = mem.loc[:r]
        members = set(row.iloc[-1][row.iloc[-1] == tier].index) if len(row) else set()
        if prev:
            rates.append(len(prev - members) / len(prev))
        prev = members
    return float(np.mean(rates)) if rates else np.nan


def main():
    t0 = time.perf_counter()
    print(f"[load] start={START}", flush=True)
    panels = drdata.load_daily_panels(start=START)
    adj, va = panels.adj_close, panels.turnover
    tlong = tiers.assign_tiers(adj, va, n_tiers=3, adv_floor=5e7, smooth_window=60, refit_every=20)
    mem = tiers.membership_panel(tlong)
    idx = adj.index
    refit_dates = list(idx[WARMUP::REFIT_EVERY])
    asof = mem.loc[:idx[-1]].iloc[-1]
    print(f"[load] {adj.shape} refits={len(refit_dates)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    summary = {}
    for TIER in TIERS:
        churn = tier_exit_rate(mem, TIER, refit_dates)
        peers = [c for c in mem.columns if asof.get(c) == TIER and adj[c].notna().sum() > WARMUP + 160]
        pool_codes = peers[:POOL_CAP]
        feat = {c: build_feat(adj, va, c) for c in pool_codes}
        feat = {c: f for c, f in feat.items() if len(f) > WARMUP + HORIZON + 40}
        pool_codes = [c for c in pool_codes if c in feat]
        targets = pool_codes[:N_TARGETS]
        print(f"\n##### TIER {TIER}: pool={len(pool_codes)} targets={len(targets)} "
              f"churn(離脱率/refit)={churn:.3f}", flush=True)

        pooled_by_refit = pe.pooled_emissions_by_refit(
            [feat[c] for c in pool_codes], refit_dates, window=WINDOW,
            n_states=N_STATES, dof=DOF, n_init=N_INIT, random_state=7)

        rows = []
        for code in targets:
            ps_ll, pl_ll, n = pe.heldout_ll_for_target(
                feat[code], pooled_by_refit, refit_dates, window=WINDOW, horizon=HORIZON,
                n_states=N_STATES, dof=DOF, n_init=N_INIT, random_state=7)
            rows.append({"code": code, "ps_ll": ps_ll, "pl_ll": pl_ll,
                         "diff(pl-ps)": pl_ll - ps_ll, "n": n})
        df = pd.DataFrame(rows)
        ps_mean, pl_mean = df["ps_ll"].mean(), df["pl_ll"].mean()
        ps_med, pl_med = df["ps_ll"].median(), df["pl_ll"].median()
        n_pl_better = int((df["diff(pl-ps)"] > 0).sum())
        winner = "pooled" if pl_mean > ps_mean else "per-stock"
        summary[TIER] = {"winner": winner, "pl_minus_ps_mean": pl_mean - ps_mean,
                         "n_pl_better": n_pl_better, "n": len(df), "churn": churn}
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
        print(f"[{TIER}] mean LL: per-stock {ps_mean:.4f} vs pooled {pl_mean:.4f}  "
              f"(median {ps_med:.4f} vs {pl_med:.4f})  pl>ps: {n_pl_better}/{len(df)}  "
              f"WINNER={winner}", flush=True)

    print("\n===== 裁定（窓外保留LL・λ固定） =====")
    for TIER, s in summary.items():
        print(f"  {TIER}: WINNER={s['winner']}  Δ(pl-ps)mean={s['pl_minus_ps_mean']:+.4f}  "
              f"pl>ps {s['n_pl_better']}/{s['n']}  churn={s['churn']:.3f}", flush=True)
    w = {t: s["winner"] for t, s in summary.items()}
    if w.get("T2") == "pooled" and w.get("T3") == "per-stock":
        landing = "(i) T2 pooled・T3 per-stock（gap-CV 反転を LL が追認）"
    elif all(v == "per-stock" for v in w.values()):
        landing = "(ii) どのティアも per-stock（バイアス>分散低減・pooling 不採用寄り）"
    else:
        landing = "(iii) 混在"
    print(f"\n>>> LANDING: {landing}", flush=True)
    print(f"[done] {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
