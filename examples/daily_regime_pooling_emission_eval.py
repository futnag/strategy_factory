"""#1#3#2：放出安定性で per-stock vs pooled を判定（T2 対照・T3 枯渇帯）。

判定可能化した比較：
  #1 持続性機構の*前*で、放出パラメータ（calm–stressed gap）の refit 間 CV を測る。
  #3 per-stock も pooled も**同一 t 混合**で推定（唯一の差＝自銘柄窓 vs ティア・プール窓）。
  #2 T2（安定帯・対照）と T3（非流動・枯渇帯・主戦場）で走らせる。
事前登録基準（pooling_eval 定数）に機械的に照らして採否を出す。λ 固定・非最適化。

実行：python examples/daily_regime_pooling_emission_eval.py
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
from invest_system.research.daily_regime.detector import _standardize_past

START = "2023-01-01"
TIERS = ["T2", "T3"]            # 対照（安定）・主戦場（枯渇）
N_TARGETS = 8
POOL_CAP = 30
N_STATES, DOF, N_INIT = 2, 4.0, 2
REFIT_EVERY, WINDOW, WARMUP = 20, 378, 252


def build_feat(adj, va, code, vol_w=20, amh_w=10):
    ret = adj[code].pct_change(fill_method=None)
    rv = ret.rolling(vol_w).std()
    amh = (ret.abs() / va[code].where(va[code] > 0)).rolling(amh_w).mean()
    return pd.DataFrame({"rv": rv, "lamihud": np.log(amh.replace(0.0, np.nan))}).dropna()


def std_window(feat, t, window):
    w = feat.loc[:t]
    w = w.to_numpy(dtype=float)[-window:]
    if len(w) < N_STATES + 2 or not np.isfinite(w).all():
        return None
    return _standardize_past(w)


def main():
    t0 = time.perf_counter()
    print(f"[load] start={START}", flush=True)
    panels = drdata.load_daily_panels(start=START)
    adj, va = panels.adj_close, panels.turnover
    print(f"[load] {adj.shape} ({time.perf_counter()-t0:.0f}s)", flush=True)
    tlong = tiers.assign_tiers(adj, va, n_tiers=3, adv_floor=5e7, smooth_window=60, refit_every=20)
    mem = tiers.membership_panel(tlong)
    idx = adj.index
    refit_dates = list(idx[WARMUP::REFIT_EVERY])
    asof = mem.loc[:idx[-1]].iloc[-1]

    for TIER in TIERS:
        peers_all = [c for c in mem.columns if asof.get(c) == TIER and adj[c].notna().sum() > WARMUP + 120]
        pool_codes = peers_all[:POOL_CAP]
        targets = pool_codes[:N_TARGETS]
        feat = {c: build_feat(adj, va, c) for c in pool_codes}
        feat = {c: f for c, f in feat.items() if len(f) > WARMUP + 40}
        targets = [c for c in targets if c in feat]
        pool_codes = [c for c in pool_codes if c in feat]
        print(f"\n##### TIER {TIER}: peers={len(peers_all)} pool={len(pool_codes)} targets={len(targets)}", flush=True)

        # pooled：ティアごとに 1 トラック（各 refit でプール・スタックを t 混合）
        def pool_iter():
            for r in refit_dates:
                peers = [c for c in pool_codes if asof_tier(mem, c, r) == TIER]
                stk = [std_window(feat[c], r, WINDOW) for c in peers if c in feat]
                stk = [s for s in stk if s is not None]
                yield (r, np.vstack(stk) if len(stk) >= 2 else None)
        pooled_track = pe.track_emissions(pool_iter(), n_states=N_STATES, dof=DOF,
                                          n_init=N_INIT, random_state=7, align_col=0)
        s_pl = pe.stability_summary(pooled_track)

        ps_sum, pl_sum, rows = {}, {}, []
        for code in targets:
            ps_iter = ((r, std_window(feat[code], r, WINDOW)) for r in refit_dates)
            ps_track = pe.track_emissions(ps_iter, n_states=N_STATES, dof=DOF,
                                          n_init=N_INIT, random_state=7, align_col=0)
            s_ps = pe.stability_summary(ps_track)
            ps_sum[code], pl_sum[code] = s_ps, s_pl
            rows.append({"code": code, "ps_gapcv": s_ps["gap_cv"], "pl_gapcv": s_pl["gap_cv"],
                         "ps_deg": s_ps["degeneracy_rate"], "pl_deg": s_pl["degeneracy_rate"],
                         "ps_refits": s_ps["n_refits"]})
        df = pd.DataFrame(rows)
        v = pe.verdict(ps_sum, pl_sum)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
        print(f"[pooled tier {TIER}] gap_cv={s_pl['gap_cv']:.3f} deg={s_pl['degeneracy_rate']:.3f} "
              f"refits={s_pl['n_refits']}", flush=True)
        print(f">>> VERDICT {TIER}: better={v['n_better']}/{v['n']} frac={v['frac_better']:.2f} "
              f"ADOPT_POOLING={v['adopt_pooling']}  (criterion: >=2/3 with gap-CV<=0.8x and deg not worse)", flush=True)

    print(f"\n[done] {time.perf_counter()-t0:.0f}s", flush=True)


def asof_tier(mem, code, t):
    row = mem.loc[:t]
    return row.iloc[-1].get(code) if len(row) else None


if __name__ == "__main__":
    main()
