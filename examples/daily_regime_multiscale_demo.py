"""step 3 配管検証：方式A（週足アンカー→pooled 日足検知器へ事前注入）の挙動デモ。

注入前（lam=0＝ベースライン）と注入後（lam>0）の日足 filtered prob_stressed を比較し、
注入が週足レジーム（bear−bull）方向に効くこと、available_from 前は効かない（PIT）ことを示す。

⚠ 週足アンカーは step 5 まで**モメンタム・スタブ**。本デモは「結合機構が機械的に正しく動くか」の
   配管検証であって、マルチスケールがベースラインを超えるかの中核証拠ではない（中核は step 5）。

実行：python examples/daily_regime_multiscale_demo.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from invest_system.research.daily_regime import data as drdata
from invest_system.research.daily_regime import multiscale, tiers, weekly_anchor
from invest_system.research.daily_regime.asof_bridge import anchor_daily_panel

START = "2023-01-01"
LAM = 2.0
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
    tlong = tiers.assign_tiers(adj, va, n_tiers=3, adv_floor=5e7, smooth_window=60, refit_every=20)
    mem = tiers.membership_panel(tlong)
    asof = mem.loc[:adj.index[-1]].iloc[-1]
    anchor_ids = set(anchor["anchor_id"].unique())
    print(f"[load] {adj.shape} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # T2 のうち、所属セクターにアンカーがあり履歴十分な銘柄を1つ
    cand = [c for c in mem.columns if asof.get(c) == "T2"
            and adj[c].notna().sum() > 412 and str(s33.get(c)) in anchor_ids]
    target = cand[0]
    sector = str(s33.get(target))
    pool_codes = [c for c in mem.columns if asof.get(c) == "T2" and adj[c].notna().sum() > 412][:30]
    if target not in pool_codes:
        pool_codes = [target] + pool_codes[:29]
    feat = {c: build_feat(adj, va, c) for c in pool_codes}
    feat = {c: f for c, f in feat.items() if len(f) > 412}
    target = target if target in feat else next(iter(feat))
    print(f"[pick] target={target} sector(anchor_id)={sector} pool={len(feat)}", flush=True)

    base = multiscale.walk_forward_multiscale(target, feat, mem, anchor, sector, lam_inject=0.0, **KW)
    inj = multiscale.walk_forward_multiscale(target, feat, mem, anchor, sector, lam_inject=LAM, **KW)

    idx = feat[target].index
    p_bear = anchor_daily_panel(anchor, idx, "p_bear", sector).reindex(idx)
    p_bull = anchor_daily_panel(anchor, idx, "p_bull", sector).reindex(idx)
    wk = (p_bear - p_bull)                                   # 週足 弱気度（available_from ラグ・PIT）

    m = base["prob_stressed"].notna() & inj["prob_stressed"].notna()
    d = (inj["prob_stressed"] - base["prob_stressed"])[m]
    wkm = wk[m]
    first_af = anchor[anchor["anchor_id"] == sector]["available_from"].min()

    print("\n===== 方式A 注入前後の日足 filtered prob_stressed =====")
    print(f"対象日数: {int(m.sum())}  |Δ|平均: {d.abs().mean():.4f}  Δ最大: {d.max():.4f}", flush=True)
    print(f"corr(Δprob_stressed, 週足bear−bull) = {d.corr(wkm):.3f}  "
          f"（正＝弱気週で stressed↑＝方式A が方向どおり効く）", flush=True)
    # PIT：available_from 前は注入差ゼロ
    before = idx[m] < first_af
    leak = d[before.values].abs().max() if before.any() else 0.0
    print(f"PIT 確認: available_from({pd.Timestamp(first_af).date()}) 前の |Δ|最大 = {leak:.6f}（0＝リーク無し）", flush=True)
    # 弱気週 vs 強気週での prob_stressed（注入前→後）
    bear_d = wkm > wkm.quantile(0.8)
    bull_d = wkm < wkm.quantile(0.2)
    print(f"弱気週: prob_stressed  base {base['prob_stressed'][m][bear_d.values].mean():.3f} "
          f"-> inj {inj['prob_stressed'][m][bear_d.values].mean():.3f}", flush=True)
    print(f"強気週: prob_stressed  base {base['prob_stressed'][m][bull_d.values].mean():.3f} "
          f"-> inj {inj['prob_stressed'][m][bull_d.values].mean():.3f}", flush=True)
    print(f"\n[done] {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
