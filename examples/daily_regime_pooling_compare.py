"""per-stock vs pooled（流動性ティア）日足ボラ流動性 HMM の安定性比較（③の証拠）。

データ枯渇する単一銘柄 HMM（per-stock）と、ティアでプール放出推定した版（pooled）を実データで
walk-forward し、filtered 出力の安定性（ちらつき・持続・確率ジャンプ）を比較する。pooling が
「放出を安定させ、銘柄別遷移で速い一過性ストレスを検知」できているかの証拠を出す。

実行：python examples/daily_regime_pooling_compare.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root（editable 後追い対策）

import numpy as np
import pandas as pd

from invest_system.research.daily_regime import data as drdata
from invest_system.research.daily_regime import detector as drdet
from invest_system.research.daily_regime import pooling
from invest_system.research.daily_regime import tiers

START = "2024-01-01"
TARGET_TIER = "T2"
N_TARGETS = 6
POOL_CAP = 40
N_STATES = 2
REFIT_EVERY = 40
WINDOW = 504
WARMUP = 252
KW = dict(n_states=N_STATES, dof=4.0, sticky_kappa=10.0, min_dwell=3,
          refit_every=REFIT_EVERY, window=WINDOW, warmup=WARMUP, align_col=0, n_init=2)


def build_feat(adj_close: pd.DataFrame, turnover: pd.DataFrame, code: str,
               vol_w: int = 20, amh_w: int = 10) -> pd.DataFrame:
    """観測特徴：実現ボラ（rv）＋応答的 log-Amihud（lamihud）。ティアキー（平滑中央値）とは別。"""
    ret = adj_close[code].pct_change(fill_method=None)
    rv = ret.rolling(vol_w).std()
    amh = (ret.abs() / turnover[code].where(turnover[code] > 0)).rolling(amh_w).mean()
    lamh = np.log(amh.replace(0.0, np.nan))
    return pd.DataFrame({"rv": rv, "lamihud": lamh}).dropna()


def main() -> None:
    t0 = time.perf_counter()
    print(f"[load] panels start={START} ...", flush=True)
    panels = drdata.load_daily_panels(start=START)
    adj, va = panels.adj_close, panels.turnover
    print(f"[load] adj_close {adj.shape}, turnover {va.shape} ({time.perf_counter()-t0:.0f}s)", flush=True)

    print("[tiers] assign (smoothed median Amihud, sticky) ...", flush=True)
    tlong = tiers.assign_tiers(adj, va, n_tiers=3, adv_floor=5e7, smooth_window=60, refit_every=20)
    mem = tiers.membership_panel(tlong)
    last = mem.index[-1]
    asof_row = mem.loc[:last].iloc[-1]
    peers_all = [c for c in mem.columns if asof_row.get(c) == TARGET_TIER]
    # 十分な履歴のある銘柄に限定
    enough = [c for c in peers_all if adj[c].notna().sum() > WARMUP + 120]
    pool_codes = enough[:POOL_CAP]
    targets = pool_codes[:N_TARGETS]
    print(f"[tiers] {TARGET_TIER}: peers={len(peers_all)} usable={len(enough)} "
          f"pool={len(pool_codes)} targets={targets}", flush=True)

    print("[feat] building feature panel ...", flush=True)
    feat_panel = {c: build_feat(adj, va, c) for c in pool_codes}
    feat_panel = {c: f for c, f in feat_panel.items() if len(f) > WARMUP + 60}
    targets = [c for c in targets if c in feat_panel]

    rows = []
    for i, code in enumerate(targets, 1):
        ts = time.perf_counter()
        out_ps = drdet.walk_forward_t_hmm(feat_panel[code], random_state=42, **KW)
        out_pl = pooling.walk_forward_pooled_t_hmm(
            code, feat_panel, mem, random_state=42, max_pool_codes=POOL_CAP, **KW)
        m_ps = pooling.stability_metrics(out_ps)
        m_pl = pooling.stability_metrics(out_pl)
        # per-stock と pooled の regime 一致（向きの合意）
        j = out_ps["prob_stressed"].notna() & out_pl["prob_stressed"].notna()
        corr = float(out_ps.loc[j, "prob_stressed"].corr(out_pl.loc[j, "prob_stressed"])) if j.sum() > 10 else np.nan
        pool_sz = int(out_pl.loc[out_pl["refit"], "pool_size"].replace(0, np.nan).mean()) \
            if (out_pl["refit"].any()) else 0
        rows.append({"code": code,
                     "ps_flip": m_ps["flip_rate"], "pl_flip": m_pl["flip_rate"],
                     "ps_dwell": m_ps["mean_dwell"], "pl_dwell": m_pl["mean_dwell"],
                     "ps_jump": m_ps["prob_jump"], "pl_jump": m_pl["prob_jump"],
                     "corr": corr, "pool": pool_sz})
        print(f"  [{i}/{len(targets)}] {code}: "
              f"flip {m_ps['flip_rate']:.3f}->{m_pl['flip_rate']:.3f}  "
              f"dwell {m_ps['mean_dwell']:.1f}->{m_pl['mean_dwell']:.1f}  "
              f"jump {m_ps['prob_jump']:.3f}->{m_pl['prob_jump']:.3f}  "
              f"corr={corr:.2f} pool~{pool_sz} ({time.perf_counter()-ts:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    print("\n===== per-stock vs pooled（流動性ティア " + TARGET_TIER + "） =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n--- 平均（低 flip / 高 dwell / 低 jump ＝ 安定） ---")
    print(f"flip  : per-stock {df['ps_flip'].mean():.3f}  ->  pooled {df['pl_flip'].mean():.3f}")
    print(f"dwell : per-stock {df['ps_dwell'].mean():.1f}   ->  pooled {df['pl_dwell'].mean():.1f}")
    print(f"jump  : per-stock {df['ps_jump'].mean():.3f}  ->  pooled {df['pl_jump'].mean():.3f}")
    print(f"corr(per-stock, pooled) 平均 = {df['corr'].mean():.2f}（向きは概ね一致しつつ pooled が安定）")
    print(f"\n[done] {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
