"""research_governance_event_value — Stage-0 K=0 cheap-kill（事前登録 docs/63）.

仮説（in-window tail のみ）: TSE「資本コスト・株価を意識した経営」開示の**窓内 新規開示**
（2025-06+ の transition・主に Standard）が t+1 forward で value+size 残差ベースの正増分を持つか。
red-team 反映: 論文機構（2024 salience shock）は窓外＝本サイクルは tail の cheap-kill のみ。
value 残差化（月次 XS 回帰 fwd~[rank BM, rank logturnover]）× forward t+1（H-16）× placebo（H-18）。
K=0（judge_grid 不使用・registry 不参照）。予想は H-17/F6 で null。

ゲート（docs/63 §2.6）: value+size 残差の event 平均が (i) 正 (ii) placebo 外（片側p<0.05）でなければ KILL。

実行: $env:PYTHONUTF8="1"; .venv/Scripts/python.exe examples/research_governance_event_value.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

SEED = 20260705
NPLAC = 3000
MIN_EV = 10          # 月の最小イベント数
MIN_UNIV = 50
DECILES = 10
PANEL = "data/processed/equities/wide"
DISC = "data/tse_capital_disclosure/disclosure_panel.parquet"


def rank01(s):
    return s.rank(pct=True)


def resid_on(fwd, bm, tv):
    """fwd を [rank BM, rank log turnover] で XS 回帰した残差（value+size 中立）。"""
    df = pd.concat({"y": fwd, "bm": bm, "tv": tv}, axis=1).dropna()
    if len(df) < MIN_UNIV:
        return None
    X = np.column_stack([np.ones(len(df)), rank01(df["bm"]).values,
                         rank01(np.log(df["tv"])).values])
    beta, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None)
    return pd.Series(df["y"].values - X @ beta, index=df.index)


def main():
    from invest_system.data.store import load_wide
    from invest_system.equities.fundamentals import fundamentals_panel

    # ── イベント: 窓内 transition（前月非開示→当月開示済・初出）─────────────
    p = pd.read_parquet(DISC)
    p["mp"] = p["month"].dt.to_period("M")
    p = p.sort_values(["local_code", "mp"])
    p["prev"] = p.groupby("local_code")["disclosed"].shift(1)
    trans = p[(p["disclosed"]) & (p["prev"] == False)]  # noqa: E712
    events = trans.groupby("local_code")["mp"].min()     # 初出 transition 月
    ev_by_month = events.reset_index().groupby("mp")["local_code"].apply(set).to_dict()
    mkt = p.drop_duplicates("local_code").set_index("local_code")["market"]

    # ── 価格・turnover・BM ───────────────────────────────────────────────
    adj = load_wide("adj_close")
    close = load_wide("close")
    turn = load_wide("turnover")
    mclose_adj = adj.resample("ME").last()
    ret = mclose_adj.pct_change()
    ret.index = ret.index.to_period("M")
    me = pd.DatetimeIndex(adj.resample("ME").last().index)
    mturn = turn.resample("ME").mean()
    mturn.index = mturn.index.to_period("M")
    pit = fundamentals_panel(me, ["Eq", "ShOutFY"])
    bm = (pit["Eq"] / (pit["ShOutFY"] * close.reindex(me))).replace([np.inf, -np.inf], np.nan)
    bm.index = bm.index.to_period("M")

    # ── 月次ループ ───────────────────────────────────────────────────────
    months = sorted(m for m in ev_by_month if (m + 1) in ret.index and m in bm.index and m in mturn.index)
    rng = np.random.default_rng(SEED)
    rows, per, per_all = [], [], []
    for M in months:
        fwd = ret.loc[M + 1]
        bmv, tvv = bm.loc[M], mturn.loc[M]
        base = pd.concat({"y": fwd, "bm": bmv, "tv": tvv}, axis=1).dropna()
        if len(base) < MIN_UNIV:
            continue
        ev_full = [c for c in ev_by_month[M] if c in base.index]
        res_all = resid_on(fwd, bmv, tvv)   # 全 datable 月（pooled 診断用・§3.2 throwaway）
        if res_all is not None and len([c for c in ev_full if c in res_all.index]) >= 3:
            per_all.append((res_all, [c for c in ev_full if c in res_all.index]))
        ev = ev_full
        if len(ev) < MIN_EV:               # 主ゲートは breadth floor=MIN_EV の月のみ
            continue
        r = base["y"]
        raw = r.loc[ev].mean() - r.mean()
        # turnover 十分位中立
        dec = pd.qcut(base["tv"].rank(method="first"), DECILES, labels=False)
        evs = set(ev)
        num, den = 0.0, 0
        for dl in range(DECILES):
            idx = dec.index[dec == dl]
            e = [c for c in idx if c in evs]
            ne = [c for c in idx if c not in evs]
            if e and ne:
                num += len(e) * (r.loc[e].mean() - r.loc[ne].mean())
                den += len(e)
        tn = num / den if den else np.nan
        # value+size 残差（本命）
        res = resid_on(fwd, bmv, tvv)
        ev_res = res.loc[[c for c in ev if c in res.index]].mean() if res is not None else np.nan
        n_std = sum(1 for c in ev if mkt.get(c) == "Standard")
        rows.append({"month": str(M), "n_ev": len(ev), "n_std": n_std, "n_univ": len(base),
                     "raw": raw, "turn_neutral": tn, "val_size_resid": ev_res})
        if res is not None:
            per.append((res, [c for c in ev if c in res.index]))
    out = pd.DataFrame(rows)

    # placebo null（value+size 残差の event 平均・同数ランダム）
    plac = np.empty(NPLAC)
    for k in range(NPLAC):
        s = 0.0
        for res, ev in per:
            idx = res.index.values
            s += res.iloc[rng.integers(0, len(idx), len(ev))].mean()
        plac[k] = s / len(per)
    m_res = out["val_size_resid"].mean()
    p_val = float((plac >= m_res).mean())

    # 診断（throwaway・§3.2）: pooled event-level（全 datable transition・月クラスタ保存の power チェック）
    pooled_ev = float(np.mean([res.loc[ev].mean() for res, ev in per_all]))  # 月平均→イベント数で加重
    pool_events = [res.loc[ev] for res, ev in per_all]
    n_pool = int(sum(len(e) for e in pool_events))
    pooled_ew = float(pd.concat(pool_events).mean())      # 全イベント等加重（月クラスタ無視）
    plac_pool = np.empty(NPLAC)
    for k in range(NPLAC):
        s = [res.iloc[rng.integers(0, len(res), len(ev))].mean() for res, ev in per_all]
        plac_pool[k] = float(np.mean(s))
    p_pool = float((plac_pool >= pooled_ev).mean())

    def ann_ir(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        return np.nan if len(x) < 2 or x.std(ddof=1) == 0 else x.mean() / x.std(ddof=1) * np.sqrt(12)

    print("=== governance_event_value  Stage-0 cheap-kill (docs/63・K=0) ===")
    print(f"窓内 transition イベント総数: {int(events.gt(events.min()-1).sum())}  "
          f"（除外=左側打切り 2025-05 開示済）  Standard比率 {100*out['n_std'].sum()/max(out['n_ev'].sum(),1):.0f}%")
    print(out.round(4).to_string(index=False))
    print(f"\n平均 生 spread          : {out['raw'].mean():+.4%}/月  IR≈{ann_ir(out['raw']):+.2f}")
    print(f"平均 turnover中立 spread : {out['turn_neutral'].mean():+.4%}/月  IR≈{ann_ir(out['turn_neutral']):+.2f}")
    print(f"平均 value+size 残差(本命): {m_res:+.4%}/月  IR≈{ann_ir(out['val_size_resid']):+.2f}  "
          f"hit {100*(out['val_size_resid']>0).mean():.0f}%")
    print(f"プラセボ null: mean {plac.mean():+.4%}  sd {plac.std():.4%}  p95 {np.percentile(plac,95):+.4%}")
    print(f"  → value+size 残差の片側 p値: {p_val:.3f}")
    print(f"[診断] pooled({len(per_all)}月/{n_pool}件・§3.2 throwaway): 月加重残差 {pooled_ev:+.4%} (片側p {p_pool:.3f})・"
          f"全イベント等加重 {pooled_ew:+.4%}")
    g_i, g_ii = m_res > 0, p_val < 0.05
    print(f"\nゲート (i) 残差>0        : {'PASS' if g_i else 'FAIL'} ({m_res:+.4%})")
    print(f"ゲート (ii) placebo外(p<.05): {'PASS' if g_ii else 'FAIL'} (p={p_val:.3f})")
    print(f"\n判定: {'SURVIVE（想定外→別サイクルでjudge＋D-7優先）' if (g_i and g_ii) else 'KILL (K=0・tail条件付きFAIL→⏸ D-7待ち)'}")


if __name__ == "__main__":
    main()
