"""(a)ループ 段1：価格逆張り value 一次に レジーム を乗せる（単層 vs 無フィルタ・コスト後）。

射程限定：測るのは「**価格逆張り一次にレジーム検知が価値を足すか**」であって、レジーム一般の当否でない。
事前登録基準を機械適用：単層① 日足のみ−無フィルタ／単層② 週足のみ−無フィルタ、各 ≥2/3・中央値正・
DSR>0.95。超えた層があって初めて 段2（二層 vs 単層）。λ・状態数・閾値・サンプリング固定。検知器層不変。

交絡監視：逆張り（下落後に買う）× ボラ抑制フィルタ（下落=高ボラで削る）は構造的に逆を向きうる →
**フィルタが抑制した逆張りエントリー局面の割合**を出力（負けた時「無価値」か「方向不整合」かの切り分け）。

読込 2020（value 756d）・検知器/アンカー 2022.5・評価 2024+・銘柄ロック10。
実行：python examples/daily_regime_loopa_step1.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from invest_system.equities.fundamentals import load_fundamentals
from invest_system.research.data_view import AsOfView
from invest_system.research.engine import backtest
from invest_system.research.daily_regime import data as drdata
from invest_system.research.daily_regime import fundamentals_pit as fp
from invest_system.research.daily_regime import multiscale, pooling, tiers, weekly_anchor
from invest_system.research.daily_regime.asof_bridge import anchor_daily_panel
from invest_system.research.daily_regime.cost import regime_cost_panel
from invest_system.research.daily_regime.strategy import RegimeAwareStrategy, ValueReversalPrimary
from invest_system.validation.dsr import deflated_sharpe_ratio_from_returns, sharpe_ratio

LOAD_START, DET_START, EVAL_START, LAM = "2020-01-01", "2022-06-01", "2024-01-01", 2.0
FILTER_THRESH, SIZE_FLOOR, LOOKBACK = 0.6, 0.2, 756
KW = dict(n_states=2, refit_every=20, window=378, warmup=252, n_init=2, align_col=0,
          min_dwell=3, random_state=7, max_pool_codes=30)
LOCKED = [("23170", "5250", "T2"), ("135A0", "5250", "T3"), ("21240", "9050", "T2"),
          ("157A0", "9050", "T3"), ("26640", "6100", "T2"), ("26740", "6100", "T3"),
          ("27330", "6050", "T2"), ("26670", "6050", "T3"), ("65160", "3650", "T2"),
          ("38560", "3650", "T3")]


def build_feat(adj, va, code, vol_w=20, amh_w=10):
    ret = adj[code].pct_change(fill_method=None)
    rv = ret.rolling(vol_w).std()
    amh = (ret.abs() / va[code].where(va[code] > 0)).rolling(amh_w).mean()
    return pd.DataFrame({"rv": rv, "lamihud": np.log(amh.replace(0.0, np.nan))}).dropna()


def psr(res):
    r = res.returns.dropna()
    try:
        return sharpe_ratio(r)
    except ValueError:                       # 標本不足・ゼロ分散のみ NaN（他の例外は隠さない）
        return np.nan


def main():
    t0 = time.perf_counter()
    codes_all = None
    panels = drdata.load_daily_panels(start=LOAD_START)
    adj, va = panels.adj_close, panels.turnover
    s33 = drdata.load_s33_map()
    mem = tiers.membership_panel(
        tiers.assign_tiers(adj, va, n_tiers=3, adv_floor=5e7, smooth_window=60, refit_every=20))
    fund = load_fundamentals(codes=[c for c, _, _ in LOCKED])
    eps_panel = fp.eps_asof_panel(fund, adj.index, lag_days=1)
    fp.assert_fundamental_pit(eps_panel, fund, "EPS", lag_days=1)        # PIT ゲート（毎回）
    sectors = sorted({s for _, s, _ in LOCKED})
    sec_codes = [c for c in adj.columns if str(s33.get(c)) in sectors]
    real = weekly_anchor.build_weekly_anchor_hmm(
        adj.loc[DET_START:, sec_codes], s33[[c for c in s33.index if c in sec_codes]], n_states=3)
    print(f"[setup] adj {adj.shape} fund {fund.shape} anchor {len(real)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ティア別プールをループ前に1回だけ precompute（巨大 mem の再スライス回避）
    asof_row = mem.loc[:adj.index[-1]].iloc[-1]
    det_obs = adj.loc[DET_START:].notna().sum()
    tier_pool = {tier: sorted([c for c in mem.columns if asof_row.get(c) == tier
                               and int(det_obs.get(c, 0)) > 412])[:30] for tier in ("T2", "T3")}

    rows = []
    for code, sector, tier in LOCKED:
        pool = list(tier_pool[tier])
        if code not in pool:
            pool = [code] + pool[:29]
        feat = {c: build_feat(adj.loc[DET_START:], va.loc[DET_START:], c) for c in pool}
        feat = {c: f for c, f in feat.items() if len(f) > 300}
        if code not in feat:
            continue
        fidx = feat[code].index
        tilt = multiscale.weekly_log_tilt(real, fidx, sector, n_states=2, lam=LAM)
        multi = pooling.walk_forward_pooled_multi(code, feat, mem, {"d0": None, "dX": tilt}, **KW)
        d0, dX = multi["d0"]["prob_stressed"], multi["dX"]["prob_stressed"]

        didx = adj.index
        amh = (adj[code].pct_change(fill_method=None).abs() / va[code].where(va[code] > 0)).rolling(20).mean()
        cost = regime_cost_panel(pd.DataFrame({code: amh}), base_bps=10.0, impact_coef=20.0)
        wk = anchor_daily_panel(real, didx, "p_bear", sector)
        frames = {"close": adj[[code]], "eps_asof": eps_panel[[code]] if code in eps_panel.columns else adj[[code]] * np.nan,
                  "d0": pd.DataFrame({code: d0.reindex(didx)}), "dX": pd.DataFrame({code: dX.reindex(didx)}),
                  "wk": pd.DataFrame({code: wk.reindex(didx)})}
        view = AsOfView(frames)
        dates = view.dates[view.dates >= pd.Timestamp(EVAL_START)][:-2]

        rg = dict(filter_thresh=FILTER_THRESH, size_floor=SIZE_FLOOR)
        prim = lambda: ValueReversalPrimary(code, lookback=LOOKBACK)  # noqa: E731
        strat = {
            "nofilter": prim(),
            "weekly": RegimeAwareStrategy(prim(), regime_field="wk", **rg),
            "daily": RegimeAwareStrategy(prim(), regime_field="d0", **rg),
            "two": RegimeAwareStrategy(prim(), regime_field="dX", **rg),
        }
        res = {k: backtest(s, view, costs_bps=cost, execution_lag=1, rebalance=dates) for k, s in strat.items()}
        srs = {k: psr(v) for k, v in res.items()}
        sr_arr = np.array([srs[k] for k in ["nofilter", "weekly", "daily", "two"]])
        srvar = float(np.nanvar(sr_arr, ddof=1))

        def _dsr(name):
            try:
                return deflated_sharpe_ratio_from_returns(res[name].returns.dropna(), srvar, 4)
            except ValueError:               # 標本不足・PSR分散項の破綻のみ NaN（他の例外は隠さない）
                return np.nan

        # 抑制割合：value 逆張りが long シグナルの日のうち、フィルタ（d0≥thresh）が抑制する割合
        ep = (eps_panel[code] / adj[code]).dropna() if code in eps_panel.columns else pd.Series(dtype=float)
        long_sig = pd.Series(False, index=dates)
        for t in dates:
            e = ep.loc[:t]
            if len(e) >= 189:
                long_sig[t] = bool(e.iloc[-1] >= e.iloc[-LOOKBACK:].median())
        stressed = (d0.reindex(dates) >= FILTER_THRESH).fillna(False)
        suppress = float((long_sig & stressed).sum() / max(int(long_sig.sum()), 1))

        rows.append({"code": code, "sec": sector, "nofilter": srs["nofilter"], "daily": srs["daily"],
                     "weekly": srs["weekly"], "two": srs["two"],
                     "md_nf": srs["daily"] - srs["nofilter"], "mw_nf": srs["weekly"] - srs["nofilter"],
                     "dsr_daily": _dsr("daily"), "dsr_weekly": _dsr("weekly"), "suppress": suppress})
        print(f"  {code}({sector}): 単層①日足−NF={srs['daily']-srs['nofilter']:+.3f} "
              f"単層②週足−NF={srs['weekly']-srs['nofilter']:+.3f} 抑制割合={suppress:.2f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    print("\n===== 段1：逆張り value 一次 × レジーム（単層 vs 無フィルタ・10銘柄） =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    n = len(df)
    for col, dcol, label in [("md_nf", "dsr_daily", "単層① 日足のみ−無フィルタ"),
                             ("mw_nf", "dsr_weekly", "単層② 週足のみ−無フィルタ")]:
        pos = int((df[col] > 0).sum()); med = float(df[col].median())
        surv = int((df[dcol] > 0.95).sum())
        adopt = (pos / n >= 2 / 3) and (med > 0)
        print(f"\n{label}: 正 {pos}/{n} ({pos/n:.0%})・中央値 {med:+.3f}・DSR>0.95 {surv}/{n} → "
              f"採用={'YES' if adopt else 'NO'}", flush=True)
    print(f"\nフィルタ抑制割合（逆張り long 局面の何割をボラ抑制が削ったか）平均={df['suppress'].mean():.2f}", flush=True)
    print("[note] 射程＝逆張り戦略系限定。負けが抑制割合の高さと相関するなら『無価値』でなく『方向不整合』。", flush=True)
    print(f"[done] {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
