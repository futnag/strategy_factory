"""中核判定（多銘柄・公正サンプル）：本丸①②を層別に・符号一貫性・中央値・DSR ＋ PEAD 除外効果。

事前登録（regime_detection_core_judgment_preregistration.md）に**機械的に**当てる：
  - 主指標＝コスト後 Sharpe（per-period, DSR 用）。本丸①=二層−日足のみ／本丸②=二層−週足のみ。
  - サンプリングはロック済み（9999除外・上位5業種×T2/T3各1・コード昇順・全数）。
  - 採用：本丸①②それぞれ ≥2/3 で符号正・中央値正（+DSR 割引）。**層別**に採否。
  - PEAD は同一窓・同一リスト。除外あり/なしの Sharpe 差で §2 のコスト後価値を読む。
  - λ・状態数・閾値・refit/窓は固定。

実行：python examples/daily_regime_core_judgment_multi.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from invest_system.research.data_view import AsOfView
from invest_system.research.engine import backtest
from invest_system.research.daily_regime import backtest as drbt
from invest_system.research.daily_regime import data as drdata
from invest_system.research.daily_regime import event_mask, multiscale, pooling, tiers, weekly_anchor
from invest_system.research.daily_regime.asof_bridge import anchor_daily_panel
from invest_system.research.daily_regime.cost import regime_cost_panel
from invest_system.research.daily_regime.strategy import PostEarningsDrift, RegimeAwareStrategy
from invest_system.validation.dsr import deflated_sharpe_ratio_from_returns, sharpe_ratio

START, LAM = "2023-01-01", 2.0
TOP_SECTORS, MIN_CONSTITUENTS, MIN_OBS = 5, 20, 412
KW = dict(n_states=2, refit_every=20, window=378, warmup=252, n_init=2, align_col=0,
          min_dwell=3, random_state=7, max_pool_codes=30)


def build_feat(adj, va, code, vol_w=20, amh_w=10):
    ret = adj[code].pct_change(fill_method=None)
    rv = ret.rolling(vol_w).std()
    amh = (ret.abs() / va[code].where(va[code] > 0)).rolling(amh_w).mean()
    return pd.DataFrame({"rv": rv, "lamihud": np.log(amh.replace(0.0, np.nan))}).dropna()


def psr(res):
    r = res.returns.dropna()
    try:
        return sharpe_ratio(r)
    except Exception:
        return np.nan


def main():
    t0 = time.perf_counter()
    panels = drdata.load_daily_panels(start=START)
    adj, va = panels.adj_close, panels.turnover
    s33 = drdata.load_s33_map()
    mem = tiers.membership_panel(
        tiers.assign_tiers(adj, va, n_tiers=3, adv_floor=5e7, smooth_window=60, refit_every=20))
    asof = mem.loc[:adj.index[-1]].iloc[-1]

    # --- サンプリング・ロック（結果非依存） ---
    counts = s33[[c for c in s33.index if c in mem.columns]].value_counts()
    qual = [str(s) for s in counts.index if str(s) != "9999" and counts[s] >= MIN_CONSTITUENTS][:TOP_SECTORS]
    selected = []
    for sec in qual:
        sec_codes = [c for c in mem.columns if str(s33.get(c)) == sec and adj[c].notna().sum() > MIN_OBS]
        for tier in ("T2", "T3"):
            tc = sorted([c for c in sec_codes if asof.get(c) == tier])
            if tc:
                selected.append((tc[0], sec, tier))
    print(f"[lock] sectors={qual}", flush=True)
    print(f"[lock] {len(selected)} 銘柄: {[(c, s, t) for c, s, t in selected]} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    sectors = sorted({s for _, s, _ in selected})
    sec_codes_all = [c for c in adj.columns if str(s33.get(c)) in sectors]
    real = weekly_anchor.build_weekly_anchor_hmm(
        adj[sec_codes_all], s33[[c for c in s33.index if c in sec_codes_all]], n_states=3)
    print(f"[anchor] real rows={len(real)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    try:
        disc = event_mask.load_disc_dates(codes=[c for c, _, _ in selected])
        have_pead = len(disc) > 0
    except Exception as exc:  # noqa: BLE001
        print(f"[pead] DiscDate 読込不可: {exc} → PEAD スキップ", flush=True)
        disc, have_pead = None, False

    rows = []
    for code, sector, tier in selected:
        pool = sorted([c for c in mem.columns if asof.get(c) == tier and adj[c].notna().sum() > MIN_OBS])[:30]
        if code not in pool:
            pool = [code] + pool[:29]
        feat = {c: build_feat(adj, va, c) for c in pool}
        feat = {c: f for c, f in feat.items() if len(f) > MIN_OBS}
        if code not in feat:
            continue
        fidx = feat[code].index
        tilt = multiscale.weekly_log_tilt(real, fidx, sector, n_states=2, lam=LAM)
        multi = pooling.walk_forward_pooled_multi(code, feat, mem, {"d0": None, "dX": tilt}, **KW)
        d0, dX = multi["d0"]["prob_stressed"], multi["dX"]["prob_stressed"]
        ret = adj[code].pct_change(fill_method=None)
        amh = (ret.abs() / va[code].where(va[code] > 0)).rolling(20).mean().reindex(fidx)
        wk = anchor_daily_panel(real, fidx, "p_bear", sector).reindex(fidx)
        res = drbt.run_value_baselines(code, adj[[code]].reindex(fidx), d0, dX, wk, amh,
                                       lookback=60, filter_thresh=0.6, size_floor=0.2,
                                       base_bps=10.0, impact_coef=20.0, execution_lag=1, warmup=252)
        srs = {k: psr(v[0]) for k, v in res.items()}
        m1 = srs["two-layer"] - srs["daily-only"]      # 本丸①（週足層）
        m2 = srs["two-layer"] - srs["weekly-only"]     # 本丸②（日足層）
        sr_arr = np.array([srs[k] for k in ["B&H", "no-filter", "weekly-only", "daily-only", "two-layer"]])
        srvar = float(np.nanvar(sr_arr, ddof=1))

        def _dsr(name):
            try:
                return deflated_sharpe_ratio_from_returns(res[name][0].returns.dropna(), srvar, 5)
            except Exception:
                return np.nan
        dsr_two, dsr_daily, dsr_weekly = _dsr("two-layer"), _dsr("daily-only"), _dsr("weekly-only")
        md_nf = srs["daily-only"] - srs["no-filter"]      # (b) 単層①：日足のみ−無フィルタ
        mw_nf = srs["weekly-only"] - srs["no-filter"]     # (b) 単層②：週足のみ−無フィルタ

        pead_diff = np.nan
        if have_pead:
            ew = event_mask.earnings_window_flag(disc[disc["Code"] == code], fidx, codes=[code], window=5)
            view = AsOfView({"close": adj[[code]].reindex(fidx), "prob_stressed": pd.DataFrame({code: d0}),
                             "earn_window": ew})
            costs = regime_cost_panel(pd.DataFrame({code: amh}), base_bps=10.0, impact_coef=20.0)
            dts = view.dates[252:-2]
            ex = backtest(RegimeAwareStrategy(PostEarningsDrift(code), event_driven=True,
                                              regime_field="prob_stressed"), view, costs_bps=costs,
                          execution_lag=1, rebalance=dts)
            nx = backtest(RegimeAwareStrategy(PostEarningsDrift(code), event_driven=False,
                                              regime_field="prob_stressed"), view, costs_bps=costs,
                          execution_lag=1, rebalance=dts)
            pead_diff = psr(ex) - psr(nx)

        rows.append({"code": code, "sec": sector, "tier": tier, "nofilter": srs["no-filter"],
                     "two": srs["two-layer"], "daily": srs["daily-only"], "weekly": srs["weekly-only"],
                     "m1_週足層": m1, "m2_日足層": m2, "md_nf": md_nf, "mw_nf": mw_nf,
                     "dsr_two": dsr_two, "dsr_daily": dsr_daily, "dsr_weekly": dsr_weekly,
                     "pead_exempt_diff": pead_diff})
        print(f"  {code}({sector}/{tier}): 本丸①={m1:+.3f} 本丸②={m2:+.3f} | "
              f"単層①日足−NF={md_nf:+.3f} 単層②週足−NF={mw_nf:+.3f} | "
              f"pead={pead_diff:+.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    print("\n===== 中核判定（多銘柄・公正サンプル・事前登録基準を機械適用） =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    n = len(df)
    for col, label in [("m1_週足層", "本丸① 週足層(二層−日足のみ)"), ("m2_日足層", "本丸② 日足層(二層−週足のみ)")]:
        pos = int((df[col] > 0).sum())
        med = float(df[col].median())
        adopt = (pos / n >= 2 / 3) and (med > 0)
        print(f"\n{label}: 正符号 {pos}/{n} ({pos/n:.0%})・中央値 {med:+.3f} → "
              f"採用={'YES' if adopt else 'NO'}（基準 ≥2/3 かつ 中央値>0）", flush=True)
    print(f"\nDSR(two-layer)>0.95 生存: {int((df['dsr_two'] > 0.95).sum())}/{n}", flush=True)

    # (b) 単層 vs 無フィルタ（レジーム全体の採否・事前登録：≥2/3・中央値正・DSR(単層)>0.95）
    print("\n--- (b) 単層 vs 無フィルタ（レジーム全体の採否） ---", flush=True)
    adopts = {}
    for col, dcol, label in [("md_nf", "dsr_daily", "単層① 日足のみ−無フィルタ"),
                             ("mw_nf", "dsr_weekly", "単層② 週足のみ−無フィルタ")]:
        pos = int((df[col] > 0).sum()); med = float(df[col].median())
        surv = int((df[dcol] > 0.95).sum())
        adopt = (pos / n >= 2 / 3) and (med > 0)
        adopts[label] = adopt
        print(f"{label}: 正 {pos}/{n} ({pos/n:.0%})・中央値 {med:+.3f}・DSR>0.95 {surv}/{n} → "
              f"採用={'YES' if adopt else 'NO'}", flush=True)
    if not any(adopts.values()):
        print(">>> 着地：どの単層も無フィルタを超えず → レジーム検知そのものを棄却"
              "（無フィルタ/B&H が本サンプル・本一次戦略の正しい着地）", flush=True)
    else:
        print(f">>> 着地：二層棄却・単層採用あり {adopts}", flush=True)

    if df["pead_exempt_diff"].notna().any():
        pe = df["pead_exempt_diff"].dropna()
        print(f"PEAD §2除外効果(exempt−no-exempt) Sharpe差: 平均 {pe.mean():+.3f}・正 {int((pe>0).sum())}/{len(pe)}", flush=True)
    print(f"\n[done] {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
