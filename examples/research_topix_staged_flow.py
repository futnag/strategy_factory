"""仮説検証：TOPIX 段階的ウエイト低減の決定論的フロー・リバーサル（docs/60）。

事前登録＝docs/60（scope=`topix_staged_flow`・K=5・実行前コミット d395e0d）。
公表スケジュール済みの機械的インデックス売り（四半期末×10段階・2022-10〜2025-01）は
実施日に一時的売り圧力→数週間でリバーサル（明田 2022）。コホート・ロング×サイズマッチ
統制・ショートの日次ドルニュートラル DiD で裁く。実効 n=10 実施日（red-team ①）。

グリッド（docs/60 §2.4 固定・5セル）:
  tsf_rev_w1_11（主）/ tsf_rev_w1_6 / tsf_rev_w1_16 / tsf_placebo_post2025 / tsf_rev_top300hedge

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_topix_staged_flow.py
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

from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.data.store import load_wide  # noqa: E402
from invest_system.equities.universe import filter_common_stocks  # noqa: E402
from invest_system.equities.margin import load_weekly_margin  # noqa: E402
from invest_system.equities.frictions import limit_lock_flags, shortable_mask  # noqa: E402
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, PrecomputedWeights, judge_grid, write_html,
)
from invest_system.research.engine import backtest  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

OOS = "2024-01"
SCOPE = "topix_staged_flow"
COST_BPS = 30.0        # 日次イベント床
BORROW_BPS = 300.0     # イベント/小型
HEDGE_N = 300
REEVAL = pd.Timestamp("2023-10-31")   # steps5以降は継続450コホート
# 実施日程（月末営業日・Jan/Apr/Jul/Oct・10段階）
IMPL_MONTHS = ["2022-10-31", "2023-01-31", "2023-04-30", "2023-07-31", "2023-10-31",
               "2024-01-31", "2024-04-30", "2024-07-31", "2024-10-31", "2025-01-31"]
# 時間プラセボ（スケジュール終了後の四半期末・強制フロー無し）
PLACEBO_MONTHS = ["2025-04-30", "2025-07-31", "2025-10-31", "2026-01-31"]


def _sr(x: pd.Series, ann: float = 252.0, lo=None, hi=None) -> float:
    r = x.dropna()
    if lo is not None:
        r = r[r.index >= pd.Timestamp(lo)]
    if hi is not None:
        r = r[r.index < pd.Timestamp(hi)]
    if r.size < 8 or float(r.std(ddof=1)) == 0.0:
        return float("nan")
    return float(sharpe_ratio(r) * np.sqrt(ann))


def snap(idx: pd.DatetimeIndex, months: list[str]) -> list[pd.Timestamp]:
    """各月末を「その月末以前の最終営業日」にスナップ。"""
    out = []
    for m in months:
        sub = idx[idx <= pd.Timestamp(m)]
        if len(sub):
            out.append(sub[-1])
    return out


def bd_window(events: list[tuple[pd.Timestamp, str]], idx: pd.DatetimeIndex,
              cols: pd.Index, lo_bd: int, hi_bd: int) -> pd.DataFrame:
    """(実施日, Code) → [impl+lo_bd, impl+hi_bd) 営業日窓の日次 bool wide。"""
    colpos = {c: i for i, c in enumerate(cols)}
    arr = np.zeros((len(idx), len(cols)), dtype=bool)
    for d, c in events:
        j = colpos.get(c)
        if j is None:
            continue
        p0 = idx.searchsorted(d)
        a, b = p0 + lo_bd, p0 + hi_bd
        if 0 <= a < len(idx) and b > a:
            arr[a:min(b, len(idx)), j] = True
    return pd.DataFrame(arr, index=idx, columns=cols)


def ls_weights(cohort_win: pd.DataFrame, hedge: pd.DataFrame) -> pd.DataFrame:
    """コホート・ロング(Σ=+1)×サイズマッチ統制EWショート(Σ=−1)。無イベント日＝現金。"""
    ev = cohort_win.fillna(False)
    ne = ev.sum(axis=1)
    active = ne > 0
    longw = ev.astype(float).div(ne.where(ne > 0), axis=0)
    hp = hedge.fillna(False) & (~ev)
    nh = hp.sum(axis=1)
    shortw = (-hp.astype(float)).div(nh.where(nh > 0), axis=0)
    w = longw.add(shortw, fill_value=0.0)
    w.loc[~active] = np.nan
    return w


def main() -> int:
    adj = load_wide("adj_close")
    turn = load_wide("turnover")
    close = load_wide("close")
    high = load_wide("high")
    low = load_wide("low")
    ul = load_wide("upper_limit")
    ll = load_wide("lower_limit")
    vol_ = load_wide("volume")
    listed = jq.fetch_listed_info().assign(Code=lambda d: d["Code"].astype(str))
    common = set(filter_common_stocks(listed)["Code"])
    cols = [c for c in adj.columns if str(c) in common]
    adj, turn, close, high, low, ul, ll, vol_ = (
        d.reindex(columns=cols) for d in (adj, turn, close, high, low, ul, ll, vol_))
    idx = adj.index
    colset = set(adj.columns)

    ev = pd.read_parquet("data/jpx_indices/topix_reduction_events.parquet")
    red = [c for c in ev[ev["action"] == "reduction_start"]["code"].astype(str).unique()]
    escaped = set(ev[ev["action"] == "reeval_escaped"]["code"].astype(str))
    cohort_all = [c for c in red if c in colset]
    cohort_cont = [c for c in cohort_all if c not in escaped]   # 継続450

    impl = snap(idx, IMPL_MONTHS)
    placebo = snap(idx, PLACEBO_MONTHS)
    print(f"=== TOPIX 段階削減リバーサル検証（scope={SCOPE}）===")
    print(f"実施日 {len(impl)}（{impl[0].date()}..{impl[-1].date()}）"
          f"／プラセボ {len(placebo)}／初期コホート {len(cohort_all)}・継続 {len(cohort_cont)}")

    # 実施日イベント: steps1-4=493, steps5-10（>=REEVAL）=継続450（復帰43除外・red-team ④）
    def impl_events(cohort_pre, cohort_post):
        out = []
        for d in impl:
            members = cohort_post if d >= REEVAL else cohort_pre
            out += [(d, c) for c in members]
        return out
    ev_impl = impl_events(cohort_all, cohort_cont)
    ev_plac = [(d, c) for d in placebo for c in cohort_cont]
    print(f"実施日イベント総数 {len(ev_impl)}（同時 ~{len(cohort_all)}銘柄×{len(impl)}日）"
          f"／プラセボ {len(ev_plac)}")

    # ユニバース・トレード可否
    traded = turn.notna() & (turn > 0)
    liq_hist = traded.rolling(252, min_periods=120).sum() >= 120
    eligible = traded & liq_hist & adj.notna()
    mw = load_weekly_margin()
    mw["Date"] = pd.to_datetime(mw["Date"]); mw["Code"] = mw["Code"].astype(str)
    shortable = shortable_mask(mw, idx).reindex(columns=cols).fillna(False)

    # サイズマッチ統制: 非コホート普通株で trailing-252 ADV がコホート ADV 帯 [0.5x,2x]（PIT）
    adv = turn.rolling(252, min_periods=120).mean()
    cohort_cols = [c for c in cohort_all if c in adv.columns]
    cohort_med = adv[cohort_cols].median(axis=1)
    lo_band = (0.5 * cohort_med)
    hi_band = (2.0 * cohort_med)
    noncohort = pd.Index([c for c in adj.columns if c not in set(cohort_all)])
    in_band = adv[noncohort].ge(lo_band, axis=0) & adv[noncohort].le(hi_band, axis=0)
    size_hedge = pd.DataFrame(False, index=idx, columns=adj.columns)
    size_hedge[noncohort] = in_band.fillna(False)
    size_hedge = size_hedge & eligible & shortable
    # top300 ヘッジ（cell5・サイズマッチの効きの識別）
    tadv = turn.rolling(252, min_periods=120).mean()
    rank = tadv.rank(axis=1, ascending=False)
    top300 = (rank <= HEDGE_N) & eligible & shortable
    print(f"サイズマッチ統制 平均 {size_hedge.sum(axis=1).mean():.0f}銘柄／"
          f"top300 平均 {top300.sum(axis=1).mean():.0f}銘柄")

    grids = [
        ("tsf_rev_w1_11", ev_impl, (1, 11), size_hedge),
        ("tsf_rev_w1_6", ev_impl, (1, 6), size_hedge),
        ("tsf_rev_w1_16", ev_impl, (1, 16), size_hedge),
        ("tsf_placebo_post2025", ev_plac, (1, 11), size_hedge),
        ("tsf_rev_top300hedge", ev_impl, (1, 11), top300),
    ]
    strategies = []
    for name, events, (lo, hi), hedge in grids:
        win = bd_window(events, idx, adj.columns, lo, hi) & eligible
        strategies.append(PrecomputedWeights(
            ls_weights(win, hedge), name=name,
            params={"n_impl_days": len(impl if events is ev_impl else placebo),
                    "window_bd": [lo, hi], "avg_names": float(win.sum(axis=1).mean())}))
        print(f"  {name:<22} 同時保有 平均={win.sum(axis=1).mean():.0f} "
              f"最大={int(win.sum(axis=1).max())}")

    no_buy, no_sell = limit_lock_flags(close, high, low, ul, ll, vol_)
    view = AsOfView({"close": adj})
    hyp = ("公表スケジュール済みの機械的インデックス売り（四半期末×10段階）は実施日に一時的"
           "売り圧力→数週間でコホートがサイズマッチ統制を上回る（リバーサル）。方向は明田2022で"
           "事前固定・実効n=10実施日。docs/60")
    rat = ("低減幅・日付・対象が事前公知のカレンダー付き強制フロー＝uninformed 純供給。指値供給者"
           "への一時プレミアムが数週間で巻き戻る。情報起点なし（H-17適合）。明田2022 実測方向。docs/60")

    with default_registry() as reg:
        v = judge_grid(strategies, view, scope=SCOPE, hypothesis=hyp,
                       economic_rationale=rat, registry=reg, costs_bps=COST_BPS,
                       execution_lag=1, adv=turn, participation=0.1,
                       no_buy=no_buy, no_sell=no_sell, short_borrow_bps=BORROW_BPS)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{SCOPE}.html"))

    # ---- 診断（throwaway・K不変）----
    print(f"\n--- 診断: IS/OOS({OOS}〜)・前後2020・maxDD ---")
    for r in v.results:
        s = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        cum = (1.0 + s).cumprod()
        mdd = float((cum / cum.cummax() - 1.0).min()) if len(s) else float("nan")
        print(f"  {r.name:<22} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={_sr(s, hi=OOS):+.2f} OOS={_sr(s, lo=OOS):+.2f} | maxDD={mdd:.1%} | n={s.size}")

    # 圧力→リバーサルのイベントスタディ形状（対サイズマッチ・impl日基準・営業日）
    ret_d = adj.pct_change(fill_method=None)
    hedge_ret = ret_d.where(size_hedge).mean(axis=1)
    ex_d = ret_d.sub(hedge_ret, axis=0)
    prof = np.zeros(20); cnt = np.zeros(20)
    for d, c in ev_impl:
        if c not in ex_d.columns:
            continue
        p0 = idx.searchsorted(d)
        for k in range(20):     # impl日-2 〜 impl+17bd を1bd刻み（index k=0 が impl-2）
            j = p0 - 2 + k
            if 0 <= j < len(idx):
                x = ex_d[c].iloc[j]
                if pd.notna(x):
                    prof[k] += float(x); cnt[k] += 1
    prof = np.where(cnt > 0, prof / np.maximum(cnt, 1), np.nan) * 1e4
    print("\n--- impl日基準の日次超過リターン（bps/日・対サイズマッチ・k=0がimpl-2bd）---")
    print("  相対bd: " + " ".join(f"{k-2:>4}" for k in range(0, 20, 2)))
    print("  bps/日: " + " ".join(f"{prof[k]:+4.0f}" for k in range(0, 20, 2)))
    print(f"  圧力(impl[-1,+1]累積)={np.nansum(prof[1:4]):+.0f}bps  "
          f"リバーサル([+1,+11)累積)={np.nansum(prof[3:13]):+.0f}bps")

    # ステップ横断の効果プロファイル（F6・前半/後半の減衰）
    print("\n--- ステップ別 [1,11) DiD リターン（F6 減衰診断）---")
    win_main = bd_window(ev_impl, idx, adj.columns, 1, 11) & eligible
    w_main = ls_weights(win_main, size_hedge)
    port = (w_main.shift(1) * ret_d).sum(axis=1)   # T+1 近似
    for i, d in enumerate(impl, 1):
        p0 = idx.searchsorted(d)
        seg = port.iloc[p0 + 1:p0 + 11].sum()
        print(f"  step{i:>2} {d.date()}  DiD累積={seg * 1e4:+5.0f}bps")

    # コスト/貸株感応（主セル）
    print(f"\n--- コスト感応（tsf_rev_w1_11・貸株{BORROW_BPS:.0f}bps）---")
    for c in [0, 15, 30, 50]:
        res = backtest(strategies[0], view, costs_bps=float(c), execution_lag=1,
                       adv=turn, participation=0.1, no_buy=no_buy, no_sell=no_sell,
                       short_borrow_bps=BORROW_BPS)
        print(f"  {c:>2}bps: net SR={_sr(res.returns):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
