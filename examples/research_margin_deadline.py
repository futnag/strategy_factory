"""仮説検証：制度信用6ヶ月期日のコホート供給（margin_deadline・docs/58）。

事前登録＝docs/58（scope=`margin_deadline`・K=4・実行前コミット）。
週次 LongStdVol（制度信用買残）の急増＝コホート形成から約26週後の期日窓で、対象銘柄の
機械的な決済売り供給によるアンダーパフォームを、コホート・ショート×流動ヘッジ・ロングの
日次ドルニュートラルで裁く。プラセボ窓（12-16週）で「急増後の一般リバーサル（H-16）」と識別。

グリッド（docs/58 §2.4 固定・4セル）:
  mdl_short_w24_28（主）/ mdl_short_w25_27 / mdl_short_w24_28_big / mdl_placebo_w12_16

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_margin_deadline.py
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
SCOPE = "margin_deadline"
HEDGE_N = 300
COST_BPS = 30.0
BORROW_BPS = 300.0
SURGE_PCT = 0.20        # Δ買残 ≥ 20% × 前週残高
SURGE_ADV = 1.0         # かつ ≥ 1.0 × 60日平均出来高（株）
SURGE_ADV_BIG = 2.0
WIN_MAIN = (168, 196)   # 24-28週（暦日）
WIN_TIGHT = (175, 189)  # 25-27週
WIN_PLACEBO = (84, 112)  # 12-16週


def _sr(x: pd.Series, ann: float = 252.0, lo=None, hi=None) -> float:
    r = x.dropna()
    if lo is not None:
        r = r[r.index >= pd.Timestamp(lo)]
    if hi is not None:
        r = r[r.index < pd.Timestamp(hi)]
    if r.size < 8 or float(r.std(ddof=1)) == 0.0:
        return float("nan")
    return float(sharpe_ratio(r) * np.sqrt(ann))


def cohort_window(events: pd.DataFrame, idx: pd.DatetimeIndex, cols: pd.Index,
                  lo_days: int, hi_days: int) -> pd.DataFrame:
    """急増イベント（週金曜, Code）→ [date+lo, date+hi) 暦日窓の日次 bool wide。"""
    colpos = {c: i for i, c in enumerate(cols)}
    arr = np.zeros((len(idx), len(cols)), dtype=bool)
    lo = pd.Timedelta(days=lo_days)
    hi = pd.Timedelta(days=hi_days)
    for d, c in zip(events["Date"], events["Code"]):
        j = colpos.get(c)
        if j is None:
            continue
        p0 = idx.searchsorted(d + lo)
        p1 = idx.searchsorted(d + hi)
        if p0 < len(idx) and p1 > p0:
            arr[p0:p1, j] = True
    return pd.DataFrame(arr, index=idx, columns=cols)


def ls_weights(cohort_win: pd.DataFrame, hedge: pd.DataFrame) -> pd.DataFrame:
    """コホート・ショート(Σ=−1)×流動ヘッジEWロング(Σ=+1)。無イベント日＝現金。"""
    ev = cohort_win.fillna(False)
    ne = ev.sum(axis=1)
    active = ne > 0
    shortw = (-ev.astype(float)).div(ne.where(ne > 0), axis=0)
    hp = hedge.fillna(False) & (~ev)
    nh = hp.sum(axis=1)
    longw = hp.astype(float).div(nh.where(nh > 0), axis=0)
    w = shortw.add(longw, fill_value=0.0)
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

    mw = load_weekly_margin()
    mw["Date"] = pd.to_datetime(mw["Date"])
    mw["Code"] = mw["Code"].astype(str)
    long_std = mw.pivot_table(index="Date", columns="Code", values="LongStdVol",
                              aggfunc="last").reindex(columns=cols)
    print(f"=== 信用期日コホート検証（scope={SCOPE}）===")
    print(f"margin_weekly: {long_std.index.min():%Y-%m-%d}..{long_std.index.max():%Y-%m-%d}"
          f"  週={len(long_std)}  銘柄={long_std.notna().any().sum()}")

    # --- 急増（コホート形成）検知＝docs/58 §2.4 の事前固定定義 ---
    adv_sh = vol_.rolling(60, min_periods=30).mean()
    adv_sh_w = adv_sh.reindex(long_std.index, method="ffill")
    prev = long_std.shift(1)
    d_long = long_std - prev
    surge = (prev > 0) & (d_long >= SURGE_PCT * prev) & (d_long >= SURGE_ADV * adv_sh_w)
    surge_big = (prev > 0) & (d_long >= SURGE_PCT * prev) & (d_long >= SURGE_ADV_BIG * adv_sh_w)

    def to_events(mask: pd.DataFrame) -> pd.DataFrame:
        s = mask.stack()
        s = s[s]
        return s.reset_index().rename(columns={"level_0": "Date", "level_1": "Code"})[
            ["Date", "Code"]]

    ev_std = to_events(surge)
    ev_big = to_events(surge_big)
    print(f"急増イベント: 標準={len(ev_std):,}  大型={len(ev_big):,}")

    # --- ユニバース・ヘッジ・ショート可否 ---
    traded = turn.notna() & (turn > 0)
    liq_hist = traded.rolling(252, min_periods=120).sum() >= 120
    eligible = traded & liq_hist & adj.notna()
    tadv = turn.rolling(252, min_periods=120).mean()
    rank = tadv.rank(axis=1, ascending=False)
    hedge = (rank <= HEDGE_N) & eligible
    shortable = shortable_mask(mw, idx).reindex(columns=cols).fillna(False)

    grids = [
        ("mdl_short_w24_28", ev_std, WIN_MAIN),
        ("mdl_short_w25_27", ev_std, WIN_TIGHT),
        ("mdl_short_w24_28_big", ev_big, WIN_MAIN),
        ("mdl_placebo_w12_16", ev_std, WIN_PLACEBO),
    ]
    strategies = []
    wins = {}
    for name, ev, (lo, hi) in grids:
        win = cohort_window(ev, idx, adj.columns, lo, hi) & eligible & shortable
        wins[name] = win
        strategies.append(PrecomputedWeights(
            ls_weights(win, hedge), name=name,
            params={"n_events": int(len(ev)), "window_days": [lo, hi],
                    "avg_names": float(win.sum(axis=1).mean())}))
        print(f"  {name:<22} 同時保有 平均={win.sum(axis=1).mean():.0f} "
              f"最大={int(win.sum(axis=1).max())}")

    no_buy, no_sell = limit_lock_flags(close, high, low, ul, ll, vol_)
    view = AsOfView({"close": adj})

    hyp = ("制度信用買残の急増コホートは約26週後の期日窓（24-28週）で機械的な決済売り供給により"
           "アンダーパフォームし、効果は期日窓に局在する（プラセボ12-16週窓では現れない）。"
           "docs/58・方向は制度ルールで事前固定")
    rat = ("制度信用は6ヶ月以内の強制決済＝買残急増は期日カレンダー付きの将来売り供給を形成。"
           "信用買い手は uninformed 追随者（IRFA2016）＝決済は情報を持たない純供給。開示イベントが"
           "存在しないためジャンプの起点が無い（H-17適合）。学術未検証の白地。docs/58 §2.2")

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
        (_, pre), (_, post) = pre_post_sharpe(s, "2020-01-01")
        cum = (1.0 + s).cumprod()
        mdd = float((cum / cum.cummax() - 1.0).min())
        print(f"  {r.name:<22} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={_sr(s, hi=OOS):+.2f} OOS={_sr(s, lo=OOS):+.2f} | "
              f"前/後2020={pre:+.2f}/{post:+.2f} | maxDD={mdd:.1%}")

    best = v.best.name if v.best else "mdl_short_w24_28"
    sb = v.series.get(best, pd.Series(dtype="float64")).dropna()
    print(f"\n--- 年次 net SR（{best}）---")
    for y, seg in sb.groupby(sb.index.year):
        print(f"  {y}: SR={_sr(seg):+.2f}  n={seg.size}")

    # 急増後 0-30週の週次超過リターン・プロファイル（対ヘッジ・記述＝期日局在の可視化）
    ret_d = adj.pct_change(fill_method=None)
    hedge_ret = ret_d.where(hedge).mean(axis=1)
    ex_d = ret_d.sub(hedge_ret, axis=0)
    prof = np.zeros(30)
    cnt = np.zeros(30)
    for d, c in zip(ev_std["Date"], ev_std["Code"]):
        if c not in ex_d.columns:
            continue
        p0 = idx.searchsorted(d)
        for wk in range(30):
            a, b = p0 + wk * 5, p0 + (wk + 1) * 5
            if b >= len(idx):
                break
            x = ex_d[c].iloc[a:b].sum()
            if pd.notna(x):
                prof[wk] += float(x)
                cnt[wk] += 1
    prof = np.where(cnt > 0, prof / np.maximum(cnt, 1), np.nan) * 1e4
    print("\n--- 急増後の週次超過リターン・プロファイル（bps/週・対ヘッジ）---")
    print("  週:  " + " ".join(f"{w:>4}" for w in range(0, 30, 2)))
    print("  bps: " + " ".join(f"{prof[w]:+4.0f}" for w in range(0, 30, 2)))
    print(f"  期日窓(24-28週)平均={np.nanmean(prof[24:28]):+.0f}bps/週  "
          f"プラセボ窓(12-16週)平均={np.nanmean(prof[12:16]):+.0f}bps/週")

    # H-16: コホート銘柄の momentum 分位（窓開始時点）
    mom12 = (adj.shift(21) / adj.shift(252) - 1.0)
    pct = mom12.rank(axis=1, pct=True)
    vals = []
    for d, c in zip(ev_std["Date"], ev_std["Code"]):
        p = idx.searchsorted(d + pd.Timedelta(days=WIN_MAIN[0]))
        if p < len(idx) and c in pct.columns:
            x = pct.iloc[p][c]
            if pd.notna(x):
                vals.append(float(x))
    if vals:
        print(f"\n--- H-16 診断: 期日窓開始時点のコホート momentum 分位 中央値={np.median(vals):.2f} ---")

    # コスト/貸株感応（主セル）
    st = strategies[0]
    print(f"\n--- コスト感応（mdl_short_w24_28・貸株{BORROW_BPS:.0f}bps）---")
    for c in [0, 15, 30, 50]:
        res = backtest(st, view, costs_bps=float(c), execution_lag=1, adv=turn,
                       participation=0.1, no_buy=no_buy, no_sell=no_sell,
                       short_borrow_bps=BORROW_BPS)
        print(f"  {c:>3}bps  net年率SR={_sr(res.returns.dropna()):+.2f}  "
              f"回転={res.turnover.mean():.3f}")
    print(f"--- 貸株感応（同・コスト{COST_BPS:.0f}bps）---")
    for b in [0, 115, 300, 500, 1000]:
        res = backtest(st, view, costs_bps=COST_BPS, execution_lag=1, adv=turn,
                       participation=0.1, no_buy=no_buy, no_sell=no_sell,
                       short_borrow_bps=float(b))
        print(f"  {b:>5}bps  net年率SR={_sr(res.returns.dropna()):+.2f}")

    print("\n※ 判定は scope=margin_deadline の DSR（K=4・docs/58 §2.6）。診断は throwaway（K不変）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
