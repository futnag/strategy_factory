"""仮説検証：流動性「変化」への過小反応（ΔLiquidity・日本主標本 PBFJ2023・docs/56）。

事前登録＝docs/56（scope=`delta_liquidity`・K=4・実行前コミット）。
L=−log(Amihud) の銘柄固有変化について、短期（1ヶ月差）は正・長期（12ヶ月差）は負の
翌月ドリフトを、残差化（vs 直近1Mリターン・L水準・log ADV）セルを主セルとして裁く。
方向は Iwanaga & Hirose (2023, PBFJ 81) の成分別報告で事前固定。

グリッド（docs/56 §2.4 固定・4セル）:
  dliq_st_q20 / dliq_st_resid_q20(H1主) / dliq_lt_q20 / dliq_lt_resid_q20(H2主)

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_delta_liquidity.py
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
from invest_system.equities.universe import filter_common_stocks, point_in_time_universe  # noqa: E402
from invest_system.equities.factors import cross_sectional_residualize  # noqa: E402
from invest_system.equities.frictions import limit_lock_flags  # noqa: E402
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.research.engine import backtest  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

OOS = "2024-01"
SCOPE = "delta_liquidity"
TOP_N = 500
COST_BPS = 15.0
BORROW_BPS = 115.0
TURN_FLOOR = 1e6  # Amihud 分母の下限（¥1百万・ゼロ割り/外れ値防止）


def _sr(x: pd.Series, lo=None, hi=None) -> float:
    r = x.dropna()
    if lo is not None:
        r = r[r.index >= pd.Timestamp(lo)]
    if hi is not None:
        r = r[r.index < pd.Timestamp(hi)]
    if r.size < 8 or float(r.std(ddof=1)) == 0.0:
        return float("nan")
    return float(sharpe_ratio(r) * np.sqrt(12))


def _rank_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    cs = []
    for t in a.index.intersection(b.index):
        both = pd.concat([a.loc[t], b.loc[t]], axis=1).dropna()
        if len(both) >= 30:
            cs.append(float(both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank())))
    return float(np.mean(cs)) if cs else float("nan")


def main() -> int:
    adj = load_wide("adj_close")
    opn = load_wide("adj_open")
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
    adj, opn, turn, close, high, low, ul, ll, vol_ = (
        d.reindex(columns=cols) for d in (adj, opn, turn, close, high, low, ul, ll, vol_))
    idx = adj.index
    print(f"=== ΔLiquidity 検証（{idx.min():%Y-%m}〜{idx.max():%Y-%m}・普通株{len(cols)}・"
          f"scope={SCOPE}）===")

    # --- 月次 Amihud → L = −log(A)（≤t 情報のみ）---
    ret = adj.pct_change(fill_method=None)
    illiq_d = ret.abs() / turn.clip(lower=TURN_FLOOR)
    per = idx.to_period("M")
    me = pd.DatetimeIndex(pd.Series(idx, index=idx).groupby(per).max().values)
    A = illiq_d.groupby(per).mean().set_axis(me)
    L = -np.log(A.where(A > 0))

    # --- 固有化（市場共通成分の除去＝各月の XS 平均を控除）と変化 ---
    def demean(df: pd.DataFrame) -> pd.DataFrame:
        return df.sub(df.mean(axis=1), axis=0)

    d_st = demean(L - L.shift(1))
    d_lt = demean(L - L.shift(12))

    # --- ユニバースとコントロール ---
    turn_m = turn.groupby(per).median().set_axis(me)
    uni = point_in_time_universe(turn_m, top_n=TOP_N, lookback=12, min_obs=6)
    adj_me = adj.reindex(me)
    ret1m = adj_me.pct_change(fill_method=None)
    tadv = turn.rolling(252, min_periods=120).mean().reindex(me)
    controls = [ret1m.where(uni), L.where(uni), np.log(tadv.where(uni))]

    sig_st = d_st.where(uni)
    sig_lt = (-d_lt).where(uni)
    sig_st_r = cross_sectional_residualize(sig_st, controls)
    sig_lt_r = cross_sectional_residualize(sig_lt, controls)

    valid = sig_lt.notna().sum(axis=1) >= 100
    print(f"シグナル有効月: {int(valid.sum())}/{len(me)}  開始={sig_lt.index[valid.argmax()]:%Y-%m}")

    strategies = [
        CrossSectionalStrategy(sig_st[valid], 0.2, name="dliq_st_q20"),
        CrossSectionalStrategy(sig_st_r[valid], 0.2, name="dliq_st_resid_q20"),
        CrossSectionalStrategy(sig_lt[valid], 0.2, name="dliq_lt_q20"),
        CrossSectionalStrategy(sig_lt_r[valid], 0.2, name="dliq_lt_resid_q20"),
    ]

    fill_px = opn.bfill(limit=3).shift(-1).reindex(me)
    view = AsOfView({"close": fill_px})
    no_buy_d, no_sell_d = limit_lock_flags(close, high, low, ul, ll, vol_)
    no_buy, no_sell = no_buy_d.reindex(me), no_sell_d.reindex(me)

    hyp = ("流動性の銘柄固有『変化』への過小反応: 短期(1M)変化は翌月正・長期(12M)変化は翌月負の"
           "ドリフト（日本主標本 PBFJ2023 の成分別報告で方向を事前固定・docs/56）")
    rat = ("流動性が変わると非流動性プレミアムが変わるが再価格付けは即時に完結しない。"
           "短期変化＝過小反応ドリフト・長期変化＝要求リターン水準の変化（期待リターン・チャネル）。"
           "静的な流動性水準・リバーサルとは残差化で区別（docs/56 §2.4）")

    with default_registry() as reg:
        v = judge_grid(strategies, view, scope=SCOPE, hypothesis=hyp,
                       economic_rationale=rat, registry=reg, costs_bps=COST_BPS,
                       adv=tadv, participation=0.1, no_buy=no_buy, no_sell=no_sell,
                       short_borrow_bps=BORROW_BPS)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{SCOPE}.html"))

    # ---- 診断（throwaway・K不変）----
    print(f"\n--- 診断: IS/OOS({OOS}〜)・前後2020・maxDD ---")
    for r in v.results:
        s = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        (_, pre), (_, post) = pre_post_sharpe(s, "2020-01-01")
        cum = (1.0 + s).cumprod()
        mdd = float((cum / cum.cummax() - 1.0).min())
        print(f"  {r.name:<20} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={_sr(s, hi=OOS):+.2f} OOS={_sr(s, lo=OOS):+.2f} | "
              f"前/後2020={pre:+.2f}/{post:+.2f} | maxDD={mdd:.1%}")

    print("\n--- 独立性（生シグナルの平均XS順位相関）---")
    mom = adj_me.shift(1) / adj_me.shift(12) - 1.0
    for label, sig in [("ΔL_ST", sig_st[valid]), ("−ΔL_LT", sig_lt[valid])]:
        print(f"  {label}: vs 直近1Mリターン ρ̄={_rank_corr(sig, ret1m.where(uni)):+.3f} / "
              f"L水準 ρ̄={_rank_corr(sig, L.where(uni)):+.3f} / "
              f"momentum ρ̄={_rank_corr(sig, mom.where(uni)):+.3f}")

    # 分位単調性（主セルの記述診断・fill→fill 翌月リターン）
    fwd = fill_px.pct_change(fill_method=None).shift(-1)
    print("\n--- 分位単調性（等加重・翌月平均リターン bps）---")
    for label, sig in [("ΔL_ST resid", sig_st_r[valid]), ("−ΔL_LT resid", sig_lt_r[valid])]:
        buckets = {q: [] for q in range(5)}
        for t in sig.index:
            row = sig.loc[t].dropna()
            f = fwd.loc[t] if t in fwd.index else None
            if len(row) < 100 or f is None:
                continue
            qs = pd.qcut(row.rank(method="first"), 5, labels=False)
            for q in range(5):
                r_ = f.reindex(row.index[qs == q]).mean()
                if pd.notna(r_):
                    buckets[q].append(float(r_))
        line = "  ".join(f"Q{q + 1}={np.mean(xs) * 1e4:+.0f}" for q, xs in buckets.items() if xs)
        print(f"  {label}: {line}（Q5-Q1={np.mean(buckets[4]) * 1e4 - np.mean(buckets[0]) * 1e4:+.0f}）")

    # コスト感応（最良セル）
    best = v.best.name if v.best else "dliq_st_resid_q20"
    st = next(s for s in strategies if s.name == best)
    print(f"\n--- コスト感応（{best}・貸株115bps）---")
    for c in [0, 15, 30, 50]:
        res = backtest(st, view, costs_bps=float(c), adv=tadv, participation=0.1,
                       no_buy=no_buy, no_sell=no_sell, short_borrow_bps=BORROW_BPS)
        print(f"  {c:>3}bps  net年率SR={_sr(res.returns.dropna()):+.2f}  "
              f"回転={res.turnover.mean():.2f}")

    print("\n※ 判定は scope=delta_liquidity の DSR（K=4・docs/56 §2.6）。診断は throwaway（K不変）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
