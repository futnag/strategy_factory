"""仮説検証：経営者予想の裁量部分（DF）の過大評価と12ヶ月 unwind（docs/59）。

事前登録＝docs/59（scope=`unexpected_forecast`・K=4・実行前コミット）。
年次短信の期初予想 NxFNp から FI=(NxFNp−NP)/TA_prev を作り、縮約期待モデル
（CROA ~ CHGROA_{-1}+ΔSales_{-1}+CTAC_{-1}+logTA+Eq/TA・拡張ウォークフォワード）の予測 NDF
との差 DF=FI−NDF を裁量的予想とする。シグナル＝−DF（DF 高→ショート）。
方向は Kitagawa & Shuto (JBFA 2024 / CARF F367 本文確認済み) で事前固定。

グリッド（docs/59 §2.4 固定・4セル）:
  uf_df_q20（主）/ uf_df_qneu_q20（F7ガード）/ uf_ndf_q20（統制・H-18）/ uf_fi_q20（未分解）

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_unexpected_forecast.py
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
from invest_system.equities.fundamentals import load_fundamentals  # noqa: E402
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
SCOPE = "unexpected_forecast"
TOP_N = 500
COST_BPS = 15.0
BORROW_BPS = 115.0
FEATS = ["chgroa", "dsales", "ctac", "log_ta", "lev"]


def _sr(x: pd.Series, ann: float = 12.0, lo=None, hi=None) -> float:
    r = x.dropna()
    if lo is not None:
        r = r[r.index >= pd.Timestamp(lo)]
    if hi is not None:
        r = r[r.index < pd.Timestamp(hi)]
    if r.size < 8 or float(r.std(ddof=1)) == 0.0:
        return float("nan")
    return float(sharpe_ratio(r) * np.sqrt(ann))


def _rank_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    cs = []
    for t in a.index.intersection(b.index):
        both = pd.concat([a.loc[t], b.loc[t]], axis=1).dropna()
        if len(both) >= 30:
            cs.append(float(both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank())))
    return float(np.mean(cs)) if cs else float("nan")


def build_firm_years() -> pd.DataFrame:
    """年次短信（CurPerType=FY）から firm-year テーブルを構築（PIT: DiscDate 付き）。"""
    f = load_fundamentals()
    f = f[f.get("CurPerType", pd.Series(dtype=object)).astype(str) == "FY"].copy()
    cols = ["Code", "DiscDate", "CurFYEn", "NP", "NxFNp", "TA", "Eq", "Sales", "CFO"]
    f = f[[c for c in cols if c in f.columns]].copy()
    for c in ["NP", "NxFNp", "TA", "Eq", "Sales", "CFO"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f["DiscDate"] = pd.to_datetime(f["DiscDate"])
    f["CurFYEn"] = pd.to_datetime(f["CurFYEn"])
    f["Code"] = f["Code"].astype(str)
    f = (f.sort_values("DiscDate")
         .drop_duplicates(subset=["Code", "CurFYEn"], keep="last"))
    f = f.sort_values(["Code", "CurFYEn"]).reset_index(drop=True)
    g = f.groupby("Code")
    f["np1"] = g["NP"].shift(1)
    f["np2"] = g["NP"].shift(2)
    f["ta1"] = g["TA"].shift(1)
    f["ta2"] = g["TA"].shift(2)
    f["sales1"] = g["Sales"].shift(1)
    f["sales2"] = g["Sales"].shift(2)
    f["sales3"] = g["Sales"].shift(3)
    f["cfo1"] = g["CFO"].shift(1)
    # 目的変数と特徴量（docs/59 §2.4・当年 y の特徴 X_y は NDF_y の予測入力）
    f["croa"] = (f["NP"] - f["np1"]) / f["ta1"]                      # 実現ΔROA（学習の目的変数）
    f["chgroa"] = f["croa"]                                          # X_y の第1成分（当年実現Δ）
    es = (f["sales1"] + f["sales2"]) / 2.0
    f["dsales"] = (f["Sales"] - es) / es                             # 2年平均期待形のΔSales
    tac = (f["NP"] - f["CFO"]) / f["TA"]
    tac1 = (f["np1"] - f["cfo1"]) / f["ta1"]
    f["ctac"] = tac - tac1
    f["log_ta"] = np.log(f["TA"].where(f["TA"] > 0))
    f["lev"] = f["Eq"] / f["TA"]
    f["fi"] = (f["NxFNp"] - f["NP"]) / f["ta1"]                      # 予想イノベーション
    f["fy_year"] = f["CurFYEn"].dt.year
    return f


def walk_forward_df(f: pd.DataFrame) -> pd.DataFrame:
    """拡張ウォークフォワードで NDF/DF を推定（コホート y は y−1 以前の実現のみで学習）。"""
    g = f.groupby("Code")
    train = f.copy()
    for c in FEATS:
        train[f"x_{c}"] = g[c].shift(1)                              # 学習ペア: CROA_y ~ X_{y−1}
    out = []
    years = sorted(f["fy_year"].dropna().unique())
    for y in years:
        tr = train[(train["fy_year"] < y)].dropna(
            subset=["croa"] + [f"x_{c}" for c in FEATS])
        te = f[f["fy_year"] == y].dropna(subset=FEATS + ["fi"])
        if len(tr) < 300 or te.empty:
            continue
        # 外れ値に頑健な OLS（学習・予測とも 1/99% クリップ＝実装衛生）
        Xtr = tr[[f"x_{c}" for c in FEATS]].to_numpy(float)
        ytr = tr["croa"].to_numpy(float)
        lo, hi = np.nanpercentile(ytr, [1, 99])
        ytr = np.clip(ytr, lo, hi)
        for j in range(Xtr.shape[1]):
            lo, hi = np.nanpercentile(Xtr[:, j], [1, 99])
            Xtr[:, j] = np.clip(Xtr[:, j], lo, hi)
        Xtr = np.column_stack([np.ones(len(Xtr)), Xtr])
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        Xte = te[FEATS].to_numpy(float)
        for j in range(Xte.shape[1]):
            lo, hi = np.nanpercentile(Xte[:, j], [1, 99])
            Xte[:, j] = np.clip(Xte[:, j], lo, hi)
        ndf = np.column_stack([np.ones(len(Xte)), Xte]) @ beta
        te = te.assign(ndf=ndf, df=te["fi"].to_numpy(float) - ndf)
        out.append(te)
    return pd.concat(out, ignore_index=True)


def annual_to_monthly(fy: pd.DataFrame, col: str, me: pd.DatetimeIndex,
                      codes: pd.Index) -> pd.DataFrame:
    """firm-year の値を「開示後の月末から次の年次開示まで（最長14ヶ月）」の月次 wide に展開。"""
    panel = pd.DataFrame(np.nan, index=me, columns=codes)
    fy = fy.dropna(subset=[col, "DiscDate"])
    for code, sub in fy.groupby("Code"):
        if code not in panel.columns:
            continue
        sub = sub.sort_values("DiscDate")
        for i, r in enumerate(sub.itertuples()):
            start = me.searchsorted(r.DiscDate, side="left")
            end_date = (sub.iloc[i + 1]["DiscDate"] if i + 1 < len(sub)
                        else r.DiscDate + pd.Timedelta(days=430))
            end = me.searchsorted(min(end_date, r.DiscDate + pd.Timedelta(days=430)),
                                  side="left")
            if start < end:
                panel.iloc[start:end, panel.columns.get_loc(code)] = getattr(r, col)
    return panel


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
    per = idx.to_period("M")
    me = pd.DatetimeIndex(pd.Series(idx, index=idx).groupby(per).max().values)

    print(f"=== 経営者予想の裁量部分 検証（scope={SCOPE}）===")
    f = build_firm_years()
    fy = walk_forward_df(f)
    print(f"firm-years: 全={len(f):,}  DF推定済み={len(fy):,}  "
          f"コホート={sorted(fy['fy_year'].unique())}")
    print(f"FI 平均={fy['fi'].mean():+.4f}  DF 平均={fy['df'].mean():+.4f}  "
          f"|DF|/|FI| 中央値={float((fy['df'].abs() / fy['fi'].abs().replace(0, np.nan)).median()):.2f}")

    sig_df = annual_to_monthly(fy.assign(v=-fy["df"]), "v", me, adj.columns)
    sig_ndf = annual_to_monthly(fy.assign(v=-fy["ndf"]), "v", me, adj.columns)
    sig_fi = annual_to_monthly(fy.assign(v=-fy["fi"]), "v", me, adj.columns)
    roa_p = annual_to_monthly(fy.assign(v=fy["NP"] / fy["TA"]), "v", me, adj.columns)
    tac_p = annual_to_monthly(fy.assign(v=(fy["NP"] - fy["CFO"]) / fy["TA"]),
                              "v", me, adj.columns)
    lev_p = annual_to_monthly(fy.assign(v=fy["lev"]), "v", me, adj.columns)

    turn_m = turn.groupby(per).median().set_axis(me)
    uni = point_in_time_universe(turn_m, top_n=TOP_N, lookback=12, min_obs=6)
    sig_df, sig_ndf, sig_fi = (s.where(uni) for s in (sig_df, sig_ndf, sig_fi))
    sig_qneu = cross_sectional_residualize(
        sig_df, [roa_p.where(uni), tac_p.where(uni), lev_p.where(uni)])

    valid = sig_df.notna().sum(axis=1) >= 100
    print(f"シグナル有効月: {int(valid.sum())}/{len(me)}  "
          f"開始={sig_df.index[valid.argmax()]:%Y-%m}  "
          f"平均カバレッジ={sig_df[valid].notna().sum(axis=1).mean():.0f}銘柄")

    strategies = [
        CrossSectionalStrategy(sig_df[valid], 0.2, name="uf_df_q20"),
        CrossSectionalStrategy(sig_qneu[valid], 0.2, name="uf_df_qneu_q20"),
        CrossSectionalStrategy(sig_ndf[valid], 0.2, name="uf_ndf_q20"),
        CrossSectionalStrategy(sig_fi[valid], 0.2, name="uf_fi_q20"),
    ]

    fill_px = opn.bfill(limit=3).shift(-1).reindex(me)
    view = AsOfView({"close": fill_px})
    no_buy_d, no_sell_d = limit_lock_flags(close, high, low, ul, ll, vol_)
    no_buy, no_sell = no_buy_d.reindex(me), no_sell_d.reindex(me)
    tadv = turn.rolling(252, min_periods=120).mean().reindex(me)

    hyp = ("期初経営者予想の裁量部分 DF（FI−縮約期待モデル予測）が高い銘柄は以後12ヶ月"
           "アンダーパフォームし低い銘柄が優越する（市場は less credible な裁量を過大評価し"
           "期中実績で緩やかに修正・docs/59・方向は日本主標本 JBFA2024 で事前固定）")
    rat = ("日本の実質強制の点予想には forecast management が混入。credible 部分は適正評価・"
           "裁量部分は過大評価され、ERROR/REVISION の実現とともに年度を通じて unwind"
           "（H-17 のジャンプ型でない12ヶ月拡散）。docs/59 §2.2")

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
        print(f"  {r.name:<16} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={_sr(s, hi=OOS):+.2f} OOS={_sr(s, lo=OOS):+.2f} | "
              f"前/後2020={pre:+.2f}/{post:+.2f} | maxDD={mdd:.1%}")

    best = v.best.name if v.best else "uf_df_q20"
    sb = v.series.get(best, pd.Series(dtype="float64")).dropna()
    print(f"\n--- 年次 net SR（{best}）---")
    for y, seg in sb.groupby(sb.index.year):
        print(f"  {y}: SR={_sr(seg):+.2f}  n={seg.size}")

    # 独立性・機構診断
    adj_me = adj.reindex(me)
    mom = adj_me.shift(1) / adj_me.shift(12) - 1.0
    ret1m = adj_me.pct_change(fill_method=None)
    vol252 = adj.pct_change(fill_method=None).rolling(252, min_periods=126).std().reindex(me)
    print("\n--- 独立性（主シグナル −DF の平均XS順位相関）---")
    print(f"  vs momentum ρ̄={_rank_corr(sig_df[valid], mom.where(uni)):+.3f} / "
          f"低ボラ ρ̄={_rank_corr(sig_df[valid], (-vol252).where(uni)):+.3f} / "
          f"ROA ρ̄={_rank_corr(sig_df[valid], roa_p.where(uni)):+.3f}")
    # DF と前年予想誤差の相関（I-44 との関係・firm-year レベル）
    f2 = fy.copy()
    f2["prior_err"] = (f2["NP"] - f2.groupby("Code")["NxFNp"].shift(1)) / f2["ta1"]
    both = f2[["df", "prior_err"]].dropna()
    print(f"  DF vs 前年予想誤差（firm-year）ρ={both['df'].corr(both['prior_err']):+.2f}"
          f"（負＝楽観の持続と整合・I-44 の軸）")
    # ERROR_{t+1} との機構検証（記述）: DF 高→翌年の予想誤差は負か
    f2["next_err"] = f2.groupby("Code")["prior_err"].shift(-1)
    both2 = f2[["df", "next_err"]].dropna()
    print(f"  DF vs 翌年予想誤差 ρ={both2['df'].corr(both2['next_err']):+.2f}"
          f"（負＝『裁量は未達に終わる』の機構・論文 H1a の再現確認）")

    # 分位単調性（主セル）
    fwd = fill_px.pct_change(fill_method=None).shift(-1)
    buckets = {q: [] for q in range(5)}
    for t in sig_df[valid].index:
        row = sig_df.loc[t].dropna()
        if len(row) < 100 or t not in fwd.index:
            continue
        qs = pd.qcut(row.rank(method="first"), 5, labels=False)
        for q in range(5):
            r_ = fwd.loc[t].reindex(row.index[qs == q]).mean()
            if pd.notna(r_):
                buckets[q].append(float(r_))
    print("\n--- 分位単調性（−DF・翌月平均リターン bps）---")
    print("  " + "  ".join(f"Q{q + 1}={np.mean(xs) * 1e4:+.0f}" for q, xs in buckets.items() if xs)
          + f"（Q5-Q1={np.mean(buckets[4]) * 1e4 - np.mean(buckets[0]) * 1e4:+.0f}）")

    # コスト感応（主セル）
    st = strategies[0]
    print(f"\n--- コスト感応（uf_df_q20・貸株115bps）---")
    for c in [0, 15, 30, 50]:
        res = backtest(st, view, costs_bps=float(c), adv=tadv, participation=0.1,
                       no_buy=no_buy, no_sell=no_sell, short_borrow_bps=BORROW_BPS)
        print(f"  {c:>3}bps  net年率SR={_sr(res.returns.dropna()):+.2f}  "
              f"回転={res.turnover.mean():.2f}")

    print("\n※ 判定は scope=unexpected_forecast の DSR（K=4・docs/59 §2.6）。診断は throwaway。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
