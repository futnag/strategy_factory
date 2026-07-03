"""仮説検証：ジャンプテールβのクロスセクション（N225 OTMプット→個別株・docs/55）。

事前登録＝docs/55（scope=`jump_tail_beta_xs`・K=4・実行前コミット）。
N225 近満期（7-45暦日）プットの deep-OTM(K/S 0.80-0.92) 平均IV − ATM(0.97-1.03) 平均IV を
左テール尺度 LT とし、ΔLT への銘柄感応度（市場リターン同時制御の二変量ローリング回帰）で
低テールβロング/高テールβショートの月次 XS L/S を裁く。方向は米国証拠（JEF 2024:
high−low −9.95%/年）で事前固定。残差化主セルの生存を副次基準化（BAB/低ボラ代理の遮断）。

グリッド（docs/55 §2.4 固定・4セル）:
  jtb_raw_q20_b252 / jtb_resid_q20_b252(主) / jtb_resid_q10_b252 / jtb_resid_q20_b126

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_jump_tail_beta.py
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
SCOPE = "jump_tail_beta_xs"
TOP_N = 500
COST_BPS = 15.0
BORROW_BPS = 115.0
DTE_LO, DTE_HI = 7, 45
DEEP_LO, DEEP_HI = 0.80, 0.92
ATM_LO, ATM_HI = 0.97, 1.03
MIN_CONTRACTS = 3
OPT_DIR = Path("data/jquants/options_225")
CACHE = Path("data/processed/n225_tail_measure.parquet")


def build_tail_measure() -> pd.DataFrame:
    """日次 [lt, atm_iv, under]（キャッシュ付き）。lt=deepOTMプットIV平均−ATMプットIV平均。"""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    rows = []
    for f in sorted(OPT_DIR.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty or "_empty" in df.columns:
            continue
        d = pd.to_datetime(df["Date"].iloc[0])
        under = df["UnderPx"].replace(0.0, np.nan).dropna()
        if under.empty:
            continue
        s = float(under.median())
        ltd = pd.to_datetime(df["LTD"])
        dte = (ltd - d).dt.days
        puts = df[(df["PCDiv"].astype(str) == "1") & (dte >= DTE_LO) & (dte <= DTE_HI)
                  & (df["IV"] > 0)]
        mny = puts["Strike"] / s
        deep = puts[(mny >= DEEP_LO) & (mny <= DEEP_HI)]["IV"]
        atm = puts[(mny >= ATM_LO) & (mny <= ATM_HI)]["IV"]
        if len(deep) < MIN_CONTRACTS or len(atm) < MIN_CONTRACTS:
            continue
        rows.append((d, float(deep.mean() - atm.mean()), float(atm.mean()), s))
    out = pd.DataFrame(rows, columns=["Date", "lt", "atm_iv", "under"]).set_index("Date")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE)
    return out


def rolling_bivariate_beta(ret: pd.DataFrame, m: pd.Series, j: pd.Series,
                           window: int, min_p: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """y=a+b_m·m+b_j·j+e のローリング閉形式。返り値 (β_jump, β_mkt)（wide・≤t）。"""
    def rmean(x):
        return x.rolling(window, min_periods=min_p).mean()

    em, ej = rmean(m), rmean(j)
    varm = rmean(m * m) - em ** 2
    varj = rmean(j * j) - ej ** 2
    covmj = rmean(m * j) - em * ej
    ey = rmean(ret)
    covym = rmean(ret.mul(m, axis=0)) - ey.mul(em, axis=0)
    covyj = rmean(ret.mul(j, axis=0)) - ey.mul(ej, axis=0)
    det = (varm * varj - covmj ** 2).replace(0.0, np.nan)
    b_j = (covyj.mul(varm, axis=0) - covym.mul(covmj, axis=0)).div(det, axis=0)
    b_m = (covym.mul(varj, axis=0) - covyj.mul(covmj, axis=0)).div(det, axis=0)
    return b_j, b_m


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
    """日次(行)ごとのクロスセクション順位相関の平均。"""
    cs = []
    for t in a.index.intersection(b.index):
        x, y = a.loc[t], b.loc[t]
        both = pd.concat([x, y], axis=1).dropna()
        if len(both) >= 30:
            cs.append(float(both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank())))
    return float(np.mean(cs)) if cs else float("nan")


def main() -> int:
    tm = build_tail_measure()
    print(f"=== ジャンプテールβ XS（scope={SCOPE}）===")
    print(f"テール尺度 LT: {tm.index.min():%Y-%m-%d}〜{tm.index.max():%Y-%m-%d}  "
          f"n={len(tm)}  平均={tm['lt'].mean():+.1f}pt  σ={tm['lt'].std():.1f}pt")

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

    ret = adj.pct_change(fill_method=None)
    m = tm["under"].pct_change().reindex(idx)
    j = tm["lt"].diff().reindex(idx)

    bj252, bm252 = rolling_bivariate_beta(ret, m, j, 252, 126)
    bj126, _ = rolling_bivariate_beta(ret, m, j, 126, 63)
    vol252 = ret.rolling(252, min_periods=126).std() * np.sqrt(252)

    # 月次リバランス（月末決定・翌営業日始値で約定＝fill価格ビュー・DP17）
    me = pd.DatetimeIndex(pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().values)
    turn_m = turn.groupby(idx.to_period("M")).median().set_axis(me)
    uni = point_in_time_universe(turn_m, top_n=TOP_N, lookback=12, min_obs=6)

    def sig_at(bj: pd.DataFrame) -> pd.DataFrame:
        s = (-bj).reindex(me)
        return s.where(uni)

    raw252 = sig_at(bj252)
    bm_me = bm252.reindex(me).where(uni)
    vol_me = vol252.reindex(me).where(uni)
    resid252 = cross_sectional_residualize(raw252, [bm_me, vol_me])
    resid126 = cross_sectional_residualize(sig_at(bj126), [bm_me, vol_me])

    # シグナルが十分に埋まる月のみ（IV 完備 2016-10 + β窓 → ~2017-10 開始）
    valid = raw252.notna().sum(axis=1) >= 100
    print(f"シグナル有効月: {int(valid.sum())}/{len(me)}  "
          f"開始={raw252.index[valid.argmax()]:%Y-%m}")

    strategies = [
        CrossSectionalStrategy(raw252[valid], 0.2, name="jtb_raw_q20_b252"),
        CrossSectionalStrategy(resid252[valid], 0.2, name="jtb_resid_q20_b252"),
        CrossSectionalStrategy(resid252[valid], 0.1, name="jtb_resid_q10_b252"),
        CrossSectionalStrategy(resid126[valid], 0.2, name="jtb_resid_q20_b126"),
    ]

    fill_px = opn.bfill(limit=3).shift(-1).reindex(me)
    view = AsOfView({"close": fill_px})
    no_buy_d, no_sell_d = limit_lock_flags(close, high, low, ul, ll, vol_)
    no_buy, no_sell = no_buy_d.reindex(me), no_sell_d.reindex(me)
    tadv = turn.rolling(252, min_periods=120).mean().reindex(me)

    hyp = ("N225 オプション由来の左テール尺度 ΔLT への感応度（ジャンプテールβ）が高い銘柄は"
           "将来劣後し低い銘柄が優越する（テールヘッジ需要の保険プレミアム・docs/55・"
           "方向は米国 JEF2024 で事前固定）")
    rat = ("クラッシュ回避選好はテールヘッジ資産に割増を払う＝高テールβは買われ過ぎで負の"
           "プレミアム。指数オプションのテール価格変動への XS 感応度は市場ベースのヘッジ能力"
           "尺度。既試 VRP（指数の保険料収穫）・IVゲート（タイミング）とは別のXSチャネル")

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

    # 独立性: シグナル vs 低ボラ(−vol)/市場β/モメンタム の平均順位相関
    m_close = adj.reindex(me)
    mom = m_close.shift(1) / m_close.shift(12) - 1.0
    print("\n--- 独立性（主セル resid252 シグナルの平均XS順位相関）---")
    print(f"  vs 低ボラ(−vol252)  ρ̄={_rank_corr(resid252[valid], -vol_me):+.3f}")
    print(f"  vs 市場β(−β_mkt)   ρ̄={_rank_corr(resid252[valid], -bm_me):+.3f}")
    print(f"  vs momentum(12-1)  ρ̄={_rank_corr(resid252[valid], mom.where(uni)):+.3f}")
    print(f"  生セル raw252 vs 低ボラ ρ̄={_rank_corr(raw252[valid], -vol_me):+.3f}"
          f"（残差化の効果確認）")
    d_atm = tm["atm_iv"].diff().reindex(idx)
    both = pd.concat([j, d_atm], axis=1).dropna()
    print(f"  ρ(ΔLT, ΔATM_IV) 日次={float(both.corr().iloc[0, 1]):+.2f}"
          f"（テール固有 vs ボラ一般の分離度）")

    # コスト/貸株感応（主セル）
    main_st = strategies[1]
    print("\n--- コスト感応（jtb_resid_q20_b252・貸株115bps）---")
    for c in [0, 15, 30, 50]:
        res = backtest(main_st, view, costs_bps=float(c), adv=tadv, participation=0.1,
                       no_buy=no_buy, no_sell=no_sell, short_borrow_bps=BORROW_BPS)
        print(f"  {c:>3}bps  net年率SR={_sr(res.returns.dropna()):+.2f}")
    print("--- 貸株感応（同・コスト15bps）---")
    for b in [0, 115, 300, 500]:
        res = backtest(main_st, view, costs_bps=COST_BPS, adv=tadv, participation=0.1,
                       no_buy=no_buy, no_sell=no_sell, short_borrow_bps=float(b))
        print(f"  {b:>4}bps  net年率SR={_sr(res.returns.dropna()):+.2f}")

    # 急落月の挙動（保険プレミアム仮説ならテール月に主セルが勝つはず）
    print("\n--- 急落月の主セル・リターン ---")
    s_main = v.series.get("jtb_resid_q20_b252", pd.Series(dtype="float64"))
    for label in ["2020-02", "2020-03", "2024-07", "2024-08"]:
        seg = s_main[s_main.index.to_period("M") == label]
        if len(seg):
            print(f"  {label}: {float(seg.iloc[0]):+.2%}")

    print("\n※ 判定は scope=jump_tail_beta_xs の DSR（K=4・docs/55 §2.6）。診断は throwaway（K不変）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
