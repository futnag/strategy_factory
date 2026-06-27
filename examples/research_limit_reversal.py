"""仮説検証：ストップ高(UL==1)後の短期リバーサル（日次イベント・レジストリ外の新軸）。

事前登録＝docs/48。`UL`/`LL`（ストップ高/安ヒットフラグ＝0/1）は `frictions.py` で執行コスト
（張り付き）としてのみ使われ、742試行のどの scope にもシグナルとして入っていない。本スクリプトは
それを**シグナル**として、PIT・貸株コスト・値幅ロック・容量・大域デフレートDSR で正規に裁く。

設計（K=3・走らせる前に固定）＝**イベント名ショート × 流動ユニバースEWロング**のドルニュートラル日次:
  lu_rev_h5         全 eligible 普通株・保有5日   （H1 主判定）
  lu_rev_h10        全 eligible 普通株・保有10日  （保有窓の頑健性）
  lu_rev_h5_illiq   低流動帯[300,1500)・保有5日   （H2 局在＝裁定の限界）
除外オーバーレイ（H3・借株不要の実用形）は診断（K外）。借株データ未保有のため貸株は保守仮定
（年率300bps）＋sweep で損益分岐を露出する（docs/48 §2.7）。

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_limit_reversal.py
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

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.data.store import load_wide  # noqa: E402
from invest_system.equities.universe import filter_common_stocks  # noqa: E402
from invest_system.equities.frictions import limit_lock_flags  # noqa: E402
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, PrecomputedWeights, judge_grid, write_html,
)
from invest_system.research.engine import backtest  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import TrialRegistry, default_registry  # noqa: E402

OOS = get_env("J_LR_OOS", "2024-01") or "2024-01"
REG_PATH = get_env("J_LR_REGISTRY", None)
HOLD_A = int(get_env("J_LR_HOLD_A", "5") or "5")        # 主保有窓（日）
HOLD_B = int(get_env("J_LR_HOLD_B", "10") or "10")      # 頑健性保有窓
HEDGE_N = int(get_env("J_LR_HEDGE_N", "300") or "300")  # 流動ヘッジ＝上位N売買代金
BAND_LO = int(get_env("J_LR_BAND_LO", "300") or "300")  # 低流動帯（H2）の上端ランク
BAND_HI = int(get_env("J_LR_BAND_HI", "1500") or "1500")
COST_BPS = float(get_env("J_LR_COST", "30") or "30")    # 片道コスト（判定）
BORROW_BPS = float(get_env("J_LR_BORROW", "300") or "300")  # 年率貸株（保守仮定）
EXCL_WIN = int(get_env("J_LR_EXCL", "20") or "20")      # 除外オーバーレイの窓（日）


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ser = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(ser.groupby(idx.to_period("M")).max().values)


def _ann_sr(r: pd.Series, ann: float) -> float:
    r = r.dropna()
    return sharpe_ratio(r) * np.sqrt(ann) if r.size >= 8 else float("nan")


def _maxdd(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    cum = (1.0 + r).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def build_ls_weights(event_win: pd.DataFrame, hedge: pd.DataFrame) -> pd.DataFrame:
    """ドルニュートラル日次ウェイト：イベント名ショート(Σ=−1) × 流動ヘッジEWロング(Σ=+1)。

    event_win: bool（窓内にストップ高を付けた当該ユニバースの銘柄＝ショート対象・PIT）。
    hedge: bool（上位流動の市場ヘッジ・PIT）。イベント名はヘッジから除外（二重計上回避）。
    イベント0件の日は全行NaN＝現金（PrecomputedWeights が dropna→空）。
    """
    ev = event_win.fillna(False)
    ne = ev.sum(axis=1)
    active = ne > 0
    short = (-ev.astype(float)).div(ne.where(ne > 0), axis=0)        # −1/E
    lp = hedge.fillna(False) & (~ev)
    nl = lp.sum(axis=1)
    longw = lp.astype(float).div(nl.where(nl > 0), axis=0)           # +1/L
    w = short.add(longw, fill_value=0.0)
    w.loc[~active] = np.nan                                          # 無イベント日＝現金
    return w


def main() -> int:
    adj = load_wide("adj_close")
    turn = load_wide("turnover")
    close = load_wide("close")
    high = load_wide("high")
    low = load_wide("low")
    ul = load_wide("upper_limit")       # 0/1 ストップ高フラグ（価格でない・docs/24 §5-10）
    ll = load_wide("lower_limit")       # 0/1 ストップ安フラグ
    vol = load_wide("volume")
    if adj.empty or turn.empty or ul.empty:
        print("ERROR: Silver 未生成。store.materialize_all() を先に実行してください。")
        return 1

    listed = jq.fetch_listed_info().assign(Code=lambda d: d["Code"].astype(str))
    common = set(filter_common_stocks(listed)["Code"])
    cols = [c for c in adj.columns if str(c) in common]
    adj, turn, close, high, low, ul, ll, vol = (
        d.reindex(columns=cols) for d in (adj, turn, close, high, low, ul, ll, vol))
    idx = adj.index
    print(f"=== ストップ高リバーサル検証  日次{len(idx)}  {idx.min():%Y-%m}..{idx.max():%Y-%m}"
          f"  普通株{len(cols)}  貸株{BORROW_BPS:.0f}bps  コスト{COST_BPS:.0f}bps ===")

    # PIT eligible（生存者バイアス排除）: 取引中 ∧ trailing252日に≥120日約定 ∧ 終値あり
    traded = turn.notna() & (turn > 0)
    liq_hist = traded.rolling(252, min_periods=120).sum() >= 120
    eligible = traded & liq_hist & adj.notna()
    # PIT 流動ランク（trailing 平均売買代金=ADV）→ ヘッジ(上位N) と 低流動帯[LO,HI)
    tadv = turn.rolling(252, min_periods=120).mean()
    rank = tadv.rank(axis=1, ascending=False)
    hedge = (rank <= HEDGE_N) & eligible
    illiq = (rank > BAND_LO) & (rank <= BAND_HI) & eligible
    print(f"ユニバース日平均  eligible={eligible.sum(axis=1).mean():.0f}  "
          f"hedge={hedge.sum(axis=1).mean():.0f}  illiq帯={illiq.sum(axis=1).mean():.0f}")

    # ストップ高イベント（PIT：UL==1 は引けで既知）と保有窓
    ulf = (ul == 1)
    print(f"UL==1 総数={int(ulf.to_numpy().sum()):,}（{ulf.sum(axis=1).mean():.1f}/日）  "
          f"LL==1 総数={int((ll == 1).to_numpy().sum()):,}")
    win_a = ulf.rolling(HOLD_A, min_periods=1).max() > 0
    win_b = ulf.rolling(HOLD_B, min_periods=1).max() > 0

    w_h5 = build_ls_weights(win_a & eligible, hedge)
    w_h10 = build_ls_weights(win_b & eligible, hedge)
    w_h5_il = build_ls_weights(win_a & illiq, hedge)

    # 値幅ロック（DP15）：執行不能フラグ（エンジンが execution_lag 分シフトして適用）
    no_buy, no_sell = limit_lock_flags(close, high, low, ul, ll, vol)

    view = AsOfView({"close": adj})
    strategies = [
        PrecomputedWeights(w_h5, name="lu_rev_h5",
                           params={"hold": HOLD_A, "universe": "all", "event": "UL"}),
        PrecomputedWeights(w_h10, name="lu_rev_h10",
                           params={"hold": HOLD_B, "universe": "all", "event": "UL"}),
        PrecomputedWeights(w_h5_il, name="lu_rev_h5_illiq",
                           params={"hold": HOLD_A, "universe": f"illiq[{BAND_LO},{BAND_HI})",
                                   "event": "UL"}),
    ]

    hyp = ("当日ストップ高(UL==1)銘柄は翌日以降アンダーパフォーム（注意誘発の過剰反応→平均回帰）。"
           "イベント名ショート×流動EWロングのドルニュートラルが、保守的貸株＋値幅ロック後もα")
    rat = ("ストップ高＝注意誘発・宝くじ需要・個人の追随による一過性の過剰反応。裁定の限界（借株困難・"
           "小型）で緩やかに反転。価格水準の単調変換でない独立イベント＝momentum代理/value共変を構造回避")

    reg_cm = TrialRegistry(REG_PATH) if REG_PATH else default_registry()
    with reg_cm as reg_db:
        v = judge_grid(strategies, view, scope="limit_reversal",
                       hypothesis=hyp, economic_rationale=rat, registry=reg_db,
                       costs_bps=COST_BPS, execution_lag=1, adv=turn,
                       no_buy=no_buy, no_sell=no_sell, short_borrow_bps=BORROW_BPS)
    print("\n" + v.report_md)
    write_html(v, "data/reports/limit_reversal.html")

    # ---- 診断（throwaway・K不変：直接 backtest はレジストリに触れない）----
    print(f"\n--- 診断: gross/IS/OOS(保留{OOS}〜)/前後2020/容量/blocked ---")
    for s in strategies:
        res = backtest(s, view, costs_bps=COST_BPS, execution_lag=1, adv=turn,
                       no_buy=no_buy, no_sell=no_sell, short_borrow_bps=BORROW_BPS)
        net, gross = res.returns.dropna(), res.gross.dropna()
        af = res.ann_factor
        is_ = net[net.index < pd.Timestamp(OOS)]
        oos = net[net.index >= pd.Timestamp(OOS)]
        (_, pre), (_, post) = pre_post_sharpe(net, "2020-01-01")
        print(f"  {s.name:<16} net={_ann_sr(net, af):+.2f} gross={_ann_sr(gross, af):+.2f} | "
              f"IS={_ann_sr(is_, af):+.2f} OOS={_ann_sr(oos, af):+.2f} | "
              f"前2020={pre:+.2f} 後={post:+.2f} | maxDD={_maxdd(net):.1%} | "
              f"容量={res.capacity_jpy/1e6:.1f}百万 blocked計={int(res.n_blocked.sum())}")

    # 年次 net 年率SR（直近死でないかの確認）＝judge の系列を再利用
    print("\n--- 年次 net 年率SR（lu_rev_h5）---")
    s5 = v.series.get("lu_rev_h5")
    if s5 is not None:
        for y, seg in s5.groupby(s5.index.year):
            print(f"  {y}: SR={_ann_sr(seg, 252.0):+.2f}  n={seg.size}")

    # 貸株感応・コスト感応（lu_rev_h5・損益分岐の露出）
    print("\n--- 貸株感応（lu_rev_h5・年率bps→net年率SR）---")
    for b in [0, 115, 300, 500, 1000, 2000]:
        res = backtest(strategies[0], view, costs_bps=COST_BPS, execution_lag=1,
                       adv=turn, no_buy=no_buy, no_sell=no_sell, short_borrow_bps=float(b))
        print(f"  貸株{b:>5}bps  net年率SR={_ann_sr(res.returns.dropna(), res.ann_factor):+.2f}")
    print("--- コスト感応（lu_rev_h5・片道bps→net年率SR・貸株300bps）---")
    for c in [0, 15, 30, 50, 100]:
        res = backtest(strategies[0], view, costs_bps=float(c), execution_lag=1,
                       adv=turn, no_buy=no_buy, no_sell=no_sell, short_borrow_bps=BORROW_BPS)
        print(f"  {c:>4}bps  net年率SR={_ann_sr(res.returns.dropna(), res.ann_factor):+.2f}  "
              f"回転={res.turnover.mean():.3f}")

    # ---- H3: 除外オーバーレイ（借株不要の実用形・診断・K外）----
    print(f"\n--- H3 除外オーバーレイ（月次・上位{HEDGE_N}EWロング・直近{EXCL_WIN}日ストップ高を除外）---")
    me = _month_ends(idx)
    hedge_me = hedge.loc[me]
    recent_ul = (ulf.rolling(EXCL_WIN, min_periods=1).max() > 0).loc[me]
    base_pool = hedge_me
    excl_pool = hedge_me & (~recent_ul)

    def _ew(pool: pd.DataFrame) -> pd.DataFrame:
        n = pool.sum(axis=1)
        return pool.astype(float).div(n.where(n > 0), axis=0)

    view_me = AsOfView({"close": adj.loc[me]})
    res_base = backtest(PrecomputedWeights(_ew(base_pool), name="base_long"),
                        view_me, costs_bps=COST_BPS)
    res_excl = backtest(PrecomputedWeights(_ew(excl_pool), name="excl_long"),
                        view_me, costs_bps=COST_BPS)
    rb, re_ = res_base.returns.dropna(), res_excl.returns.dropna()
    spread = (re_ - rb).dropna()
    n_excl = (recent_ul & hedge_me).sum(axis=1).mean()
    print(f"  素ロング   SR={_ann_sr(rb, 12.0):+.2f}  maxDD={_maxdd(rb):.1%}")
    print(f"  除外ロング SR={_ann_sr(re_, 12.0):+.2f}  maxDD={_maxdd(re_):.1%}  "
          f"（月平均除外数={n_excl:.1f}）")
    print(f"  スプレッド(除外−素) SR={_ann_sr(spread, 12.0):+.2f}  平均={spread.mean()*1e4:+.1f}bps/月"
          f"  ＝>0 なら除外がロング簿を改善（借株不要の収穫）")
    print("\n  ※ 判定は scope=limit_reversal の DSR（上表・K=3）。診断はすべて throwaway（K不変）。"
          "\n  ※ 借株データ未保有＝貸株は保守仮定。実勢逆日歩は JSF 統合が必要（docs/48 §2.7）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
