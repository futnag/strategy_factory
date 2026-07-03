"""仮説検証：信用規制イベント（増担保規制・日々公表）の指定/解除後ドリフト。

事前登録＝docs/53（scope=`margin_alert_event`・K=6・実行前コミット）。
`data/jquants/margin_alert/`（日々公表信用残・PubDate=PIT アンカー）の `PubReason` フラグ遷移から
指定(start)/解除(end)イベントを導出し、イベント銘柄 vs 流動ヘッジのドルニュートラル日次 L/S を
PIT・値幅ロック・貸株300bps・コスト30bps・大域デフレートDSR で裁く。

グリッド（docs/53 §2.4 で固定・6セル）:
  reg_start_short_h5 / reg_start_short_h10   増担保指定 → ショート（H1）
  reg_end_long_h5   / reg_end_long_h10       増担保解除 → ロング（H2）
  dp_start_short_h5 / dp_end_long_h5         日々公表の指定/解除（H3）

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_margin_alert.py
"""
from __future__ import annotations

import json
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

OOS = get_env("J_MA_OOS", "2024-01") or "2024-01"
REG_PATH = get_env("J_MA_REGISTRY", None)
HOLD_A = int(get_env("J_MA_HOLD_A", "5") or "5")
HOLD_B = int(get_env("J_MA_HOLD_B", "10") or "10")
HEDGE_N = int(get_env("J_MA_HEDGE_N", "300") or "300")
COST_BPS = float(get_env("J_MA_COST", "30") or "30")
BORROW_BPS = float(get_env("J_MA_BORROW", "300") or "300")
ALERT_DIR = Path("data/jquants/margin_alert")
EVENT_CACHE = Path("data/processed/margin_alert_events.parquet")


def load_events() -> pd.DataFrame:
    """margin_alert フラグ遷移（指定/解除）を導出。[Date, Code, flag, kind]（キャッシュ付き）。"""
    if EVENT_CACHE.exists():
        return pd.read_parquet(EVENT_CACHE)
    rows = []
    for f in sorted(ALERT_DIR.glob("date_*.parquet")):
        df = pd.read_parquet(f)
        if df.empty or "_empty" in df.columns:
            continue
        rows.append(df[["PubDate", "Code", "PubReason"]])
    al = pd.concat(rows, ignore_index=True)
    al["PubDate"] = pd.to_datetime(al["PubDate"])
    flags = pd.json_normalize(al["PubReason"].map(json.loads))
    ev_rows = []
    for flag in ("Restricted", "DailyPublication"):
        mem = al[flags[flag].astype(str).values == "1"]
        sets = mem.groupby("PubDate")["Code"].apply(set)
        dates = sorted(al["PubDate"].unique())
        sets = sets.reindex(dates).map(lambda s: s if isinstance(s, set) else set())
        for i in range(1, len(dates)):
            prev, cur = sets.iloc[i - 1], sets.iloc[i]
            ev_rows += [(dates[i], c, flag, "start") for c in cur - prev]
            ev_rows += [(dates[i], c, flag, "end") for c in prev - cur]
    ev = pd.DataFrame(ev_rows, columns=["Date", "Code", "flag", "kind"])
    EVENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(EVENT_CACHE)
    return ev


def event_window(ev: pd.DataFrame, flag: str, kind: str, idx: pd.DatetimeIndex,
                 cols: pd.Index, hold: int) -> pd.DataFrame:
    """イベント後 hold 営業日以内なら True の wide bool（PubDate 当日を含む・執行は lag=1）。"""
    sub = ev[(ev["flag"] == flag) & (ev["kind"] == kind)]
    hit = pd.DataFrame(False, index=idx, columns=cols)
    pos = idx.searchsorted(pd.DatetimeIndex(sub["Date"]))  # PubDate ≤ 直後の営業日
    for p, c in zip(pos, sub["Code"]):
        if p < len(idx) and c in hit.columns:
            hit.iat[p, hit.columns.get_loc(c)] = True
    return hit.rolling(hold, min_periods=1).max() > 0


def ls_weights(event_win: pd.DataFrame, hedge: pd.DataFrame, side: int) -> pd.DataFrame:
    """ドルニュートラル日次: side=-1 イベント名ショート×ヘッジEWロング / side=+1 その逆。"""
    ev = event_win.fillna(False)
    ne = ev.sum(axis=1)
    active = ne > 0
    evw = (side * ev.astype(float)).div(ne.where(ne > 0), axis=0)
    hp = hedge.fillna(False) & (~ev)
    nh = hp.sum(axis=1)
    hw = (-side * hp.astype(float)).div(nh.where(nh > 0), axis=0)
    w = evw.add(hw, fill_value=0.0)
    w.loc[~active] = np.nan  # 無イベント日＝現金
    return w


def _ann_sr(r: pd.Series, ann: float) -> float:
    r = r.dropna()
    return sharpe_ratio(r) * np.sqrt(ann) if r.size >= 8 else float("nan")


def _maxdd(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    cum = (1.0 + r).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def _quintile_ls_returns(sig: pd.DataFrame, ret: pd.DataFrame, q: float = 0.2) -> pd.Series:
    """診断用：シグナル上位qロング/下位qショートのEW日次リターン（sigはt-1情報＝shift済みで渡す）。"""
    rk = sig.rank(axis=1, pct=True)
    long_ = (rk >= 1 - q).astype(float)
    short = (rk <= q).astype(float)
    lw = long_.div(long_.sum(axis=1), axis=0)
    sw = short.div(short.sum(axis=1), axis=0)
    return (ret * lw).sum(axis=1) - (ret * sw).sum(axis=1)


def main() -> int:
    adj = load_wide("adj_close")
    turn = load_wide("turnover")
    close = load_wide("close")
    high = load_wide("high")
    low = load_wide("low")
    ul = load_wide("upper_limit")
    ll = load_wide("lower_limit")
    vol = load_wide("volume")
    if adj.empty or turn.empty:
        print("ERROR: Silver 未生成。store.materialize_all() を先に実行してください。")
        return 1

    listed = jq.fetch_listed_info().assign(Code=lambda d: d["Code"].astype(str))
    common = set(filter_common_stocks(listed)["Code"])
    cols = [c for c in adj.columns if str(c) in common]
    adj, turn, close, high, low, ul, ll, vol = (
        d.reindex(columns=cols) for d in (adj, turn, close, high, low, ul, ll, vol))
    idx = adj.index

    ev = load_events()
    print(f"=== 信用規制イベント検証  日次{len(idx)}  {idx.min():%Y-%m}..{idx.max():%Y-%m}"
          f"  普通株{len(cols)}  コスト{COST_BPS:.0f}bps 貸株{BORROW_BPS:.0f}bps ===")
    print(ev.groupby(["flag", "kind"]).size().to_string())

    # PIT eligible ＋ 流動ヘッジ（trailing ADV 上位N）
    traded = turn.notna() & (turn > 0)
    liq_hist = traded.rolling(252, min_periods=120).sum() >= 120
    eligible = traded & liq_hist & adj.notna()
    tadv = turn.rolling(252, min_periods=120).mean()
    rank = tadv.rank(axis=1, ascending=False)
    hedge = (rank <= HEDGE_N) & eligible

    def win(flag: str, kind: str, hold: int) -> pd.DataFrame:
        return event_window(ev, flag, kind, idx, adj.columns, hold) & eligible

    grid = [
        ("reg_start_short_h5", win("Restricted", "start", HOLD_A), -1,
         {"flag": "Restricted", "kind": "start", "hold": HOLD_A}),
        ("reg_start_short_h10", win("Restricted", "start", HOLD_B), -1,
         {"flag": "Restricted", "kind": "start", "hold": HOLD_B}),
        ("reg_end_long_h5", win("Restricted", "end", HOLD_A), +1,
         {"flag": "Restricted", "kind": "end", "hold": HOLD_A}),
        ("reg_end_long_h10", win("Restricted", "end", HOLD_B), +1,
         {"flag": "Restricted", "kind": "end", "hold": HOLD_B}),
        ("dp_start_short_h5", win("DailyPublication", "start", HOLD_A), -1,
         {"flag": "DailyPublication", "kind": "start", "hold": HOLD_A}),
        ("dp_end_long_h5", win("DailyPublication", "end", HOLD_A), +1,
         {"flag": "DailyPublication", "kind": "end", "hold": HOLD_A}),
    ]
    strategies = [PrecomputedWeights(ls_weights(w, hedge, side), name=name, params=params)
                  for name, w, side, params in grid]

    no_buy, no_sell = limit_lock_flags(close, high, low, ul, ll, vol)
    view = AsOfView({"close": adj})

    hyp = ("増担保規制/日々公表の指定後は負・解除後は正のドリフト（信用買い需要の制度的遮断/解放）。"
           "イベント名 vs 流動ヘッジのドルニュートラルが保守コスト・貸株・値幅ロック後もα（docs/53）")
    rat = ("増担保規制は証拠金率引上げで信用買いの限界需要を制度的に断つ（指定=需給悪化・解除=回復）。"
           "規制イベント日アンカー＝momentum/valueの単調変換でない独立軸。docs/53 §2.2")

    reg_cm = TrialRegistry(REG_PATH) if REG_PATH else default_registry()
    with reg_cm as reg_db:
        v = judge_grid(strategies, view, scope="margin_alert_event",
                       hypothesis=hyp, economic_rationale=rat, registry=reg_db,
                       costs_bps=COST_BPS, execution_lag=1, adv=turn,
                       no_buy=no_buy, no_sell=no_sell, short_borrow_bps=BORROW_BPS)
    print("\n" + v.report_md)
    write_html(v, "data/reports/margin_alert_event.html")

    # ---- 診断（throwaway・K不変）----
    print(f"\n--- 診断: gross/IS/OOS({OOS}〜)/前後2020/maxDD/容量/blocked ---")
    for s in strategies:
        res = backtest(s, view, costs_bps=COST_BPS, execution_lag=1, adv=turn,
                       no_buy=no_buy, no_sell=no_sell, short_borrow_bps=BORROW_BPS)
        net, gross = res.returns.dropna(), res.gross.dropna()
        af = res.ann_factor
        is_ = net[net.index < pd.Timestamp(OOS)]
        oos = net[net.index >= pd.Timestamp(OOS)]
        (_, pre), (_, post) = pre_post_sharpe(net, "2020-01-01")
        print(f"  {s.name:<20} net={_ann_sr(net, af):+.2f} gross={_ann_sr(gross, af):+.2f} | "
              f"IS={_ann_sr(is_, af):+.2f} OOS={_ann_sr(oos, af):+.2f} | "
              f"前2020={pre:+.2f} 後={post:+.2f} | maxDD={_maxdd(net):.1%} | "
              f"容量={res.capacity_jpy/1e6:.0f}百万 blocked={int(res.n_blocked.sum())}")

    # 年次 net SR（最良セル・直近死/古データ減衰の確認）
    best = v.best.name if v.best else grid[0][0]
    sb = v.series.get(best)
    if sb is not None:
        print(f"\n--- 年次 net 年率SR（{best}）---")
        for y, seg in sb.groupby(sb.index.year):
            print(f"  {y}: SR={_ann_sr(seg, 252.0):+.2f}  n={seg.size}")

    # コスト/貸株 感応（主セル2本・損益分岐の露出）
    for tgt in ("reg_start_short_h5", "reg_end_long_h5"):
        st = next(s for s in strategies if s.name == tgt)
        print(f"\n--- コスト感応（{tgt}・貸株{BORROW_BPS:.0f}bps）---")
        for c in [0, 15, 30, 50, 100]:
            res = backtest(st, view, costs_bps=float(c), execution_lag=1, adv=turn,
                           no_buy=no_buy, no_sell=no_sell, short_borrow_bps=BORROW_BPS)
            print(f"  {c:>4}bps  net年率SR={_ann_sr(res.returns.dropna(), res.ann_factor):+.2f}")
        print(f"--- 貸株感応（{tgt}・コスト{COST_BPS:.0f}bps）---")
        for b in [0, 115, 300, 500, 1000]:
            res = backtest(st, view, costs_bps=COST_BPS, execution_lag=1, adv=turn,
                           no_buy=no_buy, no_sell=no_sell, short_borrow_bps=float(b))
            print(f"  {b:>5}bps  net年率SR={_ann_sr(res.returns.dropna(), res.ann_factor):+.2f}")

    # 独立性診断: ρ vs momentum/短期リバーサル/TOPIX/value（月次B/M）
    print("\n--- 独立性（最良セル net との相関）---")
    ret_d = adj.pct_change(fill_method=None)
    if sb is not None:
        try:
            from invest_system.data.feature_store import load_feature
            mom = load_feature("momentum_12_1").reindex(index=idx, columns=adj.columns)
            mom_ls = _quintile_ls_returns(mom.shift(1), ret_d)
            print(f"  ρ(momentum L/S 日次) = {sb.corr(mom_ls):+.3f}")
            rev = (-ret_d.rolling(5).sum()).shift(1)
            rev_ls = _quintile_ls_returns(rev, ret_d)
            print(f"  ρ(短期リバーサル L/S 日次) = {sb.corr(rev_ls):+.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  （momentum/reversal 診断スキップ: {e}）")
        try:
            topix = jq.fetch_index_bars("0028").set_index("Date")["C"].astype(float)
            topix.index = pd.to_datetime(topix.index)
            print(f"  ρ(TOPIX 日次) = {sb.corr(topix.pct_change(fill_method=None)):+.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  （TOPIX 診断スキップ: {e}）")
        try:
            from invest_system.equities.fundamentals import fundamentals_panel
            me = pd.DatetimeIndex(pd.Series(idx, index=idx)
                                  .groupby(idx.to_period("M")).max().values)
            pit = fundamentals_panel(me, ["Eq", "ShOutFY"])
            raw_me = close.loc[me]
            bm = pit["Eq"] / (pit["ShOutFY"] * raw_me)
            bm = bm.replace([np.inf, -np.inf], np.nan)
            ret_m = adj.loc[me].pct_change(fill_method=None)
            val_ls = _quintile_ls_returns(bm.shift(1), ret_m)
            sb_m = (1.0 + sb).groupby(sb.index.to_period("M")).prod() - 1.0
            val_m = pd.Series(val_ls.values,
                              index=pd.PeriodIndex(val_ls.index, freq="M"))
            both = pd.concat([sb_m, val_m], axis=1).dropna()
            print(f"  ρ(value(B/M) L/S 月次) = {both.iloc[:, 0].corr(both.iloc[:, 1]):+.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  （value 診断スキップ: {e}）")

    # イベント・スタディ形状（各イベントの t+1..t+10 平均超過リターン・記述）
    print("\n--- イベント・スタディ（対ヘッジ超過・平均累積 bps）---")
    hedge_ret = (ret_d.where(hedge)).mean(axis=1)
    for flag, kind in [("Restricted", "start"), ("Restricted", "end"),
                       ("DailyPublication", "start"), ("DailyPublication", "end")]:
        sub = ev[(ev["flag"] == flag) & (ev["kind"] == kind)]
        cum = np.zeros(10)
        n = 0
        for _, r in sub.iterrows():
            if r["Code"] not in adj.columns:
                continue
            p = idx.searchsorted(r["Date"])
            if p + 11 >= len(idx):
                continue
            rr = ret_d[r["Code"]].iloc[p + 1:p + 11].values - \
                hedge_ret.iloc[p + 1:p + 11].values
            if np.isnan(rr).all():
                continue
            cum += np.nan_to_num(rr)
            n += 1
        if n:
            c = np.cumsum(cum / n) * 1e4
            print(f"  {flag:<17}{kind:<6} n={n:>5}  +1d={c[0]:+.0f} +3d={c[2]:+.0f} "
                  f"+5d={c[4]:+.0f} +10d={c[9]:+.0f}")

    print("\n※ 判定は scope=margin_alert_event の DSR（K=6・docs/53 §2.6）。診断は throwaway（K不変）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
