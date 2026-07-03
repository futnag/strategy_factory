"""仮説検証：大量保有報告イベントの filer 異質性ドリフト（docs/57）。

事前登録＝docs/57（scope=`activist_filing_drift`・K=4・実行前コミット）。
EDINET 大量保有 DB（新規4,503＋変更3,226・2022-01〜2026-06）から、filer 名簿（in_registry）・
重要提案行為・積増しのイベント別に、イベント銘柄ロング×流動ヘッジショートの日次ドルニュートラル
（保有40営業日）を裁く。方向は日本主標本（Gillan et al. 2023 PBFJ）でロング側に事前固定。

グリッド（docs/57 §2.4 固定・4セル）:
  act_new_all_h40（統制）/ act_new_reg_h40（H1主）/ act_new_prop_h40（H2）/ act_accum_reg_h40（H3）

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_activist_filing.py
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
from invest_system.equities.frictions import limit_lock_flags  # noqa: E402
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, PrecomputedWeights, judge_grid, write_html,
)
from invest_system.research.engine import backtest  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

OOS = "2024-01"
SCOPE = "activist_filing_drift"
HOLD = 40                # 保有窓（営業日・docs/57 §2.4 固定）
HEDGE_N = 300
COST_BPS = 30.0
BORROW_BPS = 115.0
ACCUM_PT = 0.01          # 積増し閾値 +1pt（比率が%スケールなら 1.0 に自動補正）
LH = Path("data/edinet/large_holdings.parquet")
CH = Path("data/edinet/large_holdings_changes.parquet")
LIST_DIR = Path("data/edinet/list")
MAP_CACHE = Path("data/processed/edinet_code_map.parquet")


def build_code_map() -> pd.Series:
    """edinetCode → secCode（上場発行体・edinet/list 全期間の union・キャッシュ付き）。"""
    if MAP_CACHE.exists():
        m = pd.read_parquet(MAP_CACHE)
        return m.set_index("edinetCode")["secCode"]
    rows = []
    for f in sorted(LIST_DIR.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty or "_empty" in df.columns or "edinetCode" not in df.columns:
            continue
        sub = df[["edinetCode", "secCode"]].dropna()
        if len(sub):
            rows.append(sub)
    m = (pd.concat(rows, ignore_index=True).astype(str)
         .drop_duplicates(subset="edinetCode", keep="last"))
    MAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(MAP_CACHE)
    return m.set_index("edinetCode")["secCode"]


def load_events(code_map: pd.Series, panel_codes: set) -> pd.DataFrame:
    ev = pd.concat([pd.read_parquet(LH), pd.read_parquet(CH)], ignore_index=True)
    ev["submit_dt"] = pd.to_datetime(ev["submit_dt"])
    ev["Date"] = ev["submit_dt"].dt.normalize()
    # sec_code 復元: 直接収録 → 無ければ issuer_edinet からマップ
    sec = ev["sec_code"].astype("string")
    mapped = ev["issuer_edinet"].map(code_map)
    ev["Code"] = sec.fillna(mapped).astype("string")
    ev = ev[ev["Code"].notna()]
    ev["Code"] = ev["Code"].str.strip()
    ev = ev[ev["Code"].isin(panel_codes)]
    # 比率スケール自動判定（fraction か % か）
    scale = 100.0 if float(ev["holding_ratio"].dropna().median()) > 1.0 else 1.0
    ev["d_ratio"] = (ev["holding_ratio"] - ev["holding_ratio_prev"]) / scale * 1.0
    return ev


def event_window(sub: pd.DataFrame, idx: pd.DatetimeIndex, cols: pd.Index,
                 hold: int) -> pd.DataFrame:
    hit = pd.DataFrame(False, index=idx, columns=cols)
    pos = idx.searchsorted(pd.DatetimeIndex(sub["Date"]))
    for p, c in zip(pos, sub["Code"]):
        if p < len(idx) and c in hit.columns:
            hit.iat[p, hit.columns.get_loc(c)] = True
    return hit.rolling(hold, min_periods=1).max() > 0


def ls_weights(event_win: pd.DataFrame, hedge: pd.DataFrame) -> pd.DataFrame:
    """イベント名ロング(Σ=+1) × 流動ヘッジEWショート(Σ=−1)。無イベント日=現金。"""
    ev = event_win.fillna(False)
    ne = ev.sum(axis=1)
    active = ne > 0
    longw = ev.astype(float).div(ne.where(ne > 0), axis=0)
    hp = hedge.fillna(False) & (~ev)
    nh = hp.sum(axis=1)
    shortw = (-hp.astype(float)).div(nh.where(nh > 0), axis=0)
    w = longw.add(shortw, fill_value=0.0)
    w.loc[~active] = np.nan
    return w


def _sr(x: pd.Series, ann: float = 252.0, lo=None, hi=None) -> float:
    r = x.dropna()
    if lo is not None:
        r = r[r.index >= pd.Timestamp(lo)]
    if hi is not None:
        r = r[r.index < pd.Timestamp(hi)]
    if r.size < 8 or float(r.std(ddof=1)) == 0.0:
        return float("nan")
    return float(sharpe_ratio(r) * np.sqrt(ann))


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

    code_map = build_code_map()
    ev = load_events(code_map, set(map(str, cols)))
    print(f"=== 大量保有イベント検証（scope={SCOPE}）===")
    print(f"tradable イベント: {len(ev):,}  {ev['Date'].min():%Y-%m}..{ev['Date'].max():%Y-%m}")
    is_new = ev["report_class"] == "new"
    print(f"  新規={int(is_new.sum()):,}（名簿={int((is_new & ev['in_registry']).sum()):,}・"
          f"重要提案={int((is_new & ev['is_important_proposal']).sum()):,}）  "
          f"積増し(名簿・+{ACCUM_PT:.0%}pt超)="
          f"{int(((~is_new) & ev['in_registry'] & (ev['d_ratio'] >= ACCUM_PT)).sum()):,}")

    traded = turn.notna() & (turn > 0)
    liq_hist = traded.rolling(252, min_periods=120).sum() >= 120
    eligible = traded & liq_hist & adj.notna()
    tadv = turn.rolling(252, min_periods=120).mean()
    rank = tadv.rank(axis=1, ascending=False)
    hedge = (rank <= HEDGE_N) & eligible

    subsets = {
        "act_new_all_h40": ev[is_new],
        "act_new_reg_h40": ev[is_new & ev["in_registry"]],
        "act_new_prop_h40": ev[is_new & ev["is_important_proposal"]],
        "act_accum_reg_h40": ev[(~is_new) & ev["in_registry"] & (ev["d_ratio"] >= ACCUM_PT)],
    }
    strategies = []
    wins = {}
    for name, sub in subsets.items():
        win = event_window(sub, idx, adj.columns, HOLD) & eligible
        wins[name] = win
        strategies.append(PrecomputedWeights(
            ls_weights(win, hedge), name=name,
            params={"n_events": int(len(sub)), "hold": HOLD}))

    no_buy, no_sell = limit_lock_flags(close, high, low, ul, ll, vol_)
    view = AsOfView({"close": adj})

    hyp = ("大量保有報告イベント後の正ドリフト＝filer 異質性（名簿アクティビスト・重要提案行為・"
           "積増しで強い）。イベント名ロング×流動ヘッジのドルニュートラルが日次30bps・"
           "値幅ロック後もα（docs/57・方向は日本主標本 PBFJ2023 で事前固定）")
    rat = ("『誰が・何の目的で』のブロック取得情報はエンゲージメント帰結の不確実性ゆえ開示時点で"
           "完全に織り込まれず段階的に実現（JP の初期CARは米国より小＝ゆっくり織り込まれる余地）。"
           "既試のアクティビスト静的スクリーン・TOB裁定とはイベント軸で別。docs/57 §2.2")

    with default_registry() as reg:
        v = judge_grid(strategies, view, scope=SCOPE, hypothesis=hyp,
                       economic_rationale=rat, registry=reg, costs_bps=COST_BPS,
                       execution_lag=1, adv=turn, participation=0.1,
                       no_buy=no_buy, no_sell=no_sell, short_borrow_bps=BORROW_BPS)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{SCOPE}.html"))

    # ---- 診断（throwaway・K不変）----
    print(f"\n--- 診断: IS/OOS({OOS}〜)・年次・maxDD ---")
    for r in v.results:
        s = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        cum = (1.0 + s).cumprod()
        mdd = float((cum / cum.cummax() - 1.0).min())
        print(f"  {r.name:<20} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={_sr(s, hi=OOS):+.2f} OOS={_sr(s, lo=OOS):+.2f} | maxDD={mdd:.1%}")
    best = v.best.name if v.best else "act_new_reg_h40"
    sb = v.series.get(best, pd.Series(dtype="float64")).dropna()
    print(f"\n--- 年次 net SR（{best}）---")
    for y, seg in sb.groupby(sb.index.year):
        print(f"  {y}: SR={_sr(seg):+.2f}  n={seg.size}")

    # H1 vs 統制（reg − all の差分系列）
    s_reg = v.series.get("act_new_reg_h40", pd.Series(dtype="float64"))
    s_all = v.series.get("act_new_all_h40", pd.Series(dtype="float64"))
    d = (s_reg - s_all).dropna()
    print(f"\n--- H1 filer異質性（名簿 − 全filer）: ΔSR={_sr(s_reg) - _sr(s_all):+.2f}  "
          f"差分系列SR={_sr(d):+.2f} ---")

    # イベントスタディ形状（対ヘッジ超過・平均累積 bps）
    ret_d = adj.pct_change(fill_method=None)
    hedge_ret = ret_d.where(hedge).mean(axis=1)
    print("\n--- イベントスタディ（対ヘッジ超過・平均累積 bps）---")
    for name, sub in subsets.items():
        cum = np.zeros(60)
        n = 0
        for _, r_ in sub.iterrows():
            p = idx.searchsorted(r_["Date"])
            if p + 61 >= len(idx) or r_["Code"] not in adj.columns:
                continue
            rr = ret_d[r_["Code"]].iloc[p + 1:p + 61].values - \
                hedge_ret.iloc[p + 1:p + 61].values
            if np.isnan(rr).all():
                continue
            cum += np.nan_to_num(rr)
            n += 1
        if n:
            c = np.cumsum(cum / n) * 1e4
            print(f"  {name:<20} n={n:>5}  +5d={c[4]:+.0f} +10d={c[9]:+.0f} "
                  f"+20d={c[19]:+.0f} +40d={c[39]:+.0f} +60d={c[59]:+.0f}")

    # F1 診断: value 因子（月次B/M L/S）との相関・対象の momentum 分位（H-16）
    try:
        from invest_system.equities.fundamentals import fundamentals_panel
        per = idx.to_period("M")
        me = pd.DatetimeIndex(pd.Series(idx, index=idx).groupby(per).max().values)
        pit = fundamentals_panel(me, ["Eq", "ShOutFY"])
        bm = (pit["Eq"] / (pit["ShOutFY"] * close.reindex(me))).replace(
            [np.inf, -np.inf], np.nan)
        ret_m = adj.reindex(me).pct_change(fill_method=None)
        rk = bm.shift(1).rank(axis=1, pct=True)
        val_ls = ((ret_m * (rk >= 0.8).div((rk >= 0.8).sum(axis=1), axis=0)).sum(axis=1)
                  - (ret_m * (rk <= 0.2).div((rk <= 0.2).sum(axis=1), axis=0)).sum(axis=1))
        sb_m = (1.0 + sb).groupby(sb.index.to_period("M")).prod() - 1.0
        val_m = pd.Series(val_ls.values, index=pd.PeriodIndex(val_ls.index, freq="M"))
        both = pd.concat([sb_m, val_m], axis=1).dropna()
        print(f"\n--- F1 診断: ρ(最良セル月次, value L/S 月次) = "
              f"{both.iloc[:, 0].corr(both.iloc[:, 1]):+.2f} ---")
    except Exception as e:  # noqa: BLE001
        print(f"\n（value 相関診断スキップ: {e}）")
    mom12 = (adj.shift(21) / adj.shift(252) - 1.0)
    pct = mom12.rank(axis=1, pct=True)
    vals = []
    for _, r_ in subsets["act_new_reg_h40"].iterrows():
        p = idx.searchsorted(r_["Date"])
        if p < len(idx) and r_["Code"] in pct.columns:
            x = pct.iloc[p][r_["Code"]]
            if pd.notna(x):
                vals.append(float(x))
    if vals:
        print(f"--- H-16 診断: 名簿filer新規イベント銘柄の momentum(12-1) 分位 "
              f"中央値={np.median(vals):.2f}（0.5=中立）---")

    # コスト/貸株感応（主セル）
    st = next(s_ for s_ in strategies if s_.name == "act_new_reg_h40")
    print(f"\n--- コスト感応（act_new_reg_h40・貸株115bps）---")
    for c in [0, 15, 30, 50]:
        res = backtest(st, view, costs_bps=float(c), execution_lag=1, adv=turn,
                       participation=0.1, no_buy=no_buy, no_sell=no_sell,
                       short_borrow_bps=BORROW_BPS)
        print(f"  {c:>3}bps  net年率SR={_sr(res.returns.dropna()):+.2f}  "
              f"回転={res.turnover.mean():.3f}")

    print("\n※ 判定は scope=activist_filing_drift の DSR（K=4・docs/57 §2.6）。診断は throwaway。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
