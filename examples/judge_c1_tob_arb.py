"""C1（TOB リスクアーブ）scope `tob_arb` の正式 DSR 裁定（Path B・docs/09 §3,§5）。

設計判断（ユーザー確定 2026-06）：
- 統合方式＝**イベントリターン直接 DSR（Path B）**。汎用バックテストエンジンは TOB の
  **テンダー決済**（成立＝買付価格で応募・市場で売らない）を表現できず成立益を取りこぼし
  符号が反転するため不採用（実測：同セルで市場出口 -6.0% vs テンダー +7.3%）。
- 各案件の実現リターンを日次 M2M 系列に：**T+1 始値**エントリー、成立＝最終(買付)価格で
  テンダー、不成立/撤回＝結果公表（終了日＋バッファ取引日）の市場価格（ギャップ取込）。
- **約定不能（T+1 ストップ高ロック）はスケジュール段でスキップ**（23%＝診断と一致。
  エンジンの no_buy キャリーは高値追いになるため使わない）。`frictions.limit_lock_flags`。
- 資金配分＝同時進行 N 件へ等加重・空き(0件)日は現金（キャッシュドラッグ反映）。
- K 規律＝STEP3 探索（スプレッド帯スキャン）を `log_scan_trials` で K に算入しデフレート。

格子 K=8：サブ期間{2016-2019, 2020-2026}×競合{除外, 含む}×スプレッド閾値{>=1%, >=3%}。
判定：各セルの日次系列で per-period Sharpe→`deflated_sharpe`（scope の K と V[SR]）、
PSR・MinTRL、8 セルの PBO(CSCV)、MinBTL。

usage:
  python examples/judge_c1_tob_arb.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_c1_tob_arb as base                            # noqa: E402
from invest_system.equities.frictions import limit_lock_flags  # noqa: E402
from invest_system.validation.dsr import (                    # noqa: E402
    _moments, min_backtest_length, probabilistic_sharpe_ratio)
from invest_system.validation.pbo import pbo_cscv             # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

JQ_DAILY = "data/jquants/daily"
FAIL_BUFFER = 5
EXTRA_TRIALS = 4          # STEP3 探索（スプレッド帯 4 バケット）を K に算入（K規律）
SCOPE = "tob_arb"
DSR_PASS = 0.95
ANN = 252.0
SUBVIEWS = {"2016-2019": ("2016-01-01", "2020-06-30"),
            "2020-2026": ("2020-01-01", None)}

HYPOTHESIS = ("現金 TOB 公表後に対象株を T+1 始値で建て、約定可能（値幅制限非ロック）な案件を"
              "保有しテンダー決済すると、成立時は買付価格への収束益、リスクは不成立時の"
              "発表前水準への急落。リスクプレミアム型でレジーム（不成立率）依存。")
RATIONALE = ("スプレッド=不成立リスクのプレミアム（merger arb）。約定フリクション(23%ロック)・"
             "高スプレッドの不成立セレクション・競合価格リーク・レジーム上昇を踏まえ、"
             "naive な高スプレッド追随は棄却寄り＝中スプレッド・非競合・約定可能で薄いエッジ。")


def wide_panels(codes: set, start: str) -> dict:
    fields = {"C": "close", "O": "open", "H": "high", "L": "low", "UL": "ul",
              "LL": "ll", "Vo": "vol", "Va": "val"}     # Va=売買代金(¥)＝容量算出
    rows = []
    for p in sorted(glob.glob(f"{JQ_DAILY}/*.parquet")):
        if Path(p).stem < start:
            continue
        df = pd.read_parquet(p)
        if "Code" not in df.columns or "C" not in df.columns or df.empty:
            continue
        rows.append(df[df["Code"].isin(codes)][["Date", "Code"]
                                               + [c for c in fields if c in df.columns]])
    long = pd.concat(rows, ignore_index=True)
    long["Date"] = pd.to_datetime(long["Date"])
    return {name: long.pivot_table(index="Date", columns="Code", values=src,
                                   aggfunc="last").sort_index()
            for src, name in fields.items() if src in long.columns}


def build_schedule(d: pd.DataFrame, panels: dict) -> pd.DataFrame:
    close, opn = panels["close"], panels["open"]
    no_buy, _ = limit_lock_flags(close, panels["high"], panels["low"],
                                 panels["ul"], panels["ll"], panels.get("vol"))
    rows = []
    for _, r in d.iterrows():
        sec = r["sec"]
        if sec is None or sec not in close.columns:
            continue
        cdates = close[sec].dropna().index
        after = cdates[cdates > r["announce_date"].normalize()]
        if len(after) == 0:
            continue
        entry = after[0]
        eo = opn[sec].get(entry) if sec in opn.columns else None
        if eo is None or pd.isna(eo) or float(eo) <= 0:
            continue
        if entry in no_buy.index and sec in no_buy.columns and bool(no_buy.at[entry, sec]):
            continue                                  # T+1 ストップ高ロック＝約定不能スキップ
        end = pd.to_datetime(r.get("end_date_displayed"), errors="coerce")
        if pd.isna(end):
            continue
        onafter = cdates[cdates >= end.normalize()]
        exit_d = onafter[0] if len(onafter) else cdates[-1]
        if r["tob_result"] in ("不成立", "撤回") and len(onafter):
            exit_d = cdates[min(cdates.get_loc(exit_d) + FAIL_BUFFER, len(cdates) - 1)]
        if exit_d <= entry:
            continue
        rows.append({"sec": sec, "entry_date": entry, "exit_date": exit_d,
                     "entry_open": float(eo), "final_price": float(r["final_price"]),
                     "arb_spread": float(r["initial_price"]) / float(eo) - 1.0,
                     "subperiod": r["subperiod"], "competing": bool(r["competing"]),
                     "result": r["tob_result"]})
    return pd.DataFrame(rows)


def deal_daily(deal: pd.Series, close: pd.DataFrame) -> pd.Series:
    """1 案件の日次 M2M リターン：始値エントリー→成立はテンダー(買付価格)/不成立は市場。"""
    seg = close[deal["sec"]].dropna().loc[deal["entry_date"]:deal["exit_date"]]
    if seg.empty:
        return pd.Series(dtype="float64")
    out = {}
    prev = deal["entry_open"]
    for dt, px in seg.items():
        if dt == deal["exit_date"] and deal["result"] == "成立":
            out[dt] = deal["final_price"] / prev - 1.0        # テンダー（買付価格で応募）
        else:
            out[dt] = float(px) / prev - 1.0
        prev = float(px)
    return pd.Series(out)


def cell_series(deals: pd.DataFrame, close: pd.DataFrame,
                index: pd.DatetimeIndex) -> pd.Series:
    """セル日次ポートフォリオ：ライブ案件に等加重、0 件日は現金（ドラッグ反映）。"""
    if deals.empty:
        return pd.Series(0.0, index=index)
    mat = pd.DataFrame({i: deal_daily(r, close) for i, (_, r) in enumerate(deals.iterrows())})
    pf = mat.reindex(index).mean(axis=1, skipna=True)          # 等加重 1/N_active
    return pf.fillna(0.0)


def cell_capacity(deals: pd.DataFrame, val: pd.DataFrame, participation: float = 0.10,
                  adv_lookback: int = 60, q: float = 0.25) -> float:
    """セルの実用 AUM(¥)：案件別の拘束 AUM = p × ADV_d × min_t(N_active,t∈hold_d) の
    下側分位点（既定 p25＝75% の案件が参加率内に収まる規模・docs/09 §6.4）。

    strict-min は単一の超低流動 micro-cap（held alone）に支配され無意味＝そういう銘柄は
    スキップ前提で分位点を採る。ADV_d=エントリー前 adv_lookback 取引日の売買代金中央値
    （PIT・取引前のみ）。同時建玉が少ない（集中）日ほど weight が大きく制約がきつい。
    """
    if deals.empty:
        return float("nan")
    iv = [(r["entry_date"], r["exit_date"]) for _, r in deals.iterrows()]

    def n_active(day):
        return sum(1 for e, x in iv if e <= day < x) or 1

    caps = []
    for _, r in deals.iterrows():
        if r["sec"] not in val.columns:
            continue
        pre = val[r["sec"]].dropna()
        pre = pre[pre.index < r["entry_date"]].tail(adv_lookback)
        adv = float(pre.median()) if len(pre) else float("nan")
        if not (adv > 0):
            continue
        hold = [d for d in val.index if r["entry_date"] <= d < r["exit_date"]]
        caps.append(participation * adv * min((n_active(d) for d in hold), default=1))
    return float(pd.Series(caps).quantile(q)) if caps else float("nan")


def _fmt_jpy(x: float) -> str:
    if not (x == x):
        return "—"
    if x >= 1e8:
        return f"¥{x/1e8:.1f}億"
    return f"¥{x/1e4:.0f}万"


def cells_of(sched_sub: pd.DataFrame) -> list:
    out = []
    for comp_label, comp_vals in [("excl", [False]), ("incl", [True, False])]:
        for thr in (0.01, 0.03):
            sub = sched_sub[sched_sub["competing"].isin(comp_vals)
                            & (sched_sub["arb_spread"] >= thr)]
            out.append((f"tob_arb({comp_label},spread>={int(thr*100)}%)",
                        {"competing": comp_label, "spread_min": thr}, sub))
    return out


def main() -> int:
    d = base.join_edinet(base.load_spine())
    panels = wide_panels(set(d["sec"].dropna()),
                         (d["announce_date"].min() - pd.Timedelta(days=10)).strftime("%Y%m%d"))
    close = panels["close"]
    sched = build_schedule(d, panels)
    print(f"スケジュール {len(sched)} 件（約定可能・ロック除外後）／"
          f"サブ期間 {sched['subperiod'].value_counts().to_dict()}")

    reg = default_registry()
    reg.log_scan_trials(scope=SCOPE, count=EXTRA_TRIALS, hypothesis=HYPOTHESIS,
                        rationale=RATIONALE)
    series_all, recs = {}, []
    for sp, (s0, s1) in SUBVIEWS.items():
        idx = close.loc[pd.Timestamp(s0):(pd.Timestamp(s1) if s1 else None)].index
        for cname, cparams, sub in cells_of(sched[sched["subperiod"] == sp]):
            name = f"{sp}/{cname}"                     # サブ期間を試行 ID に含め指紋衝突を防ぐ
            params = {"subperiod": sp, **cparams}
            pf = cell_series(sub, close, idx)
            series_all[name] = pf
            sr, sk, ku, n = _moments(pf.values)
            uid = reg.log_trial(scope=SCOPE, strategy_id=name, params=params,
                                sharpe=sr, n_obs=n, skew=sk, kurt=ku,
                                hypothesis=HYPOTHESIS, rationale=RATIONALE)
            recs.append({"cell": name, "n_deals": len(sub), "n_obs": n,
                         "sr_ann": sr * np.sqrt(ANN),
                         "psr": probabilistic_sharpe_ratio(sr, 0.0, n, sk, ku),
                         "dsr": reg.deflated_sharpe(uid),
                         "mean_ann": pf.mean() * ANN,
                         "cap": cell_capacity(sub, panels["val"]),
                         "uid": uid})

    k = reg.trial_count(SCOPE)
    aligned = pd.DataFrame(series_all).fillna(0.0)
    pbo = pbo_cscv(aligned).pbo if aligned.shape[1] >= 2 else float("nan")
    mbtl = min_backtest_length(k, 1.0)

    print(f"\n=== scope `{SCOPE}` 判定（K={k}＝格子8＋scan{EXTRA_TRIALS}・DSR闾値 {DSR_PASS}）===")
    print(f"{'cell':34s} {'n':>3} {'SR_ann':>7} {'平均/年':>7} {'DSR':>5} {'容量@10%':>9}")
    passed = []
    for r in recs:
        flag = " ◎PASS" if r["dsr"] >= DSR_PASS else ""
        if r["dsr"] >= DSR_PASS:
            passed.append(r["cell"])
        print(f"{r['cell']:34s} {r['n_deals']:3d} {r['sr_ann']:+7.2f} "
              f"{r['mean_ann']:+6.1%} {r['dsr']:5.2f} {_fmt_jpy(r['cap']):>9}{flag}")
    print(f"\nPBO(CSCV, 8セル)={pbo:.2f}  MinBTL={mbtl:.1f}年  累積K={k}")
    print(f"判定：{'PASS ' + ','.join(passed) if passed else '全セル FAIL（DSR<%.2f）' % DSR_PASS}")
    print("\n注：成立=買付価格テンダー・不成立=ギャップ取込・始値エントリー・ロック23%スキップ・"
          "等加重キャッシュドラッグ。容量@10%＝案件別拘束AUM(p×ADV×min同時建玉)のp25"
          "（75%の案件が参加率10%内・残りはスキップ）＝小資本限定でスケールしない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
