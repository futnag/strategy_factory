"""日次 run-up プレミアム（正しい粒度）：決算発表の数日前にロング→発表前に手仕舞い。

events.days_to_next_announcement（DiscDateから次回発表までの予測日数, PIT）を使い、
EarningsRunup（pre日前〜lag日前の銘柄をロング/窓外をショート）を日次で判定器にかける。
発表日（ジャンプ）は跨がない。execution_lag=1（翌足執行）。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_runup_daily.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import events  # noqa: E402
from invest_system.equities.universe import filter_common_stocks  # noqa: E402
from invest_system.research import AsOfView, EarningsRunup, judge_grid, write_html  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END = "2019-01-01", "2026-05-31"
TOP_N = int(get_env("J_RUNUP_TOPN", "120") or "120")
LIQ_DATE = "20260225"


def build_panel(frames, field):
    cols = {}
    for c, df in frames.items():
        if field in df.columns and "Date" in df.columns:
            s = df.dropna(subset=["Date"]).set_index("Date")[field]
            cols[c] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(cols).sort_index()


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    listed = jq.fetch_listed_info()
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    snap = jq.fetch_daily_quotes(LIQ_DATE)
    snap = snap[snap["Code"].astype(str).isin(common)]
    codes = (snap.sort_values("Va", ascending=False)["Code"].astype(str)
             .head(TOP_N).tolist())
    print(f"ユニバース 流動性上位{len(codes)}銘柄  期間 {START}〜{END}")

    print("日次履歴を取得中（銘柄別・初回のみ実取得）…")
    frames = {}
    for i, c in enumerate(codes, 1):
        try:
            d = jq.fetch_daily_history(c, frm=START, to=END)
            if not d.empty:
                frames[c] = d
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {c}: {str(e)[:50]}")
        if i % 40 == 0:
            print(f"  {i}/{len(codes)}")
    adjc = build_panel(frames, "AdjC")
    advp = build_panel(frames, "Va").reindex(adjc.index)
    print(f"日次パネル {adjc.shape[0]}本 × {adjc.shape[1]}銘柄")

    print("財務（DiscDate）取得中…")
    fund = []
    for c in adjc.columns:
        try:
            st = jq.fetch_statements(code=c)
            if not st.empty:
                fund.append(st)
        except Exception:  # noqa: BLE001
            pass
    fund = pd.concat(fund, ignore_index=True) if fund else pd.DataFrame()
    days = events.days_to_next_announcement(fund, adjc.index).reindex(
        index=adjc.index, columns=adjc.columns)
    print(f"発表予測日数パネル: 非NaN平均 {days.notna().sum(axis=1).mean():.0f}銘柄/日")

    view = AsOfView({"close": adjc})
    grid = [EarningsRunup(days, pre=p, lag=lag)
            for p in (15, 20, 30) for lag in (1, 3)]
    print(f"格子 {len(grid)}通り（pre×lag）\n")
    with default_registry() as reg:
        v = judge_grid(grid, view, scope="earn_runup_daily",
                       hypothesis="決算発表の数日前にロングし発表前に手仕舞うと超過リターン(announcement run-up)",
                       economic_rationale="発表前の注目・リスクプレミアム上昇による run-up を、ジャンプを跨がず捕捉",
                       registry=reg, costs_bps=15.0, execution_lag=1, adv=advp,
                       participation=0.1)
    print(v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
