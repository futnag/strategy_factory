"""新戦略種を判定器へ：③カレンダー/季節性（月末効果）④ペア/相対価値。

③ 日次TOPIXに対し turn-of-month（月末・月初ロング）を格子で判定。
④ 2銘柄（例：トヨタ72030 / ホンダ72670）の対数比 z-score 平均回帰ペアを判定。
いずれも執行ラグ1（翌足執行）・永続デフレートDSR・HTML出力。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_calendar_pairs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, CalendarStrategy, PairsStrategy, judge_grid, write_html,
)
from invest_system.validation.registry import default_registry  # noqa: E402

START, END = "2018-01-01", "2026-05-31"
PAIR = ("72030", "72670")   # トヨタ / ホンダ（自動車）


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    with default_registry() as reg:
        # ③ カレンダー（月末効果）on TOPIX
        topix = jq.fetch_index_bars(code="0000")
        tp = (topix.dropna(subset=["Date"]).set_index("Date")["C"].sort_index()
              .to_frame("0000"))
        tp = tp[~tp.index.duplicated(keep="last")]
        view_cal = AsOfView({"close": tp})
        grid_cal = [CalendarStrategy("0000", dom_start=ds, dom_end=de)
                    for ds in (25, 27) for de in (2, 3)]
        v1 = judge_grid(grid_cal, view_cal, scope="calendar_tom",
                        hypothesis="月末・月初は給与/年金等の需給で株式が上昇しやすい(turn-of-month)",
                        economic_rationale="月次の機関フロー・積立買いによる季節的な需給の偏り",
                        registry=reg, costs_bps=10.0, execution_lag=1)
        print("\n########## ③ カレンダー（月末効果・TOPIX）##########")
        print(v1.report_md)
        print("HTML:", write_html(v1, f"data/reports/{v1.scope}.html"))

        # ④ ペア（相対価値）
        frames = {}
        for c in PAIR:
            try:
                frames[c] = jq.fetch_daily_history(c, frm=START, to=END)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {c}: {str(e)[:60]}")
        close = pd.DataFrame({
            c: f.dropna(subset=["Date"]).set_index("Date")["AdjC"]
            for c, f in frames.items()})
        close = close[~close.index.duplicated(keep="last")].sort_index().dropna()
        view_pairs = AsOfView({"close": close})
        grid_pairs = [PairsStrategy(PAIR[0], PAIR[1], lookback=lb, entry=e)
                      for lb in (40, 60) for e in (1.5, 2.0)]
        v2 = judge_grid(grid_pairs, view_pairs, scope=f"pairs_{PAIR[0]}_{PAIR[1]}",
                        hypothesis="同業2銘柄の対数比は平均回帰し、乖離時の逆張りが収益化する",
                        economic_rationale="同一ファンダ要因を共有する銘柄間スプレッドの定常性を仮定",
                        registry=reg, costs_bps=15.0, execution_lag=1)
        print("\n########## ④ ペア（相対価値・トヨタ/ホンダ）##########")
        print(f"日次 {close.shape[0]}本 × {close.shape[1]}銘柄")
        print(v2.report_md)
        print("HTML:", write_html(v2, f"data/reports/{v2.scope}.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
