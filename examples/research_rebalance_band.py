"""P2-A: リバランス・デッドバンドの感応表 — docs/04 P2-A・F1(arXiv:2606.00060)/D2。

「|目標 − 保有| < band の銘柄は取引しない」cost-aware 実行フィルターが、旗艦
（switch / wf_switch）のネット成績をどう動かすかを §6.13 シナリオ B〜E の各現実性
レベル × band ∈ {0, 0.25%, 0.5%, 1.0%} で測定する。

規律（§6.12-6.13 と同じ throwaway）：
- 意思決定（PIT ウェイト・wf 割当）は §6.9-6.11 から**一切変更しない**。band は執行
  フィルタでありシグナルではない。wf の割当学習も band=0 のスリーブ・ネットで固定
  （band を割当に混ぜない）。
- **最良セルを選んで採用しない**（頑健性≠最適化・§6.11 の規律）。band が広域で単調に
  ネット改善するかだけを見る。K 不変・永続レジストリ不使用。

実行: .venv\\Scripts\\python.exe examples\\research_rebalance_band.py
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
from invest_system.equities import events  # noqa: E402
from invest_system.equities.frictions import vol_scaled_cost_bps  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.panel import (  # noqa: E402
    assemble_panel, fetch_month_end_snapshots, load_daily_panel,
)
from invest_system.equities.fundamentals import load_fundamentals, point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, RegimeSwitch, apply_rebalance_band,
    open_fill_backtest, walk_forward_regime_assignment,
)
from invest_system.timeseries import vol_regime  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
WF_WARMUP, WF_MINOBS = 24, 6
BANDS = [0.0, 0.0025, 0.005, 0.01]


def _sr(x: pd.Series, oos: bool = False) -> float:
    r = x.dropna()
    if oos:
        r = r[r.index >= pd.Timestamp(OOS)]
    return float(sharpe_ratio(r) * np.sqrt(12)) if r.size >= 8 else float("nan")


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1

    # --- 意思決定の組立（§6.13 research_value_pead_timing.py と同一・PIT）---
    print("データ組立中（§6.13 と同一・意思決定は不変）...")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=300, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})
    fund = load_fundamentals(superset)

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, umask),
                                                        sector))
    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    value = zN(value_quality_size_factors(pit, raw, adj)["book_to_market"])
    pead = zN(point_in_time(events.forecast_revision(fund), rebal, ["fcst_revision"],
                            date_col="DiscDate", lag_days=1)["fcst_revision"]
              .reindex(columns=superset))
    daily_close = load_daily_panel(field="AdjC")
    vol_m = vol_regime(daily_close).reindex(rebal, method="ffill")

    value_ls = CrossSectionalStrategy(value, 0.2, name="value")
    pead_lt = CrossSectionalStrategy(pead, 0.2, name="pead_longtilt", long_only=True)
    switch = RegimeSwitch(vol_m, {0: value_ls, 1: value_ls, 2: pead_lt},
                          name="switch")
    W = {s.name: {t: s.target_weights(view.asof(t)) for t in rebal}
         for s in (value_ls, pead_lt, switch)}

    cols = [c for c in daily_close.columns if c in set(superset)]
    open_d = load_daily_panel(field="AdjO").reindex(columns=cols)
    vol_d = daily_close[cols].pct_change().rolling(20).std()
    cost_k05 = vol_scaled_cost_bps(vol_d, base_bps=10.0, k=0.05)
    cost_k10 = vol_scaled_cost_bps(vol_d, base_bps=10.0, k=0.10)

    scenarios = [
        ("B: T+1始値・15bp固定", dict(costs_bps=15.0)),
        ("C: T+1始値・ボラ連動(k=0.05)", dict(costs_bps=cost_k05)),
        ("D: T+1始値・ボラ連動(k=0.10)", dict(costs_bps=cost_k10)),
        ("E: D + 貸株115bp（全部入り）",
         dict(costs_bps=cost_k10, short_borrow_bps=115.0)),
    ]

    # wf 割当：各シナリオの band=0 スリーブ・ネットから過去のみで学習（§6.13 と同一・
    # band は割当に混ぜない＝意思決定は band 非依存に固定）
    wf_dates = rebal[WF_WARMUP:]
    AW_by_scenario: dict[str, dict] = {}
    for label, kw in scenarios:
        rv = open_fill_backtest(W["value"], open_d, name="value", **kw).returns
        rp = open_fill_backtest(W["pead_longtilt"], open_d, name="pead", **kw).returns
        assign = walk_forward_regime_assignment(
            {"value": rv.dropna(), "pead_longtilt": rp.dropna()}, vol_m,
            min_obs=WF_MINOBS, warmup=WF_WARMUP)
        AW_by_scenario[label] = {
            t: (W[assign[t]][t] if isinstance(assign.get(t), str)
                else pd.Series(dtype="float64")) for t in wf_dates}

    print(f"=== リバランス・デッドバンド感応表（{START}〜{END}・月次・T+1始値・"
          f"意思決定は §6.9-6.11 と同一）===")
    for nm in ("switch", "wf_switch"):
        print(f"\n--- {nm} ---")
        print(f"  {'シナリオ':<24} " + " ".join(
            f"| band={b:.2%}: 回転/コスト/SR全/SR_OOS" for b in BANDS))
        for label, kw in scenarios:
            cells = []
            for band in BANDS:
                if nm == "switch":
                    wd = apply_rebalance_band(W["switch"], band)
                else:
                    wd = apply_rebalance_band(AW_by_scenario[label], band)
                res = open_fill_backtest(wd, open_d, name=nm, **kw)
                r = res.returns.dropna()
                cost_pa = float((res.gross.reindex(r.index) - r).mean()) * 12.0
                cells.append((float(res.turnover.mean()), cost_pa,
                              _sr(r), _sr(r, oos=True)))
            print(f"  {label:<24} " + " ".join(
                f"| {t:4.2f} {c:6.2%} {s:+5.2f} {o:+5.2f}"
                for t, c, s, o in cells))

    print("\n※ 読み方：band=0 が §6.13 の基準（B〜E）。回転（月次平均 Σ|Δw|）と年率実効"
          "コスト（gross−net）が band でどれだけ落ち、SR がどう動くかを見る。**広域で単調に"
          "ネット改善するか**だけが論点（最良セルの選択はしない＝頑健性≠最適化）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
