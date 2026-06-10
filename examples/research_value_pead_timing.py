"""執行タイミング・スリッページ現実性チェック：日足の3つの罠に最有力候補は耐えるか。

日足 OHLCV バックテストの構造的な甘さ（docs/03 §6.13）への回答：
  罠② 同足先読み … 月次研究は「月末終値を見て同じ終値で約定」(execution_lag=0) だった。
      → `open_fill_backtest`：同じ PIT 意思決定を**翌営業日の始値**で約定（open→open 実現）
        ＝タイムトラベル排除＋寄りギャップ（窓開け）の負担。
  罠③ 日中流動性の偏り … 固定 15bps では荒れ相場の板薄・スリッページ増を見ない。
      → `vol_scaled_cost_bps`：銘柄×日付の実現σに連動して約定を不利に滑らせる
        （ATR連動ペナルティ。switch は高ボラ月にこそ PEAD を建てるため、最も刺さる検証）。
  罠① 高安の発生順序 … 本リポジトリの戦略はバー内 PT/SL を使わないため非該当
      （トリプルバリアは終値パス）。イベント系での将来利用に備え、悲観モード
      （H/L 接触・同足は損切り優先）を labeling/triple_barrier.py に実装済み。

規律（§6.6/§6.12 と同じ throwaway 測定）：意思決定（PIT ウェイト）は §6.9-6.11 から
**一切変更せず**、約定タイミングとコストモデルのみを切替＝新たな選択なし（K 不変・
永続レジストリ不使用）。

実行: .venv\\Scripts\\python.exe examples\\research_value_pead_timing.py
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
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.portfolio import kelly_fraction  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, RegimeSwitch, Strategy, backtest,
    open_fill_backtest, walk_forward_regime_assignment,
)
from invest_system.timeseries import vol_regime  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
WF_WARMUP, WF_MINOBS = 24, 6


def _sr(x: pd.Series, oos: bool = False) -> float:
    r = x.dropna()
    if oos:
        r = r[r.index >= pd.Timestamp(OOS)]
    return float(sharpe_ratio(r) * np.sqrt(12)) if r.size >= 8 else float("nan")


def _maxdd(r: pd.Series) -> float:
    cum = (1.0 + r.dropna()).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def _row(name: str, r: pd.Series, turn: pd.Series) -> str:
    (_, pre), (_, post) = pre_post_sharpe(r, "2020-01-01")
    return (f"  {name:<16} {_sr(r):>+7.2f} {_sr(r, oos=True):>+8.2f} "
            f"{_maxdd(r):>7.1%} {pre:>+6.2f}/{post:>+5.2f} "
            f"{float(turn.reindex(r.index).mean()):>6.2f}")


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1

    # --- 意思決定の組立（research_value_pead_regime.py と同一・PIT）---
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
    # PIT ウェイトを一度だけ生成（全シナリオで共通＝意思決定は不変）
    W = {s.name: {t: s.target_weights(view.asof(t)) for t in rebal}
         for s in (value_ls, pead_lt, switch)}

    # --- 約定用の日次データ（調整後始値）とボラ連動コスト ---
    cols = [c for c in daily_close.columns if c in set(superset)]
    open_d = load_daily_panel(field="AdjO").reindex(columns=cols)
    vol_d = daily_close[cols].pct_change().rolling(20).std()
    cost_k05 = vol_scaled_cost_bps(vol_d, base_bps=10.0, k=0.05)
    cost_k10 = vol_scaled_cost_bps(vol_d, base_bps=10.0, k=0.10)
    med05 = float(np.nanmedian(cost_k05.to_numpy()))
    med10 = float(np.nanmedian(cost_k10.to_numpy()))
    print(f"=== 執行タイミング・スリッページ現実性（{START}〜{END}・月末{len(rebal)}回）===")
    print(f"ボラ連動コストの中央値: k=0.05 → {med05:.1f}bp / k=0.10 → {med10:.1f}bp"
          f"（固定15bpと比較可能な水準）")

    scenarios = [
        ("A: 同足終値執行・15bp固定（§6.10 再現）", "engine", dict(costs_bps=15.0)),
        ("B: T+1始値執行・15bp固定（窓開け負担）", "open", dict(costs_bps=15.0)),
        ("C: T+1始値・ボラ連動コスト(k=0.05)", "open", dict(costs_bps=cost_k05)),
        ("D: T+1始値・ボラ連動コスト(k=0.10)", "open", dict(costs_bps=cost_k10)),
        ("E: D + 貸株115bp（全部入り）", "open",
         dict(costs_bps=cost_k10, short_borrow_bps=115.0)),
    ]

    nets: dict[str, dict[str, pd.Series]] = {}
    for label, mode, kw in scenarios:
        nets[label] = {}
        print(f"\n=== {label} ===")
        print(f"  {'strategy':<16} {'SR(全)':>7} {'SR(OOS)':>8} {'maxDD':>7} "
              f"{'前/後2020':>13} {'回転':>6}")
        for nm in ("value", "pead_longtilt", "switch"):
            if mode == "engine":
                strat = {"value": value_ls, "pead_longtilt": pead_lt,
                         "switch": switch}[nm]
                res = backtest(strat, view, **kw)
            else:
                res = open_fill_backtest(W[nm], open_d, name=nm, **kw)
            r = res.returns.dropna()
            nets[label][nm] = r
            print(_row(nm, r, res.turnover))
        # walk-forward 適応切替（割当は当該シナリオのネットから過去のみで学習）
        Rv, Rp = nets[label]["value"], nets[label]["pead_longtilt"]
        assign = walk_forward_regime_assignment(
            {"value": Rv, "pead_longtilt": Rp}, vol_m,
            min_obs=WF_MINOBS, warmup=WF_WARMUP)
        AW = {t: (W[assign[t]][t] if isinstance(assign.get(t), str)
                  else pd.Series(dtype="float64")) for t in rebal}
        wf_dates = rebal[WF_WARMUP:]
        if mode == "engine":
            class _Replay(Strategy):
                name, params = "wf_switch", {}

                def target_weights(self, asof):
                    return AW.get(asof.asof, pd.Series(dtype="float64"))
            res = backtest(_Replay(), view, rebalance=wf_dates, **kw)
        else:
            res = open_fill_backtest({t: AW[t] for t in wf_dates}, open_d,
                                     name="wf_switch", **kw)
        r = res.returns.dropna()
        nets[label]["wf_switch"] = r
        print(_row("wf_switch", r, res.turnover))

    # --- ギャップ効果の分解（B−A）とケリー（全部入り E 基準・DP16）---
    print("\n--- 同足執行バイアスの大きさ（A→B の差・年率SR）---")
    for nm in ("value", "pead_longtilt", "switch", "wf_switch"):
        a = _sr(nets[scenarios[0][0]][nm])
        b = _sr(nets[scenarios[1][0]][nm])
        print(f"  {nm:<16} {a:+.2f} → {b:+.2f}  (Δ {b - a:+.2f})")

    full = scenarios[-1][0]
    print(f"\n=== フラクショナル・ケリー（『{full}』のネットから・DP16）===")
    for nm in ("switch", "wf_switch"):
        for frac in (0.5, 0.25):
            k = kelly_fraction(nets[full][nm], fraction=frac)
            print(f"  {nm:<10} {frac:>4.2f}×Kelly: {k.summary()}")

    print("\n※ 読み方：A→B＝「終値を見て同じ終値で約定」していたバイアス＋寄りギャップの"
          "実費。B→C/D＝荒れ相場の板薄をボラ連動で滑らせた実費（switch は高ボラ月に建玉"
          "するため最も不利な検証）。E が全部入りの保守ケース。E でも OOS 正なら、"
          "§6.10-6.12 の結論は日足データの甘さに対して頑健と言える。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
