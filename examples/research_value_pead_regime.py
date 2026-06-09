"""仮説検証：value/PEAD にレジーム条件を重ねて分離→改善するか（柱C × レジーム）。

§6.8 で確立した規律「**ゲート前に regime_breakdown で P&L のレジーム分離を確認**」を、エッジを
持つ戦略に適用する。value は唯一の耐久候補（§6.4 DSR0.79-0.84）、PEAD は IS強→OOS脆弱（§7）。
レジームで P&L が分離するなら、`RegimeGated`（＝ルールベースのメタラベル＝「この賭けに乗るか」）で
OOS安定/DSR を改善できるはず。分離が無ければ §6.8 同様の正直な負＝レジームは万能でないことを示す。

レジーム定義（事前固定・PIT・経済的根拠）：value は景気循環/ディストレス寄り→
 ① 高ボラ回避（flight-to-quality でバリュー劣後）② 強トレンド回避（モメンタム相場でバリュー劣後）。
日次マーケットの Efficiency Ratio / 実現ボラを拡張窓三分位化し、月末に as-of 整合（reindex ffill＝
先読み無）。regime[t] は月末 close[t] 由来＝因子と同 as-of。定義は事前固定＝定義探索で K を水増し
しない（KB §11.7）。ML メタラベリングは月数≈120 で過学習リスク高→まず規律版（規則メタラベル）。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_value_pead_regime.py
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
from invest_system.research import (  # noqa: E402
    AsOfView, CompositeStrategy, CrossSectionalStrategy, RegimeGated,
    judge_grid, regime_breakdown, write_html,
)
from invest_system.timeseries import trend_regime, vol_regime  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import (  # noqa: E402
    TrialRegistry, default_registry,
)

START, END, OOS = "2016-07", "2026-05", "2024-01"
SCOPE = get_env("J_VPR_SCOPE", "value_pead_regime") or "value_pead_regime"
REG_PATH = get_env("J_VPR_REGISTRY", None)


def _brk(name: str, series: pd.Series, trend_m: pd.Series, vol_m: pd.Series) -> None:
    """baseline 戦略の月次 P&L をトレンド/ボラ・レジーム別に年率Sharpe分解（ann=12）。"""
    fmt = lambda x: f"{x:+.2f}"  # noqa: E731
    bt = regime_breakdown(series, trend_m, ann=12.0).set_index("regime")
    bv = regime_breakdown(series, vol_m, ann=12.0).set_index("regime")
    def row(b):
        return "  ".join(f"r{int(k)}:SR{fmt(b.loc[k,'sharpe_ann'])}(n{int(b.loc[k,'n'])})"
                         for k in b.index)
    print(f"  [{name:<13}] トレンド {row(bt)}")
    print(f"  {'':<16} ボラ     {row(bv)}")


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=300, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    adv = turn.reindex(columns=superset)
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

    # --- レジーム（日次マーケット→拡張窓三分位→月末 as-of 整合・PIT）---
    daily = load_daily_panel(field="AdjC")
    trend_m = trend_regime(daily).reindex(rebal, method="ffill")    # 0=レンジ…2=強トレンド
    vol_m = vol_regime(daily).reindex(rebal, method="ffill")        # 0=低…2=高ボラ
    print(f"=== value/PEAD × レジーム（{START}〜{END}・月末{len(rebal)}・scope={SCOPE}）===")
    print(f"レジーム被覆: トレンド {trend_m.notna().mean():.0%} / ボラ {vol_m.notna().mean():.0%}"
          f"（月末に as-of 整合）")

    value_ls = CrossSectionalStrategy(value, 0.2, name="value")
    pead_lt = CrossSectionalStrategy(pead, 0.2, name="pead_longtilt", long_only=True)
    combo = CompositeStrategy([value_ls, pead_lt], [0.5, 0.5], name="value+pead_lt")
    # 事前固定ゲート：高ボラ/強トレンド（regime 2）を回避（allowed={0,1}）
    strategies = [
        value_ls, pead_lt, combo,
        RegimeGated(value_ls, vol_m, allowed={0, 1}, name="value|vol<=1"),
        RegimeGated(value_ls, trend_m, allowed={0, 1}, name="value|trend<=1"),
        RegimeGated(combo, vol_m, allowed={0, 1}, name="value+pead_lt|vol<=1"),
        RegimeGated(combo, trend_m, allowed={0, 1}, name="value+pead_lt|trend<=1"),
    ]

    reg_cm = TrialRegistry(REG_PATH) if REG_PATH else default_registry()
    with reg_cm as reg:
        v = judge_grid(
            strategies, view, scope=SCOPE,
            hypothesis=("value/PEAD の P&L が市場レジームで分離するなら、高ボラ/強トレンド局面を"
                        "避ける regime ゲート（規則メタラベル）で OOS安定/DSR が改善するか"),
            economic_rationale=("value は景気循環/ディストレス寄りで flight-to-quality(高ボラ)・"
                                "モメンタム相場(強トレンド)に劣後しやすい。該当局面を外せば耐久性が"
                                "増す。レジームは経済的に動機づけ・事前固定。"),
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print("\n--- regime_breakdown（baseline・年率Sharpe・分離の有無を建玉前に診断）---")
    for nm in ("value", "pead_longtilt", "value+pead_lt"):
        s = v.series.get(nm, pd.Series(dtype="float64"))
        if not s.empty:
            _brk(nm, s, trend_m, vol_m)

    print(f"\n--- IS/OOS（保留 {OOS}〜・年率Sharpe）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_ = ls[ls.index < pd.Timestamp(OOS)]
        oos = ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        (_, pre), (_, post) = pre_post_sharpe(ls, "2020-01-01")
        print(f"  {r.name:<26} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={si:+.2f} OOS={so:+.2f} | 前2020={pre:+.2f} 後={post:+.2f}")

    print("\n※ 判断：baseline で有利レジーム(r0/r1)>>不利(r2)＝分離があり、かつ gated の DSR/OOS が"
          " baseline を上回って初めてレジームに価値（§6.8 規律）。分離が無ければ正直に負を記録。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
