"""執行現実性チェック：最有力候補（value↔PEAD 切替）は日本固有の執行制約に頑健か。

フレームワーク評価で特定したギャップ 1-3 への回答（docs/03 §6.12）：
  1. 値幅制限 … ストップ高/安の**引け張り付き**日は執行不能（前回ウェイトをキャリー）
  2. ショート貸株コスト … 制度貸株料(約115bps/年)＋逆日歩バッファを短グロスに賦課。
     併せて value ショート脚の**貸借銘柄カバレッジ**（制度で売れるか）を診断
  3. ケリー基準 … 現実性込みネットからフラクショナル・ケリーで Phase 2 レバレッジを導出

規律（§6.6 の throwaway 測定と同じ）：戦略・パラメータは §6.9-6.11 で確定済みのものを
**一切変更せず**、エンジンの現実性のみを切り替えて同一戦略を再評価する＝新たな選択は
発生しない（K 不変・永続レジストリ不使用）。最悪シナリオでも結論が崩れないかを見る。

実行: .venv\\Scripts\\python.exe examples\\research_value_pead_realism.py
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
from invest_system.equities import events, margin  # noqa: E402
from invest_system.equities.frictions import (  # noqa: E402
    limit_lock_flags, short_notional_coverage, shortable_mask,
)
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
    walk_forward_regime_assignment,
)
from invest_system.timeseries import vol_regime  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
WF_WARMUP, WF_MINOBS = 24, 6
BORROW_GRID = [115.0, 200.0, 400.0]   # 制度貸株料 / +逆日歩バッファ / ストレス（bps/年）


class _Replay(Strategy):
    """事前計算した月次ウェイト（PIT生成済み）を date 引きで返す。"""

    def __init__(self, weights: dict, name: str):
        self._w = weights
        self.name = name
        self.params = {}

    def target_weights(self, asof):
        return self._w.get(asof.asof, pd.Series(dtype="float64"))


def _sr(x: pd.Series, oos: bool = False) -> float:
    r = x.dropna()
    if oos:
        r = r[r.index >= pd.Timestamp(OOS)]
    return float(sharpe_ratio(r) * np.sqrt(12)) if r.size >= 8 else float("nan")


def _maxdd(r: pd.Series) -> float:
    cum = (1.0 + r.dropna()).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1

    # --- データ組立（research_value_pead_regime.py と同一・戦略は不変）---
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
    daily = load_daily_panel(field="AdjC")
    vol_m = vol_regime(daily).reindex(rebal, method="ffill")

    value_ls = CrossSectionalStrategy(value, 0.2, name="value")
    pead_lt = CrossSectionalStrategy(pead, 0.2, name="pead_longtilt", long_only=True)
    switch = RegimeSwitch(vol_m, {0: value_ls, 1: value_ls, 2: pead_lt},
                          name="switch")

    # --- 執行フリクションの構築 ---
    hi, lo = (assemble_panel(snaps, c).reindex(columns=superset) for c in ("H", "L"))
    ul, ll = (assemble_panel(snaps, c).reindex(columns=superset) for c in ("UL", "LL"))
    vo = assemble_panel(snaps, "Vo").reindex(columns=superset)
    no_buy, no_sell = limit_lock_flags(raw, hi, lo, ul, ll, volume=vo)
    print(f"=== 執行現実性チェック（{START}〜{END}・月末{len(rebal)}回・戦略は §6.9-6.11 と同一）===")
    print(f"値幅制限フラグ（月末・ユニバース superset {len(superset)}銘柄）: "
          f"買い不能 {int(no_buy.to_numpy().sum())}セル / 売り不能 {int(no_sell.to_numpy().sum())}セル"
          f"（全 {no_buy.size} 銘柄・月）")

    # --- 貸借銘柄カバレッジ（value ショート脚の診断・戦略は変更しない）---
    weekly = margin.load_weekly_margin()
    shortable = shortable_mask(weekly, rebal)
    Wv = {t: value_ls.target_weights(view.asof(t)) for t in rebal}
    Wp = {t: pead_lt.target_weights(view.asof(t)) for t in rebal}
    cov = pd.Series({t: short_notional_coverage(Wv[t], shortable.loc[t])
                     for t in rebal if len(Wv[t])}).dropna()
    print(f"\n--- value ショート脚の貸借カバレッジ（制度信用で売建可能な想定元本比率）---")
    print(f"  平均 {cov.mean():.1%} / 最低 {cov.min():.1%}（{cov.idxmin():%Y-%m}） / "
          f"中央値 {cov.median():.1%}  ※1.0未満分は一般信用・現物ヘッジ等の追加コスト要因")

    # --- シナリオ比較（同一戦略・エンジンの現実性のみ切替）---
    scenarios = [("A: 従来仮定(15bps)", {}),
                 ("B: +値幅制限", dict(no_buy=no_buy, no_sell=no_sell))]
    for tag, b in zip("CDE", BORROW_GRID):
        scenarios.append((f"{tag}: +値幅制限+貸株{b:.0f}bp",
                          dict(no_buy=no_buy, no_sell=no_sell, short_borrow_bps=b)))

    strategies = [value_ls, pead_lt, switch]
    nets: dict[str, dict[str, pd.Series]] = {}
    for label, kw in scenarios:
        nets[label] = {}
        print(f"\n=== {label} ===")
        print(f"  {'strategy':<16} {'SR(全)':>7} {'SR(OOS)':>8} {'maxDD':>7} "
              f"{'前/後2020':>13} {'短グロス':>8} {'block':>6}")
        for s in strategies:
            res = backtest(s, view, costs_bps=15.0, **kw)
            r = res.returns.dropna()
            nets[label][s.name] = r
            (_, pre), (_, post) = pre_post_sharpe(r, "2020-01-01")
            sg = float(res.short_gross.reindex(r.index).mean())
            blk = int(res.n_blocked.reindex(r.index).sum())
            print(f"  {s.name:<16} {_sr(r):>+7.2f} {_sr(r, oos=True):>+8.2f} "
                  f"{_maxdd(r):>7.1%} {pre:>+6.2f}/{post:>+5.2f} {sg:>8.2f} {blk:>6}")
        # walk-forward 適応切替（割当はこのシナリオのネットから過去のみで学習）
        Rv, Rp = nets[label]["value"], nets[label]["pead_longtilt"]
        assign = walk_forward_regime_assignment({"value": Rv, "pead_longtilt": Rp},
                                                vol_m, min_obs=WF_MINOBS,
                                                warmup=WF_WARMUP)
        Wmap = {"value": Wv, "pead_longtilt": Wp}
        AW = {t: (Wmap[assign.get(t)][t] if isinstance(assign.get(t), str)
                  else pd.Series(dtype="float64")) for t in rebal}
        wf = _Replay(AW, name="wf_switch")
        res = backtest(wf, view, costs_bps=15.0, rebalance=rebal[WF_WARMUP:], **kw)
        r = res.returns.dropna()
        nets[label][wf.name] = r
        (_, pre), (_, post) = pre_post_sharpe(r, "2020-01-01")
        sg = float(res.short_gross.reindex(r.index).mean())
        blk = int(res.n_blocked.reindex(r.index).sum())
        print(f"  {wf.name:<16} {_sr(r):>+7.2f} {_sr(r, oos=True):>+8.2f} "
              f"{_maxdd(r):>7.1%} {pre:>+6.2f}/{post:>+5.2f} {sg:>8.2f} {blk:>6}")

    # --- ケリー基準（ギャップ3）：保守シナリオのネットから Phase 2 レバレッジを導出 ---
    conservative = scenarios[2][0]            # C: +値幅制限+貸株115bp（制度貸株料の実勢）
    print(f"\n=== フラクショナル・ケリー（シナリオ『{conservative}』のネットから推定・DP15）===")
    for nm in ("switch", "wf_switch"):
        r = nets[conservative][nm]
        for frac in (0.5, 0.25):
            k = kelly_fraction(r, fraction=frac)
            print(f"  {nm:<10} {frac:>4.2f}×Kelly: {k.summary()}")
    print("  ※ f はこの戦略リターン系列に対するレバレッジ倍率（1.0=等倍）。推定誤差・"
          "ギャップリスクのため満額は使わない（DP15）。実運用は月次で再推定し、"
          "ハード限度（最大DD・グロス上限）と併用する。")

    print("\n※ 判定の読み方：A→B の差＝値幅制限の執行不能の影響（月次・大型流動性では"
          "小さいはず）。B→C/D/E の差＝ショート貸株コストの感応度（短グロス≈1 なら年率で"
          "ほぼ borrow_bps ぶん SR の分子が削れる）。最悪セルでも OOS が正なら"
          "§6.10-6.11 の結論は執行現実性に対して頑健。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
