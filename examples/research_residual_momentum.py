"""P3-H: 残差モメンタム — 日本で機能するモメンタム仮説の検証【新 scope・K=4】。

事前登録（docs/04 P3-H への記載をもって登録・本スクリプト実行で scope を永続レジストリへ）：
- 仮説: 過去36ヶ月（t-36〜t-1）の対市場単回帰残差から計算した 12-1 残差モメンタム
  （残差の12ヶ月累積・直近1ヶ月スキップ・σ標準化あり/なし）の上位ティルト/L-S は、
  生リターン版 momentum_12_1（§6.7 で負）を OOS で上回り、DSR 基準を満たす。
- 経済的根拠: 日本のモメンタム失敗は市場ベータ起因の系統成分（リバーサルしやすい）が
  原因で、銘柄固有のアンダーリアクション成分は日本でも存在する（Chaves 2016／
  Chang et al. 2018 日本専門・アンダーリアクション説／Blitz et al. 2011・2020）。
- グリッド（K=4 上限・探索しない）: {ロングティルト, LS} × {σ標準化あり, なし}。
  回帰窓 36・12-1・市場＝ユニバース等加重は**事前固定**（変えれば K 計上の対象）。
- ベースライン（比較表示のみ・レジストリ不使用＝K 不変）: 生 momentum_12_1 と value 単体。
- 判定: judge_grid(scope="residual_momentum")。コスト15bps・容量・サブ期間・OOS(2024+)
  併記。T+1 始値（DP17）は throwaway 併記。
- 撤退基準: 全構成 FAIL なら変種調律をせず打ち切り（§6.6 の規律）。
- 前提充足: P1-B 上流汚染監査（§6.21）済み＝月次・流動性上位300 では張り付き価格の
  影響は無視できる規模（ユニバース内 0.068%・分位入替 ≲0.4%）。

実行: .venv\\Scripts\\python.exe examples\\research_residual_momentum.py
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
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.panel import (  # noqa: E402
    assemble_panel, fetch_month_end_snapshots, trailing_momentum,
)
from invest_system.equities.fundamentals import load_fundamentals, point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, residual_momentum, sector_neutralize,
    value_quality_size_factors,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, backtest, open_fill_backtest,
)
from invest_system.research.judge import judge_grid  # noqa: E402
from invest_system.research.report_html import write_html  # noqa: E402
from invest_system.equities.panel import load_daily_panel  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
SCOPE = "residual_momentum"
HYPOTHESIS = ("過去36ヶ月の対市場単回帰残差による12-1残差モメンタム（σ標準化/生・"
              "LT/LS）は、生momentum_12_1（§6.7で負）をOOSで上回りDSR基準を満たす")
RATIONALE = ("日本のモメンタム失敗は市場ベータ起因の系統成分（リバーサルしやすい）が"
             "原因で、銘柄固有のアンダーリアクション成分は日本でも存在する"
             "（Chaves 2016／Chang et al. 2018／Blitz et al. 2011・2020 の独立証拠）")


def _sr(x: pd.Series, oos: bool = False) -> float:
    r = x.dropna()
    if oos:
        r = r[r.index >= pd.Timestamp(OOS)]
    return float(sharpe_ratio(r) * np.sqrt(12)) if r.size >= 8 else float("nan")


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1

    # --- データ組立（旗艦と同一の月次 PIT パネル）---
    print("データ組立中（旗艦と同一の月次 PIT パネル）...")
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
    adv = turn.reindex(columns=superset)

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, umask),
                                                        sector))

    # --- 残差モメンタム（事前固定: 窓36・12-1・市場＝ユニバース等加重）---
    ret_m = adj.pct_change()
    mkt = ret_m.mean(axis=1)                        # 等加重市場（superset）・事前固定
    f_std = zN(residual_momentum(ret_m, mkt, standardize=True))
    f_raw = zN(residual_momentum(ret_m, mkt, standardize=False))
    n_valid = int(f_std.iloc[-1].notna().sum())
    first = f_std.dropna(how="all").index.min()
    print(f"  残差モメンタム: 初回シグナル {first:%Y-%m}（36ヶ月窓）・"
          f"最終断面の有効銘柄 {n_valid}")

    strategies = [
        CrossSectionalStrategy(f_std, 0.2, name="resmom_std_lt", long_only=True),
        CrossSectionalStrategy(f_std, 0.2, name="resmom_std_ls"),
        CrossSectionalStrategy(f_raw, 0.2, name="resmom_raw_lt", long_only=True),
        CrossSectionalStrategy(f_raw, 0.2, name="resmom_raw_ls"),
    ]

    # --- 判定（永続レジストリ・scope K=4）---
    with default_registry() as reg:
        v = judge_grid(strategies, view, scope=SCOPE, hypothesis=HYPOTHESIS,
                       economic_rationale=RATIONALE, registry=reg,
                       costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    out_html = write_html(v, "data/reports/residual_momentum.html")
    print(f"HTML: {out_html}")

    # --- OOS（2024+）と T+1 始値（DP17・throwaway 併記）---
    open_d = load_daily_panel(field="AdjO").reindex(
        columns=[c for c in superset])
    print("\n--- OOS（2024-01+）と T+1 始値リプレイ（throwaway・K 不変）---")
    print(f"  {'strategy':<16} {'SR(全)':>7} {'SR(OOS)':>8} {'T+1 SR(全)':>11} "
          f"{'T+1 SR(OOS)':>12}")
    for s in strategies:
        r = v.series.get(s.name, pd.Series(dtype="float64"))
        W = {t: s.target_weights(view.asof(t)) for t in rebal}
        W = {t: w for t, w in W.items() if len(w)}
        ro = open_fill_backtest(W, open_d, costs_bps=15.0, name=s.name).returns
        print(f"  {s.name:<16} {_sr(r):>+7.2f} {_sr(r, oos=True):>+8.2f} "
              f"{_sr(ro):>+11.2f} {_sr(ro, oos=True):>+12.2f}")

    # --- ベースライン（比較表示のみ・レジストリ不使用＝K 不変）---
    print("\n--- ベースライン（registry 不使用・§6.7 / §7.1 の照合）---")
    mom = zN(trailing_momentum(adj, lookback=12, skip=1))
    fund = load_fundamentals(superset)
    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    value = zN(value_quality_size_factors(pit, raw, adj)["book_to_market"])
    for nm, fac, lo in (("momentum_12_1_ls(生)", mom, False),
                        ("momentum_12_1_lt(生)", mom, True),
                        ("value_ls", value, False)):
        b = CrossSectionalStrategy(fac, 0.2, name=nm, long_only=lo)
        r = backtest(b, view, costs_bps=15.0).returns.dropna()
        print(f"  {nm:<22} SR(全) {_sr(r):+.2f} / SR(OOS) {_sr(r, oos=True):+.2f}")

    print("\n※ 撤退基準（事前固定）：全構成 FAIL なら変種調律（窓・スキップ・市場モデル"
          "の変更等）をせず打ち切り＝『残差化でも日本では無効』を §6.26 に記録する。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
