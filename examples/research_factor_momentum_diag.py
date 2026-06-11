"""P3-D: ファクター・モメンタムの分離診断 — docs/04 P3-D・G5(SSRN 4376898)/J3(AQR)。

G5（Neuhierl et al. 2023）は「ファクター自身の過去リターンとボラが最良の個別予測子」、
Gupta-Kelly "Factor Momentum Everywhere" も同趣旨を独立に示す（Asness の限定＝有効なのは
モメンタム軸のタイミングのみ、とも整合）。本リポジトリは vol レジーム条件付け（§6.9）を
採用済みだが「スリーブ自身の過去リターン」という第2のゲート軸は未診断だった。

規律（§6.8-6.9 で確立・建玉前診断）：
- value / pead_longtilt（§6.10 シナリオ A 構成）・tsmom_blend（§6.15 事前登録構成）の
  月次ネットについて、「自身の過去12ヶ月リターン（符号 / 拡張窓三分位）別の翌月 Sharpe
  分離」を **1回だけ** 計測する。建玉なし・戦略変更なし・K 消費ゼロ。
- ラベルは PIT：decision 月 t のラベルは r[t-12..t-1]（t 時点で全て実現済み）のみ使用。
- 分離が出た場合のゲート化は**別途事前登録（その時点で K 計上）のユーザー承認事項**。

実行: .venv\\Scripts\\python.exe examples\\research_factor_momentum_diag.py
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
from invest_system.data.external import load_external_prices  # noqa: E402
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
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, Strategy, backtest,
)
from invest_system.research.judge import regime_breakdown  # noqa: E402
from invest_system.research.strategies_tsmom import (  # noqa: E402
    annualized_vol, blend_weights, tsmom_weights,
)
from invest_system.timeseries import expanding_tertile  # noqa: E402

START, END = "2016-07", "2026-05"
TSMOM_KEYS = ["nk225_fut", "sp500", "nasdaq_comp", "gold", "silver", "platinum",
              "wti", "copper", "usdjpy", "eurjpy", "audjpy"]
LOOKBACK_M = 12          # 自身の過去リターンの窓（事前固定・探索しない）
TERTILE_MIN = 24         # 拡張窓三分位の最小観測（月次系列向け・事前固定）


class _Replay(Strategy):
    def __init__(self, weights: dict, name: str):
        self._w = weights
        self.name = name
        self.params = {}

    def target_weights(self, asof):
        return self._w.get(asof.asof, pd.Series(dtype="float64"))


def _sleeve_nets_equity() -> dict[str, pd.Series]:
    """value / pead_longtilt の月次ネット（§6.10 シナリオ A 構成・15bps）。"""
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
    out = {}
    for s in (CrossSectionalStrategy(value, 0.2, name="value"),
              CrossSectionalStrategy(pead, 0.2, name="pead_longtilt",
                                     long_only=True)):
        out[s.name] = backtest(s, view, costs_bps=15.0).returns.dropna()
    return out


def _tsmom_blend_net() -> pd.Series:
    """§6.15 事前登録構成の tsmom_blend 月次ネット（5bps・T+1始値）。"""
    cl = load_external_prices(TSMOM_KEYS, field="close")
    op = load_external_prices(TSMOM_KEYS, field="open")
    cl_ff = cl.ffill(limit=7)
    m_close = cl_ff.groupby(cl_ff.index.to_period("M")).tail(1)
    rebal = m_close.index
    vol_m = annualized_vol(cl, window=63, floor=0.05).ffill(limit=7).reindex(rebal)
    fill_px = op.bfill(limit=3).shift(-1).reindex(rebal)
    view = AsOfView({"close": fill_px})
    sets = [tsmom_weights(m_close, vol_m, lb, vol_target=0.10) for lb in (3, 6, 12)]
    strat = _Replay(blend_weights(sets), "tsmom_blend")
    return backtest(strat, view, costs_bps=5.0).returns.dropna()


def _diagnose(name: str, r: pd.Series) -> None:
    """自身の過去 LOOKBACK_M ヶ月リターン別の翌月 Sharpe 分離（PIT・1回だけ）。"""
    trail = r.rolling(LOOKBACK_M).sum().shift(1)   # r[t-12..t-1]＝t 時点で実現済み
    sign_lab = np.sign(trail).replace(0.0, np.nan)
    tert = expanding_tertile(trail, min_periods=TERTILE_MIN)
    print(f"\n--- {name}（n={len(r)}・過去{LOOKBACK_M}ヶ月自身リターンでラベル）---")
    print("  [符号] -1=直近1年が負 / +1=正")
    bd = regime_breakdown(r, sign_lab, ann=12.0)
    for _, row in bd.iterrows():
        print(f"    sign={row['regime']:+.0f}: n={int(row['n']):>3} "
              f"mean={row['mean']:+.4f} SR(ann)={row['sharpe_ann']:+.2f}")
    print(f"  [拡張窓三分位] 0=低 / 1=中 / 2=高（min_periods={TERTILE_MIN}）")
    bd = regime_breakdown(r, tert, ann=12.0)
    for _, row in bd.iterrows():
        print(f"    tert={row['regime']:.0f}:  n={int(row['n']):>3} "
              f"mean={row['mean']:+.4f} SR(ann)={row['sharpe_ann']:+.2f}")


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    print("=== P3-D ファクター・モメンタム分離診断（建玉なし・K 消費ゼロ・1回だけ）===")
    nets = _sleeve_nets_equity()
    nets["tsmom_blend"] = _tsmom_blend_net()
    for nm, r in nets.items():
        _diagnose(nm, r)
    print("\n※ 読み方：G5/J3 の仮説どおりなら「自身の過去12ヶ月が正（高）」の状態で翌月"
          " Sharpe が高いはず。分離が出てもここでは**ゲート化しない**（事前登録＋K 計上の"
          "ユーザー承認事項）。分離が無ければ『ファクター・モメンタム軸は本スリーブには"
          "効かない』と記録して終了（§6.18-6.19 と同じ早期判定）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
