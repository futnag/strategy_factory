"""仮説検証：Gold 層の価格ファクター（momentum / low-vol / reversal）＋ regime gate。

Silver(adj_close/turnover) と Gold(feature_store) だけで月次クロスセクション研究を組み、判定
ファネル（コスト・執行・容量・デフレートDSR）で正直に裁く。API 不要・全て PIT。

- 月末リバランス・流動性上位ユニバース・S33 セクター中立・z 化（既存 factors と同じ規律）。
- ファクター：momentum_12_1 / low_vol(=−vol_20) / reversal_5。
- **regime gate**：高ボラ局面(vol_regime==2)では momentum を建てない（市場レジームをメタ条件に）。

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_price_factors.py
（事前に store.materialize_all() と feature_store.materialize_features() を実行しておくこと）
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
from invest_system.data.store import load_wide  # noqa: E402
from invest_system.data.feature_store import load_feature  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize,
)
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import TrialRegistry, default_registry  # noqa: E402

TOP_N = int(get_env("J_PF_TOP_N", "500") or "500")
OOS = get_env("J_PF_OOS", "2024-01") or "2024-01"
REG_PATH = get_env("J_PF_REGISTRY", None)


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ser = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(ser.groupby(idx.to_period("M")).max().values)


def main() -> int:
    adj = load_wide("adj_close")
    turn = load_wide("turnover")
    if adj.empty or turn.empty:
        print("ERROR: Silver 未生成。store.materialize_all() を先に実行してください。")
        return 1
    me = _month_ends(adj.index)
    adj_me, turn_me = adj.loc[me], turn.loc[me]
    print(f"=== Gold 価格ファクター検証  月末{len(me)}  上位{TOP_N} ===")

    listed = jq.fetch_listed_info()
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn_me[[c for c in turn_me.columns if c in common]]
    mask = point_in_time_universe(turn_c, top_n=TOP_N, lookback=12, min_obs=6)
    superset = universe_members(mask)
    adj_me = adj_me.reindex(columns=superset)
    mask = mask.reindex(columns=superset).fillna(False)
    adv = turn_me.reindex(columns=superset)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    view = AsOfView({"close": adj_me})
    print(f"ユニバース superset={len(superset)}  月平均所属={mask.sum(axis=1).mean():.0f}")

    def zN(daily_feat: pd.DataFrame) -> pd.DataFrame:
        f = daily_feat.reindex(index=me, columns=superset)
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, mask),
                                                         sector))
    momentum = zN(load_feature("momentum_12_1"))
    low_vol = zN(-load_feature("vol_20"))
    reversal = zN(load_feature("reversal_5"))

    # regime gate：高ボラ(vol_regime==2)の月は momentum を建てない（行を NaN 化＝flat）
    reg = load_feature("regime").reindex(me)
    favorable = reg["vol_regime"] < 2
    fav_mult = favorable.astype(float).where(favorable, np.nan)
    momentum_gated = momentum.mul(fav_mult, axis=0)

    strategies = [
        CrossSectionalStrategy(momentum, 0.2, name="momentum_12_1"),
        CrossSectionalStrategy(low_vol, 0.2, name="low_vol"),
        CrossSectionalStrategy(reversal, 0.2, name="reversal_5"),
        CrossSectionalStrategy(momentum_gated, 0.2, name="momentum_regime_gated"),
    ]
    reg_cm = TrialRegistry(REG_PATH) if REG_PATH else default_registry()
    with reg_cm as reg_db:
        v = judge_grid(
            strategies, view, scope="price_factors_gold",
            hypothesis=("Gold層の価格ファクター(momentum/low-vol/reversal)と高ボラ回避の"
                        "regime gate がクロスセクションで有効か"),
            economic_rationale=("momentum=継続性、low-vol=低ボラアノマリー、reversal=短期反転、"
                                "regime=高ボラ局面のドローダウン回避。いずれも経済的に動機づけ"),
            registry=reg_db, costs_bps=15.0, adv=adv)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜・年率Sharpe）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_ = ls[ls.index < pd.Timestamp(OOS)]
        oos = ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        (_, pre), (_, post) = pre_post_sharpe(ls, "2020-01-01")
        print(f"  {r.name:<22} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={si:+.2f} OOS={so:+.2f} | 前2020={pre:+.2f} 後={post:+.2f}")
    print("\n  ※ regime gate が momentum の OOS/最大DD を改善するか（高ボラ回避の価値）を確認。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
