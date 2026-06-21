"""仮説検証：Value + Quality + Profitability ロングオンリー・ファクター（低回転）。

代替戦略ドキュメント 戦略1（Systematic Factor Investing）を裁く。ファクトリ既知＝value は唯一の
生存因子・quality 単独は負（§2.3/§6.4）だが、**V+Q+P ロングオンリー合成**としては未分離。
文書のスペック（月次低回転・流動性高い銘柄・ロングオンリー・NISA 相性）に合わせて構築する。

設計（PIT・生存者バイアス排除・既存因子規律）：
- ユニバース＝PIT 流動性上位500（文書「流動性高い銘柄に絞る」）。S33 セクター中立・XS z 化。
- 3本柱（クリーンに分離）：
  V(value)=B/M・予想E/P・CF利回り、Q(quality)=低アクルーアル・自己資本比率、
  P(profitability)=予想ROE・ROA・営業利益率。各柱は構成要素 z の nanmean、合成は3柱の nanmean。
- ロングオンリー＝上位分位ロA×ユニバース等加重ヘッジ（下位を名指しで売らない＝§7.2 と同じ）。
- 日本の「ROE向上」角度＝ΔROE(前年比改善)を足した変種も検証。
- 判定＝scope factor_vqp_longonly・月次・15bps・容量込み・デフレートDSR。

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_factor_vqp_longonly.py
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
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe, universe_members,
)
from invest_system.equities.fundamentals import fundamentals_panel  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import AsOfView, CrossSectionalStrategy, judge_grid, write_html  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

SCOPE = "factor_vqp_longonly"
OOS = get_env("J_VQP_OOS", "2024-01") or "2024-01"
TOP_N = int(get_env("J_VQP_TOP_N", "500") or "500")
FIELDS = ["ShOutFY", "TrShFY", "Eq", "TA", "EqAR", "FEPS", "FNP", "FOP", "FSales", "CFO", "NP"]


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ser = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(ser.groupby(idx.to_period("M")).max().values)


def _nanmean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    a = np.array([f.values for f in frames], dtype=float)
    cnt = np.sum(~np.isnan(a), axis=0)
    s = np.nansum(a, axis=0)
    return pd.DataFrame(np.where(cnt > 0, s / np.where(cnt == 0, 1, cnt), np.nan),
                        index=frames[0].index, columns=frames[0].columns)


def main() -> int:
    adj, raw, turn = load_wide("adj_close"), load_wide("close"), load_wide("turnover")
    if adj.empty or raw.empty:
        print("ERROR: Silver 未生成。store.materialize_all() を先に実行してください。")
        return 1
    me = _month_ends(adj.index)
    adj, raw, turn = adj.loc[me], raw.loc[me], turn.loc[me]
    print(f"=== V+Q+P ロングオンリー検証  月末{len(me)}  上位{TOP_N}  scope={SCOPE} ===")

    listed = jq.fetch_listed_info().assign(Code=lambda d: d["Code"].astype(str))
    common = set(filter_common_stocks(listed)["Code"])
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=TOP_N, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    adv = turn.reindex(columns=superset)
    sector = listed.set_index("Code")["S33"]
    view = AsOfView({"close": adj})
    print(f"ユニバース superset={len(superset)} 月平均所属={umask.sum(axis=1).mean():.0f}")

    pit = fundamentals_panel(me, FIELDS, codes=superset, lag_days=1)
    vqs = value_quality_size_factors(pit, raw, adj)

    def zN(f: pd.DataFrame) -> pd.DataFrame:
        return cross_sectional_zscore(sector_neutralize(
            apply_universe_mask(f.reindex(index=me, columns=superset), umask), sector))

    # 3本柱（クリーンに分離）
    V = _nanmean([zN(vqs["book_to_market"]), zN(vqs["earnings_yield"]), zN(vqs["cf_yield"])])
    Q = _nanmean([zN(vqs["accruals"]), zN(vqs["equity_ratio"])])
    P = _nanmean([zN(vqs["roe"]), zN(vqs["roa"]), zN(vqs["op_margin"])])
    VQP = _nanmean([V, Q, P])
    # ROE向上（前年比改善）＝profitability の動的版（日本のPBR改革角度）
    roe = vqs["roe"].reindex(index=me, columns=superset)
    d_roe = zN(roe - roe.shift(12))
    VQP_imp = _nanmean([V, Q, P, d_roe])

    strategies = [
        CrossSectionalStrategy(zN(vqs["book_to_market"]), 0.2, name="v_only_longonly", long_only=True),
        CrossSectionalStrategy(VQP, 0.2, name="vqp_longonly", long_only=True),
        CrossSectionalStrategy(VQP, 0.2, name="vqp_ls", long_only=False),
        CrossSectionalStrategy(VQP_imp, 0.2, name="vqp_improve_longonly", long_only=True),
    ]

    with default_registry() as reg:
        v = judge_grid(
            strategies, view, scope=SCOPE,
            hypothesis=("Value+Quality+Profitability のロングオンリー低回転合成は、独立な経済的"
                        "プレミアの幅(breadth)で単一 value を上回る耐久エッジになるか（文書 戦略1）"),
            economic_rationale=("value=割安リスクプレミア、quality=低アクルーアル/健全 BS、"
                                "profitability=高 ROE/ROA(RMW)。AVUV/DFSV 型の value×profitability "
                                "傾斜。ロングオンリーで取引コスト極小・NISA 整合。ΔROE は PBR 改革の"
                                "ROE向上角度"),
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    write_html(v, f"data/reports/{SCOPE}.html")

    print(f"\n--- IS/OOS（保留 {OOS}〜・年率Sharpe）・前後2020 ---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_ = ls[ls.index < pd.Timestamp(OOS)]
        oos = ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        (_, pre), (_, post) = pre_post_sharpe(ls, "2020-01-01")
        print(f"  {r.name:<22} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} maxDD={r.max_dd:+.1%} "
              f"回転={r.turnover:.2f} | IS={si:+.2f} OOS={so:+.2f} | 前2020={pre:+.2f} 後={post:+.2f}")
    print("\n  ※ 文書通りなら V+Q+P 合成が v_only を上回るはず。年5-12%の控えめリターン想定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
