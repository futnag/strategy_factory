"""仮説検証：多ファクター breadth（value × momentum × quality × low-vol の合成）。

Grinold の能動運用の基本法則 IR ≈ IC × √breadth に基づく。相関の低い、経済的根拠のある
複数ファクターを合成して「独立な賭けの数（breadth）」を増やせば、単一ファクターより高い
情報比（≒リスク調整後リターン）が得られるはず——を検証ファクトリ（DSR/PSR＋永続レジストリ）で
厳密に判定する。短期データやティックではなく、日次/月次の経済的プレミアで幅を稼ぐ方針の実証。

全ユニバースの財務は fundamentals_panel（fins_summary/ 全件 by-date ミラー）で先読みなし組立。
新シグナル low_volatility（低ボラ）と accruals（利益の質）を含む。

注: fins_summary / daily_quotes の全件ダウンロード完了後に実行すること（download_jquants.py）。
実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_breadth_factors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe, universe_members,
)
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.fundamentals import fundamentals_panel  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, low_volatility, sector_neutralize, value_quality_size_factors,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
TOP_N = 500          # 全件ミラーで広いユニバースが可能に（breadth↑）
FIELDS = ["ShOutFY", "TrShFY", "Eq", "TA", "EqAR", "FEPS", "FNP", "FOP", "FSales",
          "FDivAnn", "CFO", "NP"]


def _nanmean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """欠損許容の平均（1ファクター欠けても残りで合成）。全NaNセルのみ NaN。"""
    a = np.array([f.values for f in frames], dtype=float)
    cnt = np.sum(~np.isnan(a), axis=0)
    s = np.nansum(a, axis=0)
    return pd.DataFrame(np.where(cnt > 0, s / np.where(cnt == 0, 1, cnt), np.nan),
                        index=frames[0].index, columns=frames[0].columns)


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=TOP_N, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    adv = turn.reindex(columns=superset)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})

    # 全ユニバース財務 as-of パネル（fins_summary/ 全件ミラー・先読みなし・銘柄別ネットワーク不要）
    pit = fundamentals_panel(rebal, FIELDS, codes=superset, lag_days=1)
    vqs = value_quality_size_factors(pit, raw, adj)

    def zN(f: pd.DataFrame) -> pd.DataFrame:
        return cross_sectional_zscore(sector_neutralize(
            apply_universe_mask(f.reindex(columns=superset), umask), sector))

    # 4つの古典的スタイル（相関が低い＝breadthを稼ぐ）
    value = zN(vqs["book_to_market"])
    momentum = zN(vqs["momentum"])
    quality = _nanmean([zN(vqs["roe"]), zN(vqs["accruals"])])     # 利益水準＋利益の質(新)
    low_vol = zN(low_volatility(adj, window=12))                  # 低ボラ(新)
    factors = {"value": value, "momentum": momentum, "quality": quality, "low_vol": low_vol}
    multi = _nanmean(list(factors.values()))                     # 多ファクター合成＝breadth

    strats = [CrossSectionalStrategy(s, 0.2, name=n) for n, s in factors.items()]
    strats.append(CrossSectionalStrategy(multi, 0.2, name="multifactor"))

    with default_registry() as reg:
        v = judge_grid(
            strats, view, scope="breadth_factors",
            hypothesis="相関の低い複数ファクターの合成は IR≈IC×√breadth により単一ファクターを上回る",
            economic_rationale="value/momentum/quality/low-vol は独立な経済的プレミアで、合成は"
                               "独立な賭けの数(breadth)を増やし情報比を高める（速度ではなく幅で勝つ）",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    singles = []
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_ = ls[ls.index < pd.Timestamp(OOS)]
        oos = ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {r.name:<12} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")
        if r.name != "multifactor":
            singles.append(r.sr_ann)

    mr = next((r for r in v.results if r.name == "multifactor"), None)
    if mr and singles:
        best = max(singles)
        print("\n--- 判定（breadth 効果）---")
        print(f"  multifactor 全SR={mr.sr_ann:+.2f} DSR={mr.dsr:.2f} / 単一最良SR={best:+.2f}")
        if mr.sr_ann >= best and mr.dsr >= max(r.dsr for r in v.results if r.name != "multifactor"):
            print("  ◎ 合成が単一最良を上回る＝幅(breadth)拡大の効果を確認（IR≈IC×√breadth）。")
        elif mr.sr_ann >= best:
            print("  ○ SRは単一最良以上だが DSR は単一に劣る（試行数Kの増加で閾値上昇＝健全な保守化）。")
        else:
            print("  ・合成は単一最良を超えず（個別が弱い/相関が高い）。さらなる独立シグナルが必要。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
