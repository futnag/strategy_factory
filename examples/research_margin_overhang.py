r"""仮説検証 ⑤：信用買残オーバーハング（"買い残の重し"・供給側の需給）。

信用買残（LongVol・週次・全銘柄）を**浮動株で正規化**した overhang = 信用買残/(発行済−自己株) が
厚い銘柄は、いずれ反対売買で売られる**供給オーバーハング**を抱え将来アンダーパフォームする——を
判定器で裁く。テスト済みの short_to_long（信用売残/買残＝買/売"比率"）とは**別の正規化**（残高の
"大きさ"を浮動株比で測る）＝レジストリ未登録の新規 scope。

シグナル: overhang = LongVol_asof / float_asof（PIT・T+2公表ラグ）。高い＝重し＝低品質。
**低 overhang をロング/高 overhang をショート**（= −overhang）。分位 q∈{0.1,0.2,0.3}（K=3）。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\Scripts\python.exe examples\research_margin_overhang.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import margin as mg  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe, universe_members,
)
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.fundamentals import fundamentals_panel, point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
    winsorize_cross_sectional,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
TOP_N = 500
QS = [0.1, 0.2, 0.3]
FIELDS = ["ShOutFY", "TrShFY", "Eq", "FEPS", "FNP", "FSales", "FDivAnn", "CFO", "NP", "TA"]


def _avg_xs_corr(a: pd.DataFrame, b: pd.DataFrame, min_names: int = 20) -> float:
    vals = []
    for t in a.index:
        x, y = a.loc[t], b.reindex(columns=a.columns).loc[t]
        m = x.notna() & y.notna()
        if int(m.sum()) >= min_names:
            xx, yy = x[m].to_numpy(float), y[m].to_numpy(float)
            if xx.std() > 0 and yy.std() > 0:
                vals.append(float(np.corrcoef(xx, yy)[0, 1]))
    return float(np.nanmean(vals)) if vals else float("nan")


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1

    print(f"=== ⑤ 信用買残オーバーハング {START}〜{END} 上位{TOP_N} ===")
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
    print(f"パネル {adj.shape[0]}ヶ月 × ユニバース{len(superset)}銘柄")

    # 週次信用残を as-of（T+2公表＝lag 4日）。LongVol=信用買残, ShrtVol=信用売残
    weekly = mg.load_weekly_margin()
    weekly["Code"] = weekly["Code"].astype(str)
    pit_m = point_in_time(weekly, rebal, ["LongVol", "ShrtVol"],
                          date_col="Date", code_col="Code", lag_days=4)
    longv = pit_m["LongVol"].reindex(columns=superset)
    shrtv = pit_m["ShrtVol"].reindex(columns=superset)

    # 浮動株（発行済 − 自己株）as-of
    pit_f = fundamentals_panel(rebal, FIELDS, codes=superset, lag_days=1)
    flt = (pit_f["ShOutFY"].reindex(columns=superset)
           .sub(pit_f["TrShFY"].reindex(columns=superset), fill_value=0.0))
    flt = flt.where(flt > 0)

    overhang = (longv / flt).reindex(columns=superset)        # 信用買残/浮動株
    cov = int((~overhang.isna()).sum().sum())
    print(f"overhang 有効セル {cov:,}（信用買残/浮動株）")

    def prep(f: pd.DataFrame) -> pd.DataFrame:
        fm = apply_universe_mask(f.reindex(columns=superset), umask)
        return cross_sectional_zscore(sector_neutralize(
            winsorize_cross_sectional(fm), sector))

    signal = prep(-overhang)               # 低 overhang=ロング / 高=ショート（重し仮説）
    strats = [CrossSectionalStrategy(signal, q, name=f"margin_overhang(q={q})") for q in QS]

    with default_registry() as reg:
        v = judge_grid(
            strats, view, scope="margin_overhang",
            hypothesis="信用買残が浮動株比で厚い銘柄は将来の反対売買（供給オーバーハング）で"
                       "アンダーパフォームする（買い残の重し）",
            economic_rationale="信用買いは期限付きの一時的需要で、いずれ反対売買の供給に転じる。"
                               "浮動株比で厚いほど将来の売り圧力＝価格の重し。レバレッジ清算/追証も"
                               "価格非感応の強制売り。反対側は残高を見ない参加者。short_to_long(比率)とは別の正規化。",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    # 独立性：テスト済 short_to_long（売残/買残）・momentum と相関
    s2l = prep((shrtv / longv.replace(0, np.nan)))
    vqs = value_quality_size_factors(pit_f, raw, adj)
    mom = prep(vqs["momentum"])
    print("\n--- 独立性（overhang シグナルと既知factorの月平均XS相関）---")
    for nm, other in (("short_to_long", s2l), ("momentum", mom)):
        print(f"  vs {nm:<14} ρ̄ = {_avg_xs_corr(signal, other):+.2f}")
    print("  |ρ̄| が小さいほど独立（既試 short_to_long の焼き直しでない）。")

    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_, oos = ls[ls.index < pd.Timestamp(OOS)], ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {r.name:<22} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | IS={si:+.2f} OOS={so:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
