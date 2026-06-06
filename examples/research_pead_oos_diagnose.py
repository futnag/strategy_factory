"""PEAD（予想改訂）のOOS失敗(IS+0.38→OOS-0.18)の解明。

診断：
 ① IC（情報係数＝月次の factor vs 翌月リターンの順位相関）の IS/OOS と推移 …予測力の減衰
 ② ロング脚（上方修正）/ショート脚（下方修正）の超過リターンを分解 …どちらが崩れたか
 ③ value との相関の IS/OOS 変化 …合成を引っ張った要因か

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_pead_oos_diagnose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import events  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.panel import (  # noqa: E402
    assemble_panel, fetch_month_end_snapshots, forward_returns,
)
from invest_system.equities.fundamentals import point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)

START, END, OOS = "2016-07", "2026-05", "2024-01"


def fetch_fundamentals(codes):
    fr = []
    for c in codes:
        try:
            st = jq.fetch_statements(code=c)
            if not st.empty:
                fr.append(st)
        except Exception:  # noqa: BLE001
            pass
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def _split(s):
    s = s.dropna()
    return s[s.index < pd.Timestamp(OOS)], s[s.index >= pd.Timestamp(OOS)]


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
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    fwd = forward_returns(adj)
    fund = fetch_fundamentals(superset)

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, umask),
                                                        sector))
    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    value = zN(value_quality_size_factors(pit, raw, adj)["book_to_market"])
    pead = zN(point_in_time(events.forecast_revision(fund), rebal, ["fcst_revision"],
                            date_col="DiscDate", lag_days=1)["fcst_revision"]
              .reindex(columns=superset))

    # ① IC（順位相関）
    ic = pd.Series({t: pead.loc[t].corr(fwd.loc[t], method="spearman") for t in rebal}
                   ).dropna()
    ic_is, ic_oos = _split(ic)
    # ② 脚分解（q=0.2, 超過=脚平均−ユニバース平均）
    longs, shorts = {}, {}
    for t in rebal:
        f = pead.loc[t].dropna()
        r = fwd.loc[t]
        v = f.index.intersection(r.dropna().index)
        f, r = f[v], r[v]
        if len(f) < 20:
            continue
        k = max(1, int(len(f) * 0.2))
        o = f.sort_values()
        mkt = r.mean()
        longs[t] = r[o.index[-k:]].mean() - mkt        # 上方修正の超過
        shorts[t] = r[o.index[:k]].mean() - mkt        # 下方修正の超過
    longs, shorts = pd.Series(longs), pd.Series(shorts)
    l_is, l_oos = _split(longs)
    s_is, s_oos = _split(shorts)
    # ③ value との相関
    corr = pd.Series({t: pead.loc[t].corr(value.loc[t]) for t in rebal}).dropna()
    c_is, c_oos = _split(corr)

    print(f"=== PEAD OOS診断（IS:{START}〜{OOS}前 / OOS:{OOS}〜{END}）===")
    print("\n① 情報係数 IC（月次・順位相関, 予測力）")
    print(f"   IS  平均IC={ic_is.mean():+.3f}（正勝率{(ic_is>0).mean():.0%}, n={ic_is.size}）")
    print(f"   OOS 平均IC={ic_oos.mean():+.3f}（正勝率{(ic_oos>0).mean():.0%}, n={ic_oos.size}）")
    print("\n② 脚分解（月次・対ユニバース超過リターン, 年率換算）")
    print(f"   上方修正ロング脚  IS={l_is.mean()*12:+.2%} / OOS={l_oos.mean()*12:+.2%}")
    print(f"   下方修正ショート脚 IS={s_is.mean()*12:+.2%} / OOS={s_oos.mean()*12:+.2%}")
    print(f"   （LS≒ロング超過−ショート超過。ショート脚の超過が正に転じると損失）")
    print("\n③ value との平均クロスセクション相関")
    print(f"   IS={c_is.mean():+.2f} / OOS={c_oos.mean():+.2f}")

    print("\n--- 診断 ---")
    if ic_oos.mean() < 0.5 * ic_is.mean():
        print(f"  ・予測力(IC)が IS{ic_is.mean():+.3f}→OOS{ic_oos.mean():+.3f} と減衰/反転"
              "＝シグナル自体の劣化（混雑・レジーム）。")
    if s_oos.mean() > s_is.mean() + 0.002:
        print("  ・下方修正(ショート脚)が OOS で相対上昇＝ショートが効かず損失を主導。")
    if l_oos.mean() < l_is.mean() - 0.002:
        print("  ・上方修正(ロング脚)の超過も OOS で縮小。")
    if c_oos.mean() < c_is.mean() - 0.05:
        print("  ・valueとの相関がOOSで低下/負方向＝局面で対立し合成を相殺。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
