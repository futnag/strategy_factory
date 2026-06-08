"""生存者バイアス除去：ポイントインタイム・ユニバースで主要ファクターを再検証。

従来の固定ユニバース（全期間流動性で上位300を選抜）は先読み・生存者バイアスを持つ。
ここでは各時点の trailing 流動性で上位を選ぶ時変ユニバース（point_in_time_universe）に
切り替え、size/value/momentum/short_to_long の結論（size減衰・value復活・short_to_long）
がバイアス除去後も生き残るかを、同じ規律（中立化・両側デフレートDSR・サブ期間）で見る。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\jp_pit_universe_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import margin  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.panel import (  # noqa: E402
    assemble_panel, fetch_month_end_snapshots, forward_returns,
)
from invest_system.equities.fundamentals import load_fundamentals, point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.equities.backtest import long_short_returns  # noqa: E402
from invest_system.equities.stability import (  # noqa: E402
    pre_post_sharpe, subperiod_sharpes, time_decayed_sharpe,
)
from invest_system.validation.dsr import (  # noqa: E402
    sharpe_ratio, deflated_sharpe_ratio_from_returns,
)

START = get_env("J_EQ_START", "2016-07") or "2016-07"
END = get_env("J_EQ_END", "2026-05") or "2026-05"
TOP_N = int(get_env("J_EQ_TOP_N", "300") or "300")
LOOKBACK = 12          # trailing 流動性の窓（月）
QUANTILE, COSTS_BPS, LAG_DAYS = 0.2, 15.0, 4
MIN_NAMES, HALFLIFE, SPLIT = 20, 36.0, "2020-01-01"


def fetch_fundamentals(codes):
    # fins_summary/ 全件 by-date ミラーから長形式取得（旧 by-code statements/ も併合・重複除去）
    return load_fundamentals(codes)


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    print(f"=== PITユニバース再検証  {START}〜{END}  上位{TOP_N}（trailing{LOOKBACK}月）===")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]

    mask = point_in_time_universe(turn_c, top_n=TOP_N, lookback=LOOKBACK, min_obs=6)
    superset = universe_members(mask)
    print(f"superset(時変ユニバースの和集合)={len(superset)} 銘柄, "
          f"月平均所属={mask.sum(axis=1).mean():.0f}")
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    mask = mask.reindex(columns=superset).fillna(False)
    fwd = forward_returns(adj)
    rebal = adj.index
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]

    print(f"財務取得（superset {len(superset)}、未キャッシュのみ実取得）…")
    fund = fetch_fundamentals(superset)
    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    ff = value_quality_size_factors(pit, raw, adj)
    stl = point_in_time(margin.margin_imbalance(margin.load_weekly_margin()),
                        rebal, ["short_to_long"], date_col="Date",
                        lag_days=LAG_DAYS)["short_to_long"].reindex(columns=superset)

    def prep(f):
        # PITユニバースmaskを適用 → セクター中立 → z
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, mask),
                                                        sector))
    factors = {"size": prep(ff["size"]), "book_to_market": prep(ff["book_to_market"]),
               "momentum": prep(ff["momentum"]), "short_to_long": prep(stl)}

    series, sr_pp = {}, []
    for name, fac in factors.items():
        ls = long_short_returns(fac, fwd, quantile=QUANTILE, costs_bps=COSTS_BPS,
                                min_names=MIN_NAMES).dropna()
        series[name] = ls
        if ls.size >= 8 and ls.std(ddof=1) > 0:
            sr_pp.append(sharpe_ratio(ls))
    sr_var = float(np.var(sr_pp, ddof=1)) if len(sr_pp) > 1 else 0.0
    n_trials = 2 * len(factors)

    print(f"\n--- PITユニバース結果（n_trials={n_trials}, V[SR]={sr_var:.4f}）---")
    print(f"{'factor':<16}{'n':>4}{'SR':>7}{'dec':>7}{'side':>7}{'DSR':>6} | サブ期間 / 前後")
    for name, ls in series.items():
        if ls.size < 8 or ls.std(ddof=1) == 0:
            print(f"{name:<16} データ不足")
            continue
        sr = sharpe_ratio(ls) * np.sqrt(12)
        dec = time_decayed_sharpe(ls, HALFLIFE) * np.sqrt(12)
        d_hi = deflated_sharpe_ratio_from_returns(ls.values, sr_var, n_trials)
        d_lo = deflated_sharpe_ratio_from_returns((-ls).values, sr_var, n_trials)
        dsr, side = (d_lo, "低(逆)") if d_lo > d_hi else (d_hi, "高")
        thirds = " ".join(f"{s:+.2f}" for _, _, s in subperiod_sharpes(ls, 3))
        (_, pre), (_, post) = pre_post_sharpe(ls, SPLIT)
        print(f"{name:<16}{ls.size:>4}{sr:>7.2f}{dec:>7.2f}{side:>7}{dsr:>6.2f} | "
              f"{thirds} / 前{pre:+.2f} 後{post:+.2f}")

    print("\n--- 比較メモ（固定300ユニバース→PITユニバース）---")
    print("  バイアス除去後も結論が変わらなければ頑健。size減衰・value復活・")
    print("  short_to_long(低=ロング, 減衰) の各所見が生き残るかを上表で確認。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
