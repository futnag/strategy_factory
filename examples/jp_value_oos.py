"""value を軸にした厳密OOS検証（PITユニバース・選択バイアスに正直）。

これまでの探索で「生存者バイアス除去後に唯一生き残る候補＝value」と分かった。
ここでは value を事前確定の戦略として、(1) IS/OOS 保留分割の整合性、(2) 選択
バイアスに正直なデフレートDSR（E[maxSR]）、(3) minTRL（認定に要する観測長）で
正直に評価する。チェリーピックを避けるため、単一指標(book_to_market)だけでなく
標準的なバリュー合成(B/M・E/P・CF/P・S/P)も同時に見る。

注意：探索段階で全期間を見ているため真のOOS純度は失われている。本質的な厳密さは
「因子は完全に因果的(PIT)」＋「選択を E[maxSR]/minTRL で罰する」点にある。OOSの
IS整合性は補助証拠。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\jp_value_oos.py
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
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.validation.dsr import (  # noqa: E402
    _moments, deflated_sharpe_ratio, min_track_record_length,
    probabilistic_sharpe_ratio, sharpe_ratio,
)

START = get_env("J_EQ_START", "2016-07") or "2016-07"
END = get_env("J_EQ_END", "2026-05") or "2026-05"
OOS_START = get_env("J_EQ_OOS", "2024-01") or "2024-01"   # これ以降を保留OOS
TOP_N, LOOKBACK = 300, 12
QUANTILE, COSTS_BPS = 0.2, 15.0
PROGRAM_TRIALS = 30          # プログラム全体で見た因子数（選択バイアスの保守的見積り）
VALUE_FIELDS = ["ShOutFY", "TrShFY", "Eq", "FEPS", "CFO", "FSales"]


def fetch_fundamentals(codes):
    # fins_summary/ 全件 by-date ミラーから長形式取得（旧 by-code statements/ も併合・重複除去）
    return load_fundamentals(codes)


def _nanmean_frames(frames):
    stack = np.array([f.values for f in frames], dtype=float)     # (k, T, N)
    cnt = np.sum(~np.isnan(stack), axis=0)
    ssum = np.nansum(stack, axis=0)
    arr = np.where(cnt > 0, ssum / np.where(cnt == 0, 1, cnt), np.nan)
    return pd.DataFrame(arr, index=frames[0].index, columns=frames[0].columns)


def stats(ls, n_trials, sr_var):
    ls = ls.dropna()
    sr, sk, ku, n = _moments(ls.values)
    psr0 = probabilistic_sharpe_ratio(sr, 0.0, n, sk, ku)
    dsr = deflated_sharpe_ratio(sr, sr_var, n_trials, n, sk, ku)
    try:
        mtrl = min_track_record_length(sr, 0.0, sk, ku, 0.95)
    except ValueError:
        mtrl = float("inf")
    return dict(n=n, sr_ann=sr * np.sqrt(12), psr0=psr0, dsr=dsr, mtrl=mtrl)


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    print(f"=== value 厳密OOS  IS:{START}〜{OOS_START}前 / OOS:{OOS_START}〜{END} ===")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    mask = point_in_time_universe(turn_c, top_n=TOP_N, lookback=LOOKBACK, min_obs=6)
    superset = universe_members(mask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    mask = mask.reindex(columns=superset).fillna(False)
    fwd = forward_returns(adj)
    rebal = adj.index
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    print(f"PITユニバース superset={len(superset)}（月平均{mask.sum(axis=1).mean():.0f}）")

    fund = fetch_fundamentals(superset)
    pit = point_in_time(fund, rebal, VALUE_FIELDS, lag_days=1)
    ff = value_quality_size_factors(pit, raw, adj)

    def prep(f):
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, mask),
                                                        sector))
    metrics = {k: prep(ff[k]) for k in
               ("book_to_market", "earnings_yield", "cf_yield", "sales_yield")}
    value_composite = _nanmean_frames(list(metrics.values()))

    cand = {"book_to_market": metrics["book_to_market"],
            "value_composite": value_composite}
    # デフレート用：4指標＋合成の per-period SR 分散（value探索の試行分散）
    srs = []
    series = {}
    for k, f in {**metrics, "value_composite": value_composite}.items():
        ls = long_short_returns(f, fwd, quantile=QUANTILE, costs_bps=COSTS_BPS,
                                min_names=20).dropna()
        series[k] = ls
        if ls.size >= 8 and ls.std(ddof=1) > 0:
            srs.append(sharpe_ratio(ls))
    sr_var = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.01

    print(f"\n--- 事前確定バリュー戦略の評価（V[SR]={sr_var:.4f}）---")
    for name, f in cand.items():
        ls = series[name]
        full = stats(ls, n_trials=len(srs), sr_var=sr_var)
        prog = stats(ls, n_trials=PROGRAM_TRIALS, sr_var=sr_var)
        is_ls = ls[ls.index < pd.Timestamp(OOS_START)]
        oos_ls = ls[ls.index >= pd.Timestamp(OOS_START)]
        sr_is = (sharpe_ratio(is_ls) * np.sqrt(12)
                 if is_ls.size >= 8 and is_ls.std(ddof=1) > 0 else np.nan)
        if oos_ls.size >= 8 and oos_ls.std(ddof=1) > 0:
            o_sr, o_sk, o_ku, o_n = _moments(oos_ls.values)
            sr_oos = o_sr * np.sqrt(12)
            psr_oos = probabilistic_sharpe_ratio(o_sr, 0.0, o_n, o_sk, o_ku)
            hit_oos = float((oos_ls > 0).mean())
        else:
            sr_oos = psr_oos = hit_oos = np.nan
        print(f"\n[{name}]")
        print(f"  全期間 n={full['n']}  SR(ann)={full['sr_ann']:+.2f}  "
              f"PSR(>0)={full['psr0']:.2f}")
        print(f"  デフレートDSR: value内探索(n={len(srs)})={full['dsr']:.2f} / "
              f"プログラム全体(n={PROGRAM_TRIALS})={prog['dsr']:.2f}")
        print(f"  minTRL(95%認定に要する月数)={full['mtrl']:.0f}  "
              f"（保有 {full['n']} か月）")
        print(f"  IS  SR(ann)={sr_is:+.2f}  ({is_ls.size}か月)")
        print(f"  OOS SR(ann)={sr_oos:+.2f}  勝率={hit_oos:.0%}  "
              f"PSR(>0)={psr_oos:.2f}  ({oos_ls.size}か月)")
        (_, pre), (_, post) = pre_post_sharpe(ls, "2020-01-01")
        print(f"  前2020 SR={pre:+.2f} / 後2020 SR={post:+.2f}")

    print("\n--- 判定 ---")
    print("  ・DSR≥0.95 かつ OOSもIS同方向で正 → 実運用候補。")
    print("  ・minTRL >> 保有月数 なら『本物でも認定に履歴不足』＝Phase2は小口/監視で。")
    print("  ・プログラム全体デフレートDSRが低い → 選択バイアス込みでは未確立。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
