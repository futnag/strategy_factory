"""value × short_to_long の補完合成（レジーム横断の安定性を狙う）。

観測：short_to_long は 2016-19 に強く直近で減衰、value は逆（昔弱く直近強い）。
効く局面が真逆＝補完的。両者を符号を揃えて合成すれば「どの局面でも効く」より安定な
戦略になり得る——を、サブ期間安定性（主証拠）＋デフレートDSR（参考）で検証する。

注意（正直性）：short_to_long の向きは発見済みの符号（低=ロング）を用いるため、
DSR は楽観側に偏る。頑健な証拠は『前後2020の両方で正か』というサブ期間安定性。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\jp_combo_value_margin.py
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
from invest_system.equities.universe import select_universe  # noqa: E402
from invest_system.equities.panel import (  # noqa: E402
    assemble_panel, fetch_month_end_snapshots, forward_returns,
)
from invest_system.equities.fundamentals import point_in_time  # noqa: E402
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
QUANTILE, COSTS_BPS, LAG_DAYS = 0.2, 15.0, 4
MIN_NAMES, HALFLIFE, SPLIT = 20, 36.0, "2020-01-01"


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


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    print(f"=== value × short_to_long 補完合成  {START}〜{END}  上位{TOP_N} ===")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    universe = select_universe(listed, turn, top_n=TOP_N, min_obs=24)
    adj, raw = adj.reindex(columns=universe), raw.reindex(columns=universe)
    fwd = forward_returns(adj)
    rebal = adj.index
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]

    fund = fetch_fundamentals(universe)
    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    ff = value_quality_size_factors(pit, raw, adj)
    stl = point_in_time(margin.margin_imbalance(margin.load_weekly_margin()),
                        rebal, ["short_to_long"], date_col="Date",
                        lag_days=LAG_DAYS)["short_to_long"].reindex(columns=universe)

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(f, sector))
    # 符号を「高い=ロング（期待アウトパフォーム）」に揃える
    val_s = zN(ff["book_to_market"])      # 割安ロング
    stl_s = -zN(stl)                       # 低 short_to_long をロング（発見符号）
    mom_s = zN(ff["momentum"])             # 参考
    combo = (val_s + stl_s) / 2.0          # 補完合成
    signals = {"value": val_s, "short_to_long(rev)": stl_s,
               "momentum": mom_s, "combo(val+stl)": combo}

    series, sr_pp = {}, []
    for name, sig in signals.items():
        ls = long_short_returns(sig, fwd, quantile=QUANTILE, costs_bps=COSTS_BPS,
                                min_names=MIN_NAMES).dropna()
        series[name] = ls
        if ls.size >= 8 and ls.std(ddof=1) > 0:
            sr_pp.append(sharpe_ratio(ls))
    sr_var = float(np.var(sr_pp, ddof=1)) if len(sr_pp) > 1 else 0.0
    n_trials = 2 * len(signals)            # 符号探索込で保守的

    print(f"\n--- 結果（セクター中立・コスト{COSTS_BPS:.0f}bps・LS{QUANTILE:.0%}）  "
          f"n_trials={n_trials}, V[SR]={sr_var:.4f} ---")
    print(f"{'signal':<20}{'SR':>7}{'dec':>7}{'DSR':>6} | サブ期間 / 前後2020")
    rows = []
    for name, ls in series.items():
        if ls.size < 8 or ls.std(ddof=1) == 0:
            continue
        sr = sharpe_ratio(ls) * np.sqrt(12)
        dec = time_decayed_sharpe(ls, HALFLIFE) * np.sqrt(12)
        dsr = deflated_sharpe_ratio_from_returns(ls.values, sr_var, n_trials)
        thirds = subperiod_sharpes(ls, k=3)
        (_, pre), (_, post) = pre_post_sharpe(ls, SPLIT)
        seg = " ".join(f"{s:+.2f}" for _, _, s in thirds)
        print(f"{name:<20}{sr:>7.2f}{dec:>7.2f}{dsr:>6.2f} | {seg} / "
              f"前{pre:+.2f} 後{post:+.2f}")
        rows.append({"name": name, "sr": sr, "dsr": dsr, "pre": pre, "post": post})

    print("\n--- 判定 ---")
    c = next((r for r in rows if r["name"] == "combo(val+stl)"), None)
    if c:
        both_pos = c["pre"] > 0 and c["post"] > 0
        singles = [r for r in rows if r["name"] in ("value", "short_to_long(rev)")]
        best_single_dsr = max((r["dsr"] for r in singles), default=0)
        print(f"  合成: SR={c['sr']:+.2f}, DSR={c['dsr']:.2f}, "
              f"前2020={c['pre']:+.2f} / 後2020={c['post']:+.2f}")
        if both_pos:
            print("  ◎ 合成は前後2020の両局面で正＝補完が効きレジーム横断で安定。", end="")
        else:
            print("  ・合成は片局面で非正＝補完が不十分。", end="")
        print(f" 単独最良DSR={best_single_dsr:.2f} → 合成DSR={c['dsr']:.2f}"
              f"（{'改善' if c['dsr'] > best_single_dsr else '非改善'}）。")
        if c["dsr"] >= 0.95:
            print("  ★ 合成がDSR≥0.95＝多重検定後も有意（要・厳密OOS追確認）。")
        else:
            print("  ※ 符号向きは発見済み符号を使用＝DSRは楽観側。頑健な証拠はサブ期間"
                  "安定性。次：厳密OOS（直近2-3年を完全未使用）／メタラベルで局面ゲート。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
