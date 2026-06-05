"""信用・空売りファクターのクロスセクション検証（Standard・10年・pillar C規律）。

価格・財務に続く第3の系統＝需給（信用残高・空売り残高）が、value/size の先で
独立したエッジを持つかを正直に検証する。同じ規律：ポイントインタイム整合（公表
遅延を考慮したラグ）・セクター中立・試行数デフレートDSR・サブ期間安定性。

新ファクター（個別株・クロスセクション）:
  margin_imbalance = (信用買残−売残)/(買残+売残)   週次
  short_to_long    = 信用売残/信用買残              週次
  short_interest   = 対発行株数の空売り残高(報告者合算)  報告ベース
※ 符号仮説は中立。各ファクターは「高い=ロング」で評価し、負なら逆張り側が候補。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\jp_margin_short_study.py
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
    cross_sectional_zscore, sector_neutralize,
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
QUANTILE = 0.2
COSTS_BPS = 15.0
LAG_DAYS = 4          # 週次信用(T+2公表)・空売り残高(T+2開示)の公表遅延を吸収
MIN_MONTHS = 8
MIN_NAMES = 20
MIN_OBS_UNIV = 24
HALFLIFE = 36.0
SPLIT_DATE = "2020-01-01"


def evaluate(name, factor, fwd, sector):
    z = cross_sectional_zscore(sector_neutralize(factor, sector))
    ls = long_short_returns(z, fwd, quantile=QUANTILE, costs_bps=COSTS_BPS,
                            min_names=MIN_NAMES).dropna()
    st = {"name": name, "n": int(ls.size), "mean_m": np.nan,
          "sharpe_ann": np.nan, "sharpe_dec": np.nan}
    if ls.size >= MIN_MONTHS and ls.std(ddof=1) > 0:
        spp = sharpe_ratio(ls)
        st.update(mean_m=float(ls.mean()), sharpe_pp=spp,
                  sharpe_ann=spp * np.sqrt(12),
                  sharpe_dec=time_decayed_sharpe(ls, HALFLIFE) * np.sqrt(12))
    else:
        st["sharpe_pp"] = np.nan
    return ls, st


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    print(f"=== 信用・空売りファクター研究  {START}〜{END}  上位{TOP_N}銘柄 ===")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj = assemble_panel(snaps, "AdjC")
    turn = assemble_panel(snaps, "Va")
    universe = select_universe(listed, turn, top_n=TOP_N, min_obs=MIN_OBS_UNIV)
    adj = adj.reindex(columns=universe)
    fwd = forward_returns(adj)
    rebal = adj.index
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"] \
        if "S33" in listed.columns else pd.Series(dtype=object)
    print(f"パネル {adj.shape[0]}か月 × {len(universe)}銘柄（流動性上位）")

    print("信用・空売りキャッシュをロード中…")
    weekly = margin.load_weekly_margin()
    positions = margin.load_short_positions()
    print(f"  週次信用 {len(weekly):,}行 / 空売り残高 {len(positions):,}行")
    mi = margin.margin_imbalance(weekly)
    si = margin.short_interest(positions)

    pit_m = point_in_time(mi, rebal, ["margin_imbalance", "short_to_long"],
                          date_col="Date", code_col="Code", lag_days=LAG_DAYS)
    pit_s = point_in_time(si, rebal, ["short_interest"],
                          date_col="Date", code_col="Code", lag_days=LAG_DAYS)
    factors = {
        "margin_imbalance": pit_m["margin_imbalance"].reindex(columns=universe),
        "short_to_long": pit_m["short_to_long"].reindex(columns=universe),
        # 空売り残高は0.5%未満は非開示＝該当無しは0（小さい空売り）として全銘柄に定義
        "short_interest": pit_s["short_interest"].reindex(columns=universe).fillna(0.0),
    }

    series, stats = {}, []
    for name, fac in factors.items():
        ls, st = evaluate(name, fac, fwd, sector)
        series[name] = ls
        stats.append(st)

    valid = [s for s in stats if not np.isnan(s["sharpe_pp"])]
    sr_pp = np.array([s["sharpe_pp"] for s in valid])
    sr_var = float(np.var(sr_pp, ddof=1)) if len(sr_pp) > 1 else 0.0
    # 符号探索（高/低どちらをロングか）を考慮し試行数を 2x で保守的にデフレート
    n_trials = max(2 * len(valid), 1)
    for s in valid:
        ls = series[s["name"]]
        try:
            d_hi = deflated_sharpe_ratio_from_returns(ls.values, sr_var, n_trials)
            d_lo = deflated_sharpe_ratio_from_returns((-ls).values, sr_var, n_trials)
        except Exception:  # noqa: BLE001
            d_hi = d_lo = np.nan
        # 有利な側（高ロング or 低ロング=逆張り）のDSRを採用
        if np.isnan(d_hi) or np.isnan(d_lo):
            s["dsr"], s["side"] = np.nan, "?"
        elif d_lo > d_hi:
            s["dsr"], s["side"] = d_lo, "低(逆)"
        else:
            s["dsr"], s["side"] = d_hi, "高"

    valid.sort(key=lambda s: s["dsr"] if not np.isnan(s["dsr"]) else -9, reverse=True)
    print(f"\n=== 結果（セクター中立・コスト{COSTS_BPS:.0f}bps・LS{QUANTILE:.0%}・"
          f"ラグ{LAG_DAYS}日）===")
    print(f"試行数(符号探索込) n_trials={n_trials}, V[SR]={sr_var:.4f}")
    print(f"{'factor':<18}{'n':>4}{'SR(ann)':>9}{'SR_dec':>8}{'side':>7}{'DSR':>7}")
    print("-" * 56)
    for s in valid:
        print(f"{s['name']:<18}{s['n']:>4}{s['sharpe_ann']:>9.2f}"
              f"{s['sharpe_dec']:>8.2f}{s['side']:>7}{s['dsr']:>7.2f}")

    print("\n--- サブ期間安定性（年率Sharpe）---")
    for s in valid:
        ls = series[s["name"]]
        thirds = subperiod_sharpes(ls, k=3)
        (npre, shpre), (npost, shpost) = pre_post_sharpe(ls, SPLIT_DATE)
        seg = "  ".join(f"{lbl}:{sh:+.2f}" for lbl, _, sh in thirds)
        print(f"  {s['name']:<16} 全{s['sharpe_ann']:+.2f} | {seg} | "
              f"前{SPLIT_DATE[:4]}:{shpre:+.2f} 後:{shpost:+.2f}")

    surv = [s for s in valid if not np.isnan(s["dsr"]) and s["dsr"] >= 0.95]
    print("\n--- 判定（DSR≥0.95）---")
    if surv:
        for s in surv:
            print(f"  ★ {s['name']}（{s['side']}ロング）: SR(ann)={s['sharpe_ann']:+.2f}, "
                  f"DSR={s['dsr']:.3f}")
    else:
        b = valid[0] if valid else None
        print("  生存ファクター無し。", end="")
        if b:
            print(f" 最良 {b['name']}（{b['side']}ロング）DSR={b['dsr']:.2f}（未達）。")
        print("  ※ |SR|大かつ符号が安定なら次段(因果/メタ/合成)で再検証する価値あり。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
