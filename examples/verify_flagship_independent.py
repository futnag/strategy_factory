"""P1-A: 旗艦の独立再計算（cross-engine 突合）— docs/04 P1-A・A2(arXiv:2603.20319)。

実装リスク（同一戦略でもエンジンが違えば結果が変わる）の検証：
- データ組立・ウェイト生成は research_value_pead_realism.py のシナリオ A 仕様と同一
  （月次・15bps 固定・execution_lag=0）。**ウェイト系列の再利用は docs/04 P1-A が許容**。
- ウェイト→損益（実現・コスト・回転率）と判定指標（年率 SR・maxDD）は、
  `research/engine.py`・`validation/dsr.py` を**使わず**、素朴・冗長・可読性優先の
  純 Python ループで独立に再計算し、エンジン出力と突合する。
- 突合基準（事前固定）：月次ネットリターンの最大絶対差 < 1e-8。
- 余力分（docs/04 P1-A タスク4）：T+1 始値（open→open）も `open_fill_backtest` と突合。
- 規律：throwaway＝新規試行ゼロ・永続レジストリ不使用（K 不変）。

実行: .venv\\Scripts\\python.exe examples\\verify_flagship_independent.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.config import get_env  # noqa: E402
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
# 参照値の生成にのみ使用（独立再計算側は import しない）
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, RegimeSwitch, backtest, open_fill_backtest,
)
from invest_system.timeseries import vol_regime  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
COSTS_BPS = 15.0
TOL = 1e-8          # 突合基準（事前固定）


# =========================================================================
# 独立再計算（engine.py / dsr.py 非依存・素朴・冗長・可読性優先）
# =========================================================================
def naive_close_fill_replay(weights_by_date: dict, close: pd.DataFrame,
                            dates, costs_bps: float) -> pd.Series:
    """素朴リプレイ：決定足の終値で執行し、翌足で実現する（execution_lag=0）。

    会計仕様（§6.10 シナリオ A をそのまま式にしたもの）：
      gross[t] = Σ_i w_t[i] × (close[t+1, i] / close[t, i] − 1)   ※価格欠損は寄与 0
      turn[t]  = Σ_i |w_t[i] − w_{t-1}[i]|   ※空シグナル月は全清算＝|w_{t-1}| を計上
      net[t]   = gross[t] − costs_bps / 1e4 × turn[t]
    ベクトル化せず 1 銘柄ずつ加算する（独立性・可読性優先）。
    """
    idx = list(close.index)
    next_of = {idx[k]: idx[k + 1] for k in range(len(idx) - 1)}
    out: dict[pd.Timestamp, float] = {}
    prev: dict[str, float] = {}
    for t in dates:
        nxt = next_of[t]
        w_raw = weights_by_date.get(t)
        w: dict[str, float] = {}
        if w_raw is not None:
            for code, x in w_raw.items():
                x = float(x)
                if x != 0.0:
                    w[str(code)] = x
        gross = 0.0
        for code, wi in w.items():
            p0 = float(close.at[t, code]) if code in close.columns else math.nan
            p1 = float(close.at[nxt, code]) if code in close.columns else math.nan
            if math.isfinite(p0) and math.isfinite(p1) and p0 != 0.0:
                gross += wi * (p1 / p0 - 1.0)
        turn = 0.0
        for code in set(w) | set(prev):
            turn += abs(w.get(code, 0.0) - prev.get(code, 0.0))
        out[t] = gross - costs_bps / 1e4 * turn
        prev = w
    return pd.Series(out, dtype="float64")


def naive_open_fill_replay(weights_by_date: dict, open_daily: pd.DataFrame,
                           costs_bps: float) -> pd.Series:
    """素朴リプレイ：決定日の翌営業日の始値で約定し、次の約定日まで保有（open→open）。

    仕様（docs/03 §6.13 シナリオ B＝open_fill_backtest と同じ会計）：
      約定日 f = 決定日 t より後の最初の営業日。最後の決定は評価不能で落とす。
      gross[t] = Σ_i w_t[i] × (open[f_{k+1}, i] / open[f_k, i] − 1)  ※欠損は寄与 0
      cost[t]  = costs_bps / 1e4 × Σ_i |w_t[i] − w_{t-1}[i]|
    """
    didx = list(open_daily.index)

    def first_after(t: pd.Timestamp):
        for d in didx:                       # 素朴な線形探索（独立性優先）
            if d > t:
                return d
        return None

    fills = []
    for t in sorted(weights_by_date, key=pd.Timestamp):
        f = first_after(pd.Timestamp(t))
        if f is not None:
            fills.append((pd.Timestamp(t), f, weights_by_date[t]))
    out: dict[pd.Timestamp, float] = {}
    prev: dict[str, float] = {}
    for k in range(len(fills) - 1):
        t, f0, w_raw = fills[k]
        f1 = fills[k + 1][1]
        w: dict[str, float] = {}
        for code, x in w_raw.items():
            x = float(x)
            if x != 0.0:
                w[str(code)] = x
        gross = 0.0
        for code, wi in w.items():
            p0 = float(open_daily.at[f0, code]) if code in open_daily.columns else math.nan
            p1 = float(open_daily.at[f1, code]) if code in open_daily.columns else math.nan
            if math.isfinite(p0) and math.isfinite(p1) and p0 != 0.0:
                gross += wi * (p1 / p0 - 1.0)
        turn = 0.0
        for code in set(w) | set(prev):
            turn += abs(w.get(code, 0.0) - prev.get(code, 0.0))
        out[t] = gross - costs_bps / 1e4 * turn
        prev = w
    return pd.Series(out, dtype="float64")


def naive_ann_sharpe(returns: pd.Series, periods_per_year: float = 12.0) -> float:
    """素朴式の年率 Sharpe（dsr.py 非依存）：mean/std(ddof=1)×√12。"""
    vals = [float(x) for x in returns.dropna()]
    n = len(vals)
    if n < 2:
        return math.nan
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    if var <= 0:
        return math.nan
    return mean / math.sqrt(var) * math.sqrt(periods_per_year)


def naive_max_drawdown(returns: pd.Series) -> float:
    """素朴式の最大ドローダウン：累積価値のピーク比最小。"""
    cum, peak, mdd = 1.0, 1.0, 0.0
    for x in returns.dropna():
        cum *= 1.0 + float(x)
        peak = max(peak, cum)
        mdd = min(mdd, cum / peak - 1.0)
    return mdd


def compare(label: str, engine_r: pd.Series, naive_r: pd.Series) -> tuple[bool, float]:
    """月次ネット系列の突合。(一致したか, 最大絶対差) を返し詳細を表示する。"""
    common = engine_r.index.intersection(naive_r.index)
    only_e = engine_r.index.difference(naive_r.index)
    only_n = naive_r.index.difference(engine_r.index)
    diffs = (engine_r.reindex(common) - naive_r.reindex(common)).abs()
    max_diff = float(diffs.max()) if len(diffs) else float("nan")
    ok = (len(only_e) == 0 and len(only_n) == 0
          and math.isfinite(max_diff) and max_diff < TOL)
    print(f"  {label:<22} 共通 {len(common)}ヶ月  最大|差| {max_diff:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    if len(only_e) or len(only_n):
        print(f"    index 不一致: engine のみ {list(only_e)} / 独立側のみ {list(only_n)}")
    if not ok and len(diffs):
        worst = diffs.sort_values(ascending=False).head(3)
        for d, v in worst.items():
            print(f"    {d:%Y-%m}: engine={engine_r[d]:+.10f} "
                  f"naive={naive_r[d]:+.10f} 差={v:.3e}")
    return ok, max_diff


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1

    # --- データ組立（research_value_pead_realism.py シナリオ A と同一・戦略不変）---
    print("データ組立中（§6.12 シナリオ A 仕様と同一）...")
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
    daily = load_daily_panel(field="AdjC")
    vol_m = vol_regime(daily).reindex(rebal, method="ffill")

    value_ls = CrossSectionalStrategy(value, 0.2, name="value")
    pead_lt = CrossSectionalStrategy(pead, 0.2, name="pead_longtilt", long_only=True)
    switch = RegimeSwitch(vol_m, {0: value_ls, 1: value_ls, 2: pead_lt},
                          name="switch")

    print(f"=== P1-A 独立再計算（{START}〜{END}・月次{len(rebal)}回・15bps・lag=0・"
          f"突合基準 |差|<{TOL:g}）===")

    # --- 1) 終値執行（シナリオ A）：engine.backtest vs 素朴リプレイ ---
    print("\n--- 終値執行（決定足終値・翌足実現＝シナリオ A）---")
    all_ok = True
    naive_net = {}
    for s in (value_ls, pead_lt, switch):
        weights = {t: s.target_weights(view.asof(t)) for t in rebal}
        ref = backtest(s, view, costs_bps=COSTS_BPS).returns
        naive = naive_close_fill_replay(weights, adj, list(rebal[:-1]), COSTS_BPS)
        ok, _ = compare(s.name, ref, naive)
        all_ok &= ok
        naive_net[s.name] = naive

    # --- 2) 判定指標（年率 SR・maxDD）も素朴式で再計算（dsr.py 非依存）---
    print("\n--- 判定指標の独立再計算（switch・素朴式）---")
    r = naive_net["switch"].dropna()
    oos = r[r.index >= pd.Timestamp(OOS)]
    print(f"  年率SR(全期間) {naive_ann_sharpe(r):+.2f} / SR(OOS {OOS}+) "
          f"{naive_ann_sharpe(oos):+.2f} / maxDD {naive_max_drawdown(r):.1%}")
    print("  参照（docs/03 §6.10/§6.12 シナリオA の報告値）: SR +0.97 / OOS +0.94 / "
          "maxDD −8.3%")

    # --- 3) T+1 始値（シナリオ B 系）：open_fill_backtest vs 素朴リプレイ ---
    print("\n--- T+1 始値執行（決定翌営業日の始値・open→open）---")
    open_daily = load_daily_panel(field="AdjO")
    w_switch = {t: switch.target_weights(view.asof(t)) for t in rebal}
    w_switch = {t: w for t, w in w_switch.items() if len(w)}
    ref_of = open_fill_backtest(w_switch, open_daily, costs_bps=COSTS_BPS).returns
    naive_of = naive_open_fill_replay(w_switch, open_daily, COSTS_BPS)
    ok, _ = compare("switch(T+1始値)", ref_of, naive_of)
    all_ok &= ok

    print(f"\n=== 総合判定: {'PASS（全系列が独立実装と一致）' if all_ok else 'FAIL（差分あり＝原因を特定し文書化のこと）'} ===")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
