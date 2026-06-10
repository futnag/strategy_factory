"""仮説検証：旗艦（value↔PEAD switch）× マルチアセットTSMOM の合成（柱C×柱E）。

§6.15 の「保留」判定を裁く最終ゲート：TSMOM の採否は単体 DSR でなく**旗艦との合成が
リスク調整後を改善するか**で決める（§6.5 breadth・IR≈IC×√breadth）。両者は経済系統が
独立（日本株バリュー需給 vs グローバル・マクロの漸進的織り込み）・実測相関 +0.1〜0.2・
危機時の非対称性が逆（switch=平時に強い／TSMOM=COVID窓で正）＝合成の教科書的候補。

事前登録（K=4・選別回避の設計）：
  ① switch 単体（§6.10 正準・15bps・容量込み）＝ベースライン
  ② tsmom_blend 単体（§6.15 の a priori 等加重ブレンド。**best-of-grid の 12m を選ばない**
     ＝同一データで勝った変種を持ち込む選択バイアスを避ける）
  ③ combo_eqcap：資本 50/50（毎月リバランス）
  ④ combo_eqrisk：リスク 50/50（PIT 逆ボラ、トレーリング36ヶ月・最低12ヶ月、t−1 までの
     実現リターンのみ使用）
合成はスリーブ純資産指数（コスト控除後ネットの cumprod）を資産とみなす2資産ビューで
エンジンに載せる。合成層のコストは 0bps（コストは各スリーブ内で計上済み。スリーブ間の
資金移動は証拠金の付け替えで実取引を伴わない）。月境界はスリーブごとの暦（JP月末 vs
グローバル月末）が数日ずれるが、スリーブは各自の暦で運用される実務と同じ＝月単位で整合。

数理的な事前予想（正直に書く）：SR 差が大きい（switch≈1.0 vs blend≈0.3）ため、等リスク
合成の SR は (SR1+SR2)/√(2(1+ρ)) ≈ 0.87 と**単体 switch を下回り得る**。限界改善条件
SR2 > ρ×SR1 ≈ 0.2 は満たすため小さい配分なら必ず改善するが、50/50 が最適とは限らない。
判定は SR/DSR に加えて maxDD・危機窓・OOS で多面的に見る。

実行: .venv\\Scripts\\python.exe examples\\research_switch_tsmom_combo.py
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
from invest_system.data.external import load_external_prices  # noqa: E402
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
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.portfolio import kelly_fraction  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, RegimeSwitch, Strategy, backtest,
    judge_grid, write_html,
)
from invest_system.research.strategies_tsmom import (  # noqa: E402
    annualized_vol, blend_weights, tsmom_weights,
)
from invest_system.timeseries import vol_regime  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
SCOPE = "switch_tsmom_combo"
TSMOM_KEYS = ["nk225_fut", "sp500", "nasdaq_comp", "gold", "silver", "platinum",
              "wti", "copper", "usdjpy", "eurjpy", "audjpy"]


class _Replay(Strategy):
    def __init__(self, weights: dict, name: str, params: dict):
        self._w = weights
        self.name = name
        self.params = params

    def target_weights(self, asof):
        return self._w.get(asof.asof, pd.Series(dtype="float64"))


def _sr(x: pd.Series, lo=None, hi=None) -> float:
    r = x.dropna()
    if lo is not None:
        r = r[r.index >= pd.Timestamp(lo)]
    if hi is not None:
        r = r[r.index < pd.Timestamp(hi)]
    return float(sharpe_ratio(r) * np.sqrt(12)) if r.size >= 8 else float("nan")


def _maxdd(r: pd.Series) -> float:
    cum = (1.0 + r.dropna()).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def _flagship_switch_net() -> pd.Series:
    """§6.10 正準構成の switch 月次ネット（15bps・JP月末暦）を再現する。"""
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
    return backtest(switch, view, costs_bps=15.0).returns.dropna()


def _tsmom_blend_net() -> pd.Series:
    """§6.15 事前登録構成の tsmom_blend 月次ネット（5bps・T+1始値・グローバル月末暦）。"""
    cl = load_external_prices(TSMOM_KEYS, field="close")
    op = load_external_prices(TSMOM_KEYS, field="open")
    cl_ff = cl.ffill(limit=7)
    m_close = cl_ff.groupby(cl_ff.index.to_period("M")).tail(1)
    rebal = m_close.index
    vol_m = annualized_vol(cl, window=63, floor=0.05).ffill(limit=7).reindex(rebal)
    fill_px = op.bfill(limit=3).shift(-1).reindex(rebal)
    view = AsOfView({"close": fill_px})
    sets = [tsmom_weights(m_close, vol_m, lb, vol_target=0.10) for lb in (3, 6, 12)]
    strat = _Replay(blend_weights(sets), "tsmom_blend", {})
    return backtest(strat, view, costs_bps=5.0).returns.dropna()


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    print("スリーブ生成中（switch=§6.10 正準 / tsmom_blend=§6.15 事前登録構成）…")
    rs = _flagship_switch_net()
    rt = _tsmom_blend_net()

    # --- 月単位で整合（各スリーブは自分の暦で運用＝月ラベルで結合）---
    s = rs.copy()
    s.index = rs.index.to_period("M")
    t = rt.copy()
    t.index = rt.index.to_period("M")
    t = t[~t.index.duplicated(keep="last")]
    months = s.index.intersection(t.index)
    R = pd.DataFrame({"switch": s.reindex(months), "tsmom": t.reindex(months)}).dropna()
    canon = pd.DatetimeIndex([rs.index[list(s.index).index(m)] for m in R.index])
    R.index = canon                              # 正準 index＝JP 月末の実営業日
    print(f"共通期間: {R.index[0]:%Y-%m}〜{R.index[-1]:%Y-%m}（{len(R)}ヶ月） "
          f"月次相関={R['switch'].corr(R['tsmom']):+.2f}")

    # --- スリーブ純資産指数の2資産ビュー（合成をエンジン契約に載せる）---
    # ラベル整合：R[t] は「月 t→t+1 に実現するスリーブ・リターン」（engine 規約）。
    # NAV は実現後の翌ラベルに置く＝ pct_change[t+1]==R[t] となり、決定日 t の合成
    # ウェイトがちょうど R[t] を受け取る（決定時点で R[t] は未知＝PIT 整合）。
    nav = (1.0 + R).cumprod()
    nav.index = list(R.index[1:]) + [R.index[-1] + pd.offsets.MonthEnd(1)]
    base = pd.DataFrame(1.0, index=[R.index[0]], columns=nav.columns)
    sleeve_px = pd.concat([base, nav]).sort_index()
    view = AsOfView({"close": sleeve_px})
    dates = sleeve_px.index

    w_sw = {d: pd.Series({"switch": 1.0}) for d in dates}
    w_ts = {d: pd.Series({"tsmom": 1.0}) for d in dates}
    w_cap = {d: pd.Series({"switch": 0.5, "tsmom": 0.5}) for d in dates}
    # 逆ボラ（PIT：決定日 t に既知なのは t−1 ラベルまでの実現＝shift(1)）
    vol36 = R.shift(1).rolling(36, min_periods=12).std()
    w_risk = {}
    for d in dates:
        v = vol36.loc[d] if d in vol36.index else pd.Series(dtype="float64")
        iv = (1.0 / v).replace([np.inf, -np.inf], np.nan).dropna()
        if len(iv) == 2:
            w_risk[d] = iv / iv.sum()
        else:
            w_risk[d] = pd.Series({"switch": 1.0})   # ボラ推定前は旗艦のみ（保守）
    strategies = [
        _Replay(w_sw, "switch_only", {"sleeves": "switch"}),
        _Replay(w_ts, "tsmom_only", {"sleeves": "tsmom_blend"}),
        _Replay(w_cap, "combo_eqcap", {"rule": "capital_5050"}),
        _Replay(w_risk, "combo_eqrisk", {"rule": "inverse_vol36_pit"}),
    ]

    with default_registry() as reg:
        v = judge_grid(
            strategies, view, scope=SCOPE,
            hypothesis=("経済系統が独立で相関+0.1〜0.2の2スリーブ（日本株 value↔PEAD switch と"
                        "マルチアセットTSMOM）の合成は、リスク調整後（DSR/maxDD/OOS）を"
                        "単体旗艦より改善するか（breadth・IR≈IC×√breadth）"),
            economic_rationale=("switch=日本株の割安需給とレジーム切替（平時に強い）、TSMOM="
                                "グローバル・マクロの漸進的織り込み（危機窓で正）＝収益源と"
                                "非対称性が逆の独立スリーブ。コストは各スリーブ内で控除済み・"
                                "合成層は証拠金付け替えのみ＝0bps。SR差が大きく等リスクでは"
                                "SR が下がり得ることも事前予想に明記（docstring）。"),
            registry=reg, costs_bps=0.0)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    # --- 多面診断：maxDD・OOS・危機窓・限界改善・ケリー ---
    print(f"\n--- 多面比較（保留 {OOS}〜・年率）---")
    for r in v.results:
        x = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        (_, pre), (_, post) = pre_post_sharpe(x, "2020-01-01")
        print(f"  {r.name:<13} SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} "
              f"maxDD={_maxdd(x):>6.1%} | IS={_sr(x, hi=OOS):+.2f} "
              f"OOS={_sr(x, lo=OOS):+.2f} | 前/後2020={pre:+.2f}/{post:+.2f}")

    # 限界改善条件（ポートフォリオ理論）：SR_t > ρ×SR_s なら小配分で必ず改善
    sr_s, sr_t = _sr(R["switch"]), _sr(R["tsmom"])
    rho = float(R["switch"].corr(R["tsmom"]))
    alpha = R["tsmom"] - (rho * R["tsmom"].std() / R["switch"].std()) * R["switch"]
    print(f"\n  限界改善条件: SR_tsmom={sr_t:+.2f} vs ρ×SR_switch={rho * sr_s:+.2f} "
          f"→ {'満たす（小配分の追加は理論上必ず改善）' if sr_t > rho * sr_s else '満たさない'}")
    print(f"  残差アルファ（switch でヘッジ後の tsmom）: 年率 "
          f"{float(alpha.mean() * 12):+.2%}・SR {_sr(alpha):+.2f}")
    for label, lo, hi in [("2020-02..03(COVID)", "2020-02", "2020-04"),
                          ("2022(金利急騰)", "2022-01", "2023-01")]:
        seg = R[(R.index >= lo) & (R.index < hi)]
        print(f"  {label:<20} switch {float((1 + seg['switch']).prod() - 1):+7.2%} / "
              f"tsmom {float((1 + seg['tsmom']).prod() - 1):+7.2%}")

    best_combo = max((r for r in v.results if r.name.startswith("combo")),
                     key=lambda r: r.dsr)
    rb = v.series[best_combo.name].dropna()
    for frac in (0.5, 0.25):
        k = kelly_fraction(rb, fraction=frac)
        print(f"  {best_combo.name} {frac:.2f}×Kelly: {k.summary()}")

    print("\n※ 採否規準（事前登録）：合成が switch 単体を DSR と maxDD の**両方**で上回れば"
          "「TSMOM オーバーレイを Phase 2 に併載」、SR/DSR が劣るが DD・危機耐性のみ改善なら"
          "「フォワードで並走・本載せ見送り」、どちらも改善しなければ「不採用（保留解除）」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
