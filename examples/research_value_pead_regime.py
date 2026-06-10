"""仮説検証：value/PEAD にレジーム条件を重ねて分離→改善するか（柱C × レジーム）。

§6.8 で確立した規律「**ゲート前に regime_breakdown で P&L のレジーム分離を確認**」を、エッジを
持つ戦略に適用する。value は唯一の耐久候補（§6.4 DSR0.79-0.84）、PEAD は IS強→OOS脆弱（§7）。
レジームで P&L が分離するなら、`RegimeGated`（＝ルールベースのメタラベル＝「この賭けに乗るか」）で
OOS安定/DSR を改善できるはず。分離が無ければ §6.8 同様の正直な負＝レジームは万能でないことを示す。

レジーム定義（事前固定・PIT・経済的根拠）：value は景気循環/ディストレス寄り→
 ① 高ボラ回避（flight-to-quality でバリュー劣後）② 強トレンド回避（モメンタム相場でバリュー劣後）。
日次マーケットの Efficiency Ratio / 実現ボラを拡張窓三分位化し、月末に as-of 整合（reindex ffill＝
先読み無）。regime[t] は月末 close[t] 由来＝因子と同 as-of。定義は事前固定＝定義探索で K を水増し
しない（KB §11.7）。ML メタラベリングは月数≈120 で過学習リスク高→まず規律版（規則メタラベル）。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_value_pead_regime.py
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
from invest_system.research import (  # noqa: E402
    AsOfView, CompositeStrategy, CrossSectionalStrategy, RegimeGated,
    RegimeSwitch, Strategy, backtest, judge_grid, regime_breakdown,
    walk_forward_regime_assignment, write_html,
)
from invest_system.timeseries import trend_regime, vol_regime  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import (  # noqa: E402
    TrialRegistry, default_registry,
)

START, END, OOS = "2016-07", "2026-05", "2024-01"
SCOPE = get_env("J_VPR_SCOPE", "value_pead_regime") or "value_pead_regime"
SWITCH_SCOPE = get_env("J_VPR_SWITCH_SCOPE", "value_pead_switch") or "value_pead_switch"
WF_SCOPE = get_env("J_VPR_WF_SCOPE", "value_pead_wfswitch") or "value_pead_wfswitch"
WF_WARMUP = int(get_env("J_VPR_WF_WARMUP", "24") or "24")     # 初期推定に充てる月数
WF_MINOBS = int(get_env("J_VPR_WF_MINOBS", "6") or "6")       # 同一レジームの過去最小本数
REG_PATH = get_env("J_VPR_REGISTRY", None)


class _Replay(Strategy):
    """事前計算した月次ウェイト（PIT生成済み）を date 引きで返す（適応切替の判定用）。"""

    def __init__(self, weights: dict, name: str, params: dict):
        self._w = weights
        self.name = name
        self.params = params

    def target_weights(self, asof):
        return self._w.get(asof.asof, pd.Series(dtype="float64"))


def _robustness_sweep(daily, view, rebal, wf_dates, value_ls, pead_lt, Wmap, Rv, Rp):
    """vol窓×コストの感応度スイープ（throwaway・**全構成の分布**で評価）。

    頑健性≠最適化：最良セルを選ばず、最悪セルでもエッジが残るかを見る（最良選別は p-hack）。
    value/PEAD 重みは regime 非依存ゆえ再利用。各 vol窓で switch/wf を gross+turnover で1回 backtest し、
    net SR は各コストで解析的（net=gross−cost·turnover）＝安価。永続レジストリには記録しない。
    """
    vol_windows = [20, 40, 60, 90, 120]
    costs = [10.0, 15.0, 25.0, 40.0]

    def net_sr(res, cost, oos=False):
        net = (res.gross - cost / 1e4 * res.turnover).dropna()
        if oos:
            net = net[net.index >= pd.Timestamp(OOS)]
        return float(sharpe_ratio(net) * np.sqrt(12)) if net.size >= 8 else float("nan")

    print("\n=== 頑健性スイープ（vol窓 × コスト・throwaway・分布で評価）===")
    print(f"  vol窓={vol_windows} / コストbps={[int(c) for c in costs]} / 指標=年率Sharpe")
    worst = {}
    for label in ("switch", "wf"):
        print(f"\n  [{label}]  行=vol窓 / 上段=全期間SR・下段=OOS-SR / 列=コスト{[int(c) for c in costs]}")
        oos_cells = []
        for vw in vol_windows:
            vmw = vol_regime(daily, window=vw).reindex(rebal, method="ffill")
            if label == "switch":
                strat = RegimeSwitch(vmw, {0.0: value_ls, 1.0: value_ls, 2.0: pead_lt},
                                     name=f"switch@{vw}")
            else:
                asg = walk_forward_regime_assignment(
                    {"value": Rv, "pead_longtilt": Rp}, vmw,
                    min_obs=WF_MINOBS, warmup=WF_WARMUP)
                AWw = {t: (Wmap[asg.get(t)][t] if isinstance(asg.get(t), str)
                           else pd.Series(dtype="float64")) for t in rebal}
                strat = _Replay(AWw, name=f"wf@{vw}", params={})
            res = backtest(strat, view, costs_bps=0.0, rebalance=wf_dates)
            allr = [net_sr(res, c) for c in costs]
            oosr = [net_sr(res, c, oos=True) for c in costs]
            oos_cells += oosr
            print(f"   {vw:>4} 全 " + "".join(f"{x:>+7.2f}" for x in allr))
            print(f"   {'':>4} OOS" + "".join(f"{x:>+7.2f}" for x in oosr))
        worst[label] = min(oos_cells)
    print(f"\n  最悪セルの OOS-SR: static switch={worst['switch']:+.2f} / "
          f"wf_switch={worst['wf']:+.2f}  （全{len(vol_windows) * len(costs)}構成で正なら"
          "結論は特定 vol窓/コストに依存しない＝頑健）")


def _isoos(v) -> None:
    """GridVerdict の各戦略を IS/OOS・pre/post-2020 の年率Sharpe で表示。"""
    for r in v.results:
        ls = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_ = ls[ls.index < pd.Timestamp(OOS)]
        oos = ls[ls.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        (_, pre), (_, post) = pre_post_sharpe(ls, "2020-01-01")
        print(f"  {r.name:<34} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={si:+.2f} OOS={so:+.2f} | 前2020={pre:+.2f} 後={post:+.2f}")


def _brk(name: str, series: pd.Series, trend_m: pd.Series, vol_m: pd.Series) -> None:
    """baseline 戦略の月次 P&L をトレンド/ボラ・レジーム別に年率Sharpe分解（ann=12）。"""
    fmt = lambda x: f"{x:+.2f}"  # noqa: E731
    bt = regime_breakdown(series, trend_m, ann=12.0).set_index("regime")
    bv = regime_breakdown(series, vol_m, ann=12.0).set_index("regime")
    def row(b):
        return "  ".join(f"r{int(k)}:SR{fmt(b.loc[k,'sharpe_ann'])}(n{int(b.loc[k,'n'])})"
                         for k in b.index)
    print(f"  [{name:<13}] トレンド {row(bt)}")
    print(f"  {'':<16} ボラ     {row(bv)}")


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
    adv = turn.reindex(columns=superset)
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

    # --- レジーム（日次マーケット→拡張窓三分位→月末 as-of 整合・PIT）---
    daily = load_daily_panel(field="AdjC")
    trend_m = trend_regime(daily).reindex(rebal, method="ffill")    # 0=レンジ…2=強トレンド
    vol_m = vol_regime(daily).reindex(rebal, method="ffill")        # 0=低…2=高ボラ
    print(f"=== value/PEAD × レジーム（{START}〜{END}・月末{len(rebal)}・scope={SCOPE}）===")
    print(f"レジーム被覆: トレンド {trend_m.notna().mean():.0%} / ボラ {vol_m.notna().mean():.0%}"
          f"（月末に as-of 整合）")

    value_ls = CrossSectionalStrategy(value, 0.2, name="value")
    pead_lt = CrossSectionalStrategy(pead, 0.2, name="pead_longtilt", long_only=True)
    combo = CompositeStrategy([value_ls, pead_lt], [0.5, 0.5], name="value+pead_lt")
    # 事前固定ゲート：高ボラ/強トレンド（regime 2）を回避（allowed={0,1}）
    strategies = [
        value_ls, pead_lt, combo,
        RegimeGated(value_ls, vol_m, allowed={0, 1}, name="value|vol<=1"),
        RegimeGated(value_ls, trend_m, allowed={0, 1}, name="value|trend<=1"),
        RegimeGated(combo, vol_m, allowed={0, 1}, name="value+pead_lt|vol<=1"),
        RegimeGated(combo, trend_m, allowed={0, 1}, name="value+pead_lt|trend<=1"),
    ]

    reg_cm = TrialRegistry(REG_PATH) if REG_PATH else default_registry()
    with reg_cm as reg:
        v = judge_grid(
            strategies, view, scope=SCOPE,
            hypothesis=("value/PEAD の P&L が市場レジームで分離するなら、高ボラ/強トレンド局面を"
                        "避ける regime ゲート（規則メタラベル）で OOS安定/DSR が改善するか"),
            economic_rationale=("value は景気循環/ディストレス寄りで flight-to-quality(高ボラ)・"
                                "モメンタム相場(強トレンド)に劣後しやすい。該当局面を外せば耐久性が"
                                "増す。レジームは経済的に動機づけ・事前固定。"),
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print("\n--- regime_breakdown（baseline・年率Sharpe・分離の有無を建玉前に診断）---")
    for nm in ("value", "pead_longtilt", "value+pead_lt"):
        s = v.series.get(nm, pd.Series(dtype="float64"))
        if not s.empty:
            _brk(nm, s, trend_m, vol_m)

    print(f"\n--- IS/OOS（gate・保留 {OOS}〜・年率Sharpe）---")
    _isoos(v)

    # === レジーム条件付き切替（§6.9 の逆ボラ依存を利用：平常=value, 混乱=PEAD）===
    print(f"\n=== レジーム切替 value↔PEAD（別 scope={SWITCH_SCOPE}）===")
    value_gate = RegimeGated(value_ls, vol_m, allowed={0, 1}, name="value|vol<=1")
    pead_turb = RegimeGated(pead_lt, vol_m, allowed={2}, name="pead_lt|vol==2")
    switch = RegimeSwitch(vol_m, {0: value_ls, 1: value_ls, 2: pead_lt},
                          name="switch(value@vol<=1,pead@vol==2)")
    reg_cm2 = TrialRegistry(REG_PATH) if REG_PATH else default_registry()
    with reg_cm2 as reg2:
        vs = judge_grid(
            [value_ls, value_gate, pead_turb, switch], view, scope=SWITCH_SCOPE,
            hypothesis=("value と PEAD はボラ依存が逆（value=平常・PEAD=混乱）。レジームで切替えれば"
                        "value|vol<=1 が現金待機する高ボラ月を PEAD で埋め、被覆と耐久性が上がるか"),
            economic_rationale=("高ボラ(flight-to-quality)で value は劣後するが予想改訂(PEAD)は不確実性下で"
                                "情報価値が増し効く。相補スリーブの切替で通期 risk-adjusted を改善。"
                                "注：割当は §6.9 の全期間 breakdown 由来＝in-sample 設計・OOS で要検証。"),
            registry=reg2, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + vs.report_md)
    print("HTML:", write_html(vs, f"data/reports/{vs.scope}.html"))
    print(f"\n--- IS/OOS（switch・保留 {OOS}〜・年率Sharpe）---")
    _isoos(vs)

    # === walk-forward 適応切替（割当を各時点で過去のみから学習＝in-sample 設計を排除）===
    print(f"\n=== walk-forward 適応切替（別 scope={WF_SCOPE}・warmup{WF_WARMUP}・min_obs{WF_MINOBS}）===")
    Wv = {t: value_ls.target_weights(view.asof(t)) for t in rebal}   # sleeve 月次ウェイト
    Wp = {t: pead_lt.target_weights(view.asof(t)) for t in rebal}
    Rv, Rp = v.series.get("value"), v.series.get("pead_longtilt")    # sleeve 月次ネット
    assign = walk_forward_regime_assignment({"value": Rv, "pead_longtilt": Rp}, vol_m,
                                            min_obs=WF_MINOBS, warmup=WF_WARMUP)
    Wmap = {"value": Wv, "pead_longtilt": Wp}
    AW = {t: (Wmap[assign.get(t)][t] if isinstance(assign.get(t), str)
              else pd.Series(dtype="float64")) for t in rebal}
    wf = _Replay(AW, name="wf_switch(value/pead@vol,past-learned)",
                 params={"rule": "argmax_past_mean_per_regime", "regime": "vol_tertile",
                         "min_obs": WF_MINOBS, "warmup": WF_WARMUP})
    wf_dates = rebal[WF_WARMUP:]                                     # 有効区間のみ判定
    static_map = {0.0: "value", 1.0: "value", 2.0: "pead_longtilt"}  # §6.10 の静的割当
    agree = np.mean([assign.get(t) == static_map.get(vol_m.get(t))
                     for t in wf_dates if pd.notna(vol_m.get(t))])
    cash = np.mean([not isinstance(assign.get(t), str) for t in wf_dates])
    reg_cm3 = TrialRegistry(REG_PATH) if REG_PATH else default_registry()
    with reg_cm3 as reg3:
        vw = judge_grid(
            [value_ls, value_gate, switch, wf], view, scope=WF_SCOPE, rebalance=wf_dates,
            hypothesis=("§6.10 の静的 switch の割当を各時点で過去のみから学習する walk-forward に置換しても"
                        "（in-sample 設計を排除しても）DSR/OOS が維持されるか＝エッジが本物か過学習か"),
            economic_rationale=("レジーム別に過去実績が最良の sleeve を因果的に選ぶ。value↔平常/PEAD↔混乱が"
                                "安定なら過去から再発見でき、walk-forward でも switch と同等＝過学習でない証拠。"),
            registry=reg3, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + vw.report_md)
    print("HTML:", write_html(vw, f"data/reports/{vw.scope}.html"))
    print(f"\n--- IS/OOS（walk-forward・保留 {OOS}〜・年率Sharpe）---")
    _isoos(vw)
    print(f"\n  walk-forward 割当が §6.10 静的割当と一致した割合: {agree:.0%}"
          f"（高いほどパターンが安定・過去から学習可能）／現金月: {cash:.0%}")

    print("\n※ 最終判定：wf_switch が switch（静的・in-sample）と同等の DSR/OOS を保てば、割当は過去から"
          " 学習可能＝エッジは過学習でない。崩れれば静的 switch は in-sample 産物。真の将来検証は 2026-05 以降。")

    if (get_env("J_VPR_ROBUST", "0") or "0") == "1":      # 頑健性スイープ（opt-in・throwaway）
        _robustness_sweep(daily, view, rebal, wf_dates, value_ls, pead_lt, Wmap, Rv, Rp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
