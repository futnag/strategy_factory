"""仮説検証：value+PEAD 合成にメタラベル局面ゲート（二次モデル）を乗せる。

事前登録（FROZEN）：docs/25-meta-gate-preregistration.md。
- 一次＝value+pead_lt 合成（research_value_pead_longtilt.py と同一構成・無改修）。
- 二次＝各月 t で「来月 primary が勝つ確率」を t 以前のみで walk-forward 学習し、
  bet_size_from_prob（AFML snippet 10.1）で連続サイズ∈[0,1]→一次ウェイトに乗じる。
- 局面特徴量5本（凍結）：TOPIX ボラ/モメンタム・value 自己 trailing Sharpe・
  当月ディスパージョン・海外勢ネットフロー強度。すべて PIT。
- 判定：ungated と gated を同一 scope `value_pead_meta_gate` の K で同時デフレート。
  PASS = gated OOS SR ≥ ungated AND gated DSR ≥ ungated AND gated DSR ≥ 0.95。

これは**一度きり**の判定。結果を見てから特徴量・モデル・分割・基準を変えない（docs/25 §5）。
事前に `examples/fetch_indices_flows.py` で指数・投資部門別キャッシュを用意すること。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_value_pead_meta_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import events, flows  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.fundamentals import load_fundamentals, point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CompositeStrategy, CrossSectionalStrategy, MetaGatedStrategy,
    backtest, fit_meta_gate, judge_grid, regime_breakdown, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
WARMUP, EMBARGO = 36, 1               # FROZEN: docs/25 §3.3


def _trailing_sharpe(net: pd.Series, window: int = 12) -> pd.Series:
    """各月までの trailing window Sharpe（per-period）。PIT 用に shift(1)（≤t-1 の実績）。"""
    def _sr(x):
        sd = np.std(x, ddof=1)
        return float(np.mean(x) / sd) if sd > 0 else np.nan
    return net.rolling(window).apply(_sr, raw=True).shift(1)


def build_features(rebal, adj, umask, value_net) -> pd.DataFrame:
    """凍結5特徴量（全て t 時点で PIT）。欠損は当月ゲート無効として扱われる。"""
    feat = pd.DataFrame(index=rebal)

    # 1-2. TOPIX 実現ボラ・モメンタム（直近6か月）
    try:
        topix = jq.fetch_index_bars(code="0000")
        tcl = topix.dropna(subset=["Date"]).set_index("Date")["C"].sort_index()
        tm = tcl.reindex(rebal, method="ffill")
        tret = tm.pct_change()
        feat["topix_vol"] = tret.rolling(6).std()
        feat["topix_mom"] = tm / tm.shift(6) - 1.0
    except Exception as e:                                   # noqa: BLE001
        print(f"WARN: TOPIX 指数の取得に失敗（topix 特徴量は NaN）: {e}")
        feat["topix_vol"] = np.nan
        feat["topix_mom"] = np.nan

    # 3. value スリーブ自身の trailing 12m Sharpe（自己状態＝崩れ検知）
    feat["value_trail"] = _trailing_sharpe(value_net, 12).reindex(rebal)

    # 4. 当月ユニバースの横断ディスパージョン（直近1か月リターンの銘柄間 std）
    r1 = adj.pct_change()
    feat["dispersion"] = apply_universe_mask(r1, umask).std(axis=1).reindex(rebal)

    # 5. 海外勢ネットフロー強度（週次→月末 ffill・PIT）
    inv = flows.load_investor_types()
    fi = flows.net_flow_intensity(inv, investor="foreign", section="TokyoNagoya")
    if fi.empty:
        print("WARN: 投資部門別データ未取得（flow_intensity は NaN）。"
              "先に examples/fetch_indices_flows.py を実行。")
        feat["flow_intensity"] = np.nan
    else:
        feat["flow_intensity"] = fi.sort_index().reindex(rebal, method="ffill")
    return feat


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

    # --- 一次モデル（無改修）---
    value_ls = CrossSectionalStrategy(value, 0.2, name="value")
    pead_lt = CrossSectionalStrategy(pead, 0.2, name="pead_longtilt", long_only=True)
    combo = CompositeStrategy([value_ls, pead_lt], [0.5, 0.5], name="value+pead_lt")

    # 一次のネット系列（メタラベルの教師＝walk-forward 内でのみ過去を参照）
    combo_net = backtest(combo, view, costs_bps=15.0, adv=adv).returns.dropna()
    value_net = backtest(value_ls, view, costs_bps=15.0, adv=adv).returns.dropna()

    # --- 局面特徴量（凍結5本・PIT）---
    feat = build_features(rebal, adj, umask, value_net).reindex(combo_net.index)
    cov = feat.notna().all(axis=1).sum()
    print(f"特徴量カバレッジ（全5本そろう月）= {cov}/{len(feat)} 月")

    # --- 診断ファースト：ungated の局面分離（throwaway・K 不変）---
    regime = (feat["topix_mom"] > 0).astype(float)         # 1=上昇局面 / 0=下落局面
    bd = regime_breakdown(combo_net, regime)
    print("\n--- 診断: ungated value+pead_lt のTOPIXモメンタム局面別 ---")
    print(bd.to_string(index=False))
    print("（分離が無ければゲートは無意味＝NO-GO。あれば下で正式判定）")

    # --- 二次モデル（walk-forward メタゲート）---
    scale = fit_meta_gate(combo_net, feat, warmup=WARMUP, embargo=EMBARGO)
    active = scale.notna().sum()
    print(f"\nメタゲート有効月数 = {active}/{len(scale)}"
          f"（warmup={WARMUP}・特徴欠損月は無効）")
    gated = MetaGatedStrategy(combo, scale, name="value+pead_lt|meta_gate")

    # --- 正式判定：ungated と gated を同一 scope で同時デフレート ---
    with default_registry() as reg:
        v = judge_grid([combo, gated], view, scope="value_pead_meta_gate",
                       hypothesis="value+PEADのOOS失速は局面依存で、PIT局面特徴の二次メタモデルで勝率を学習しサイズを絞れば改善する",
                       economic_rationale="PEADショートは割安株を売りvalueロングと衝突する。崩れは個別でなく市場局面に駆動＝局面条件付きで露出を絞れば不利局面の負けを構造的に削れる",
                       registry=reg, costs_bps=15.0, adv=adv, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    # --- IS/OOS（保留 OOS〜）---
    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    res = {r.name: r for r in v.results}
    for name in ("value+pead_lt", "value+pead_lt|meta_gate"):
        r = res.get(name)
        if r is None:
            continue
        ls = v.series.get(name, pd.Series(dtype="float64")).dropna()
        oos = ls[ls.index >= pd.Timestamp(OOS)]
        is_ = ls[ls.index < pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {name:<26} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={si:+.2f} OOS={so:+.2f}")

    # --- 事前確定の PASS 判定（docs/25 §4）---
    un, ga = res.get("value+pead_lt"), res.get("value+pead_lt|meta_gate")
    if un and ga:
        def _oos_sr(nm):
            s = v.series[nm].dropna()
            s = s[s.index >= pd.Timestamp(OOS)]
            return sharpe_ratio(s) * np.sqrt(12) if s.size >= 8 else np.nan
        oos_un, oos_ga = _oos_sr("value+pead_lt"), _oos_sr("value+pead_lt|meta_gate")
        c1 = oos_ga >= oos_un
        c2 = ga.dsr >= un.dsr
        c3 = ga.dsr >= 0.95
        print("\n--- 事前確定 PASS 判定 ---")
        print(f"  ① gated OOS SR({oos_ga:+.2f}) ≥ ungated({oos_un:+.2f}): {c1}")
        print(f"  ② gated DSR({ga.dsr:.2f}) ≥ ungated({un.dsr:.2f}): {c2}")
        print(f"  ③ gated DSR ≥ 0.95: {c3}")
        if c1 and c2 and c3:
            print("  ◎ PASS — メタゲートが多重検定後も有意に改善。")
        elif c1 and c2:
            print("  △ 改善は確認（①②成立）だが DSR0.95 未達＝動的メタ重みは効くが単独認定に不足。")
        else:
            print("  ・改善せず。docs/26 に正当な負の結果として記録（再調整しない）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
