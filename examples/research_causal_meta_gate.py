"""因果メタラベル局面ゲートの正式判定（事前登録 FROZEN）。

事前登録：docs/34-causal-meta-gate-preregistration.md
- 一次＝value+pead_lt（docs/25 と同一・無改修）
- 二次＝base_meta（凍結5特徴）と causal_meta（+因果5特徴）
- judge_grid scope=value_pead_causal_meta_gate で3戦略を同時デフレート

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"
      .venv\\Scripts\\python.exe examples\\research_causal_meta_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import events, flows  # noqa: E402
from invest_system.equities.panel import assemble_panel, fetch_month_end_snapshots  # noqa: E402
from invest_system.equities.fundamentals import load_fundamentals, point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.research import (  # noqa: E402
    AsOfView, CompositeStrategy, CrossSectionalStrategy, MetaGatedStrategy,
    backtest, fit_meta_gate, judge_grid, regime_breakdown, write_html,
)
from invest_system.research.causal_sector.meta import build_causal_meta_features  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
WARMUP, EMBARGO = 36, 1
SCOPE = "value_pead_causal_meta_gate"


def _trailing_sharpe(net: pd.Series, window: int = 12) -> pd.Series:
    def _sr(x):
        sd = np.std(x, ddof=1)
        return float(np.mean(x) / sd) if sd > 0 else np.nan
    return net.rolling(window).apply(_sr, raw=True).shift(1)


def build_base_features(rebal, adj, umask, value_net) -> pd.DataFrame:
    """凍結5特徴量（docs/25 §3.2）。"""
    feat = pd.DataFrame(index=rebal)
    try:
        topix = jq.fetch_index_bars(code="0000")
        tcl = topix.dropna(subset=["Date"]).set_index("Date")["C"].sort_index()
        tm = tcl.reindex(rebal, method="ffill")
        tret = tm.pct_change()
        feat["topix_vol"] = tret.rolling(6).std()
        feat["topix_mom"] = tm / tm.shift(6) - 1.0
    except Exception as e:  # noqa: BLE001
        print(f"WARN: TOPIX: {e}")
        feat["topix_vol"] = np.nan
        feat["topix_mom"] = np.nan
    feat["value_trail"] = _trailing_sharpe(value_net, 12).reindex(rebal)
    r1 = adj.pct_change()
    feat["dispersion"] = apply_universe_mask(r1, umask).std(axis=1).reindex(rebal)
    inv = flows.load_investor_types()
    fi = flows.net_flow_intensity(inv, investor="foreign", section="TokyoNagoya")
    if fi.empty:
        print("WARN: flow_intensity NaN — fetch_indices_flows.py を先に実行")
        feat["flow_intensity"] = np.nan
    else:
        feat["flow_intensity"] = fi.sort_index().reindex(rebal, method="ffill")
    return feat


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1

    print("=" * 72)
    print(f"因果メタゲート正式判定（scope={SCOPE}・FROZEN docs/34）")
    print("=" * 72)

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
        return cross_sectional_zscore(
            sector_neutralize(apply_universe_mask(f, umask), sector))

    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    value = zN(value_quality_size_factors(pit, raw, adj)["book_to_market"])
    pead = zN(point_in_time(
        events.forecast_revision(fund), rebal, ["fcst_revision"],
        date_col="DiscDate", lag_days=1)["fcst_revision"].reindex(columns=superset))

    value_ls = CrossSectionalStrategy(value, 0.2, name="value")
    pead_lt = CrossSectionalStrategy(pead, 0.2, name="pead_longtilt", long_only=True)
    combo = CompositeStrategy([value_ls, pead_lt], [0.5, 0.5], name="value+pead_lt")

    combo_net = backtest(combo, view, costs_bps=15.0, adv=adv).returns.dropna()
    value_net = backtest(value_ls, view, costs_bps=15.0, adv=adv).returns.dropna()

    base_feat = build_base_features(rebal, adj, umask, value_net).reindex(combo_net.index)
    print("因果特徴量構築（代表6業種）…")
    causal_feat = build_causal_meta_features(combo_net.index)
    full_feat = base_feat.join(causal_feat, how="left")

    cov_b = base_feat.notna().all(axis=1).sum()
    cov_f = full_feat.notna().all(axis=1).sum()
    print(f"特徴カバレッジ: base={cov_b}/{len(combo_net)}  causal_full={cov_f}/{len(combo_net)} 月")

    # --- 診断：因果レジーム分離（ungated）---
    edge = causal_feat["causal_edge"].reindex(combo_net.index)
    causal_regime = (edge.fillna(0) > 0).astype(float)
    stat_regime = (base_feat["topix_mom"] > 0).astype(float)
    print("\n--- 診断: ungated の統計的レジーム（TOPIX mom）---")
    print(regime_breakdown(combo_net, stat_regime, ann=12).to_string(index=False))
    print("\n--- 診断: ungated の因果的レジーム（edge符号）---")
    print(regime_breakdown(combo_net, causal_regime, ann=12).to_string(index=False))

    # --- メタゲート ---
    scale_base = fit_meta_gate(combo_net, base_feat, warmup=WARMUP, embargo=EMBARGO)
    scale_causal = fit_meta_gate(combo_net, full_feat, warmup=WARMUP, embargo=EMBARGO)
    print(f"\nメタゲート有効月: base={scale_base.notna().sum()}  "
          f"causal={scale_causal.notna().sum()}/{len(combo_net)}")

    gated_base = MetaGatedStrategy(combo, scale_base, name="value+pead_lt|base_meta")
    gated_causal = MetaGatedStrategy(combo, scale_causal, name="value+pead_lt|causal_meta")

    # --- 正式判定 ---
    with default_registry() as reg:
        v = judge_grid(
            [combo, gated_base, gated_causal], view,
            scope=SCOPE,
            hypothesis="value+PEADのOOS失速は因果構造状態に依存し、"
                       "セクター因果エッジ特徴量を加えたメタモデルで改善する",
            economic_rationale="VALUE→RETエッジのstructural breakはパフォーマンスブレイクに"
                               "先行しうる。不利因果レジームでメタゲートが露出を縮小すれば"
                               "value+PEAD脚衝突の負けを削れる（LdP因果的FI原則）",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1,
        )
    print("\n" + v.report_md)
    html_path = write_html(v, f"data/reports/{v.scope}.html")
    print("HTML:", html_path)

    # --- IS/OOS ---
    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    res = {r.name: r for r in v.results}
    for name in ("value+pead_lt", "value+pead_lt|base_meta", "value+pead_lt|causal_meta"):
        r = res.get(name)
        if r is None:
            continue
        ls = v.series.get(name, pd.Series(dtype="float64")).dropna()
        oos = ls[ls.index >= pd.Timestamp(OOS)]
        is_ = ls[ls.index < pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {name:<32} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={si:+.2f} OOS={so:+.2f}")

    # --- PASS 判定（docs/34 §4）---
    un = res.get("value+pead_lt")
    ca = res.get("value+pead_lt|causal_meta")
    ba = res.get("value+pead_lt|base_meta")
    if un and ca:
        def _oos_sr(nm):
            s = v.series[nm].dropna()
            s = s[s.index >= pd.Timestamp(OOS)]
            return sharpe_ratio(s) * np.sqrt(12) if s.size >= 8 else np.nan

        oos_un, oos_ca = _oos_sr("value+pead_lt"), _oos_sr("value+pead_lt|causal_meta")
        oos_ba = _oos_sr("value+pead_lt|base_meta") if ba else np.nan
        c1 = oos_ca >= oos_un
        c2 = ca.dsr >= un.dsr
        c3 = ca.dsr >= 0.95
        print("\n--- 事前確定 PASS 判定（docs/34 §4）---")
        print(f"  ① causal OOS SR({oos_ca:+.2f}) ≥ ungated({oos_un:+.2f}): {c1}")
        print(f"  ② causal DSR({ca.dsr:.2f}) ≥ ungated({un.dsr:.2f}): {c2}")
        print(f"  ③ causal DSR ≥ 0.95: {c3}")
        if ba:
            print(f"  [副次] causal OOS SR ≥ base_meta({oos_ba:+.2f}): {oos_ca >= oos_ba}")
        if c1 and c2 and c3:
            print("  ◎ PASS — 因果メタゲートが多重検定後も有意に改善。")
        elif c1 and c2:
            print("  △ 改善確認（①②）だが DSR0.95 未達。")
        else:
            print("  ・PASS 未達。docs/35 に記録（再調整しない）。")

    # 結果保存
    out = ROOT / "docs" / "35-causal-meta-gate-results.md"
    lines = [
        "# 35 — 因果メタゲート正式判定結果",
        "",
        f"> scope=`{SCOPE}`  事前登録: docs/34",
        "",
        f"**K={v.k}**  DSR閾値={v.dsr_threshold}",
        "",
        v.report_md,
        "",
        f"HTML: `{html_path}`",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n結果保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())