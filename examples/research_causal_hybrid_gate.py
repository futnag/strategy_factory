"""OOS特化ハイブリッド因果メタゲートの正式判定（FROZEN docs/39）。

base_meta + 因果不利レジーム時のみ 0.5x 縮小。純因果版・ungated と4戦略比較。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"
      .venv\\Scripts\\python.exe examples\\research_causal_hybrid_gate.py
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
from invest_system.research.causal_sector.hybrid_gate import (  # noqa: E402
    adverse_causal_regime, fit_hybrid_causal_gate,
)
from invest_system.research.causal_sector.meta import (  # noqa: E402
    build_causal_meta_features_all33, feature_columns_all33,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
WARMUP, EMBARGO = 36, 1
SCOPE = "value_pead_causal_hybrid_gate"


def _trailing_sharpe(net: pd.Series, window: int = 12) -> pd.Series:
    def _sr(x):
        sd = np.std(x, ddof=1)
        return float(np.mean(x) / sd) if sd > 0 else np.nan
    return net.rolling(window).apply(_sr, raw=True).shift(1)


def build_base_features(rebal, adj, umask, value_net) -> pd.DataFrame:
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
    feat["flow_intensity"] = (
        fi.sort_index().reindex(rebal, method="ffill") if not fi.empty else np.nan
    )
    return feat


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: J_QUANTS_API_KEY required")
        return 1

    print("=" * 72)
    print(f"ハイブリッド因果メタゲート正式判定（scope={SCOPE}・docs/39）")
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

    print("全33業種因果特徴量（キャッシュ）…")
    causal_all33 = build_causal_meta_features_all33(combo_net.index, verbose=False)
    feat_all33 = base_feat.join(causal_all33[feature_columns_all33()], how="left")

    adverse = adverse_causal_regime(causal_all33.reindex(combo_net.index))
    n_adverse = int(adverse.sum())
    print(f"  adverse 月数: {n_adverse}/{len(combo_net)} "
          f"({100 * n_adverse / len(combo_net):.1f}%)")

    print("\n--- 診断: ungated × adverse 因果レジーム ---")
    print(regime_breakdown(combo_net, adverse.astype(float), ann=12).to_string(index=False))

    scale_base = fit_meta_gate(combo_net, base_feat, warmup=WARMUP, embargo=EMBARGO)
    scale_causal = fit_meta_gate(combo_net, feat_all33, warmup=WARMUP, embargo=EMBARGO)
    scale_hybrid, adverse_applied = fit_hybrid_causal_gate(
        combo_net, base_feat, causal_all33, warmup=WARMUP, embargo=EMBARGO,
    )
    n_shrink = int((adverse_applied & scale_base.notna()).sum())
    print(f"\nハイブリッド縮小適用月: {n_shrink}（base有効かつadverse）")

    gated_base = MetaGatedStrategy(combo, scale_base, name="value+pead_lt|base_meta")
    gated_causal = MetaGatedStrategy(combo, scale_causal, name="value+pead_lt|causal_all33")
    gated_hybrid = MetaGatedStrategy(combo, scale_hybrid, name="value+pead_lt|hybrid_all33")

    with default_registry() as reg:
        v = judge_grid(
            [combo, gated_base, gated_causal, gated_hybrid], view,
            scope=SCOPE,
            hypothesis="base_metaを維持し因果は不利レジーム時のみ0.5x縮小すれば"
                       "OOS改善と全期間DSRを両立する",
            economic_rationale="純因果MLゲートはISで過剰縮小。三重不利条件ANDのみ"
                               "downside保護に因果を使えばmirageと過学習を避けつつ"
                               "value+PEADの脚衝突損失を削れる",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1,
        )
    print("\n" + v.report_md)
    html_path = write_html(v, f"data/reports/{v.scope}.html")
    print("HTML:", html_path)

    res = {r.name: r for r in v.results}
    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    names = (
        "value+pead_lt", "value+pead_lt|base_meta",
        "value+pead_lt|causal_all33", "value+pead_lt|hybrid_all33",
    )
    for name in names:
        r = res.get(name)
        if r is None:
            continue
        ls = v.series.get(name, pd.Series(dtype="float64")).dropna()
        oos = ls[ls.index >= pd.Timestamp(OOS)]
        is_ = ls[ls.index < pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        print(f"  {name:<36} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={si:+.2f} OOS={so:+.2f}")

    un = res.get("value+pead_lt")
    ca = res.get("value+pead_lt|causal_all33")
    hy = res.get("value+pead_lt|hybrid_all33")
    if un and hy:
        def _oos_sr(nm):
            s = v.series[nm].dropna()
            s = s[s.index >= pd.Timestamp(OOS)]
            return sharpe_ratio(s) * np.sqrt(12) if s.size >= 8 else np.nan

        oos_un, oos_hy, oos_ca = (
            _oos_sr("value+pead_lt"),
            _oos_sr("value+pead_lt|hybrid_all33"),
            _oos_sr("value+pead_lt|causal_all33") if ca else np.nan,
        )
        c1, c2, c3 = oos_hy >= oos_un, hy.dsr >= un.dsr, hy.dsr >= 0.95
        print("\n--- PASS 判定（docs/39 §4）---")
        print(f"  ① hybrid OOS SR({oos_hy:+.2f}) ≥ ungated({oos_un:+.2f}): {c1}")
        print(f"  ② hybrid DSR({hy.dsr:.2f}) ≥ ungated({un.dsr:.2f}): {c2}")
        print(f"  ③ hybrid DSR ≥ 0.95: {c3}")
        if ca:
            print(f"  [副次] hybrid 全SR({hy.sr_ann:+.2f}) ≥ causal({ca.sr_ann:+.2f}): "
                  f"{hy.sr_ann >= ca.sr_ann}")
            print(f"  [副次] hybrid OOS SR ≥ causal({oos_ca:+.2f}): {oos_hy >= oos_ca}")
        if c1 and c2 and c3:
            print("  ◎ PASS")
        elif c1 and c2:
            print("  △ ①②成立・DSR0.95未達")
        else:
            print("  ・PASS未達 → docs/40 に記録")

    out = ROOT / "docs" / "40-causal-hybrid-gate-results.md"
    out.write_text(
        "\n".join([
            "# 40 — ハイブリッド因果メタゲート判定結果",
            "",
            f"> scope=`{SCOPE}`  事前登録: docs/39",
            f"> adverse月: {n_adverse}  縮小適用: {n_shrink}",
            "",
            f"**K={v.k}**",
            "",
            v.report_md,
            "",
            f"HTML: `{html_path}`",
        ]),
        encoding="utf-8",
    )
    print(f"\n結果保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())