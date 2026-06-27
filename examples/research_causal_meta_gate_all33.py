"""全33業種因果メタゲートの正式判定（事前登録 FROZEN docs/37）。

代表6業種版（docs/34）と全33業種版を同一 judge_grid で比較。
scope=value_pead_causal_meta_gate_all33

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"
      .venv\\Scripts\\python.exe examples\\research_causal_meta_gate_all33.py
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
from invest_system.research.causal_sector.config import REPRESENTATIVE_SECTORS  # noqa: E402
from invest_system.research.causal_sector.meta import (  # noqa: E402
    build_causal_meta_features, build_causal_meta_features_all33,
    feature_columns_all33,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START, END, OOS = "2016-07", "2026-05", "2024-01"
WARMUP, EMBARGO = 36, 1
SCOPE = "value_pead_causal_meta_gate_all33"


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
    print(f"全33業種因果メタゲート正式判定（scope={SCOPE}・docs/37）")
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

    print("\n[1/2] 代表6業種因果特徴量…")
    causal_rep6 = build_causal_meta_features(
        combo_net.index, sectors=list(REPRESENTATIVE_SECTORS))
    print("[2/2] 全33業種因果特徴量…")
    causal_all33 = build_causal_meta_features_all33(combo_net.index, verbose=True)

    feat_rep6 = base_feat.join(causal_rep6, how="left")
    feat_all33 = base_feat.join(causal_all33[feature_columns_all33()], how="left")

    cov_r6 = feat_rep6.notna().all(axis=1).sum()
    cov_a33 = feat_all33.notna().all(axis=1).sum()
    print(f"\n特徴カバレッジ: rep6={cov_r6}/{len(combo_net)}  "
          f"all33={cov_a33}/{len(combo_net)} 月")
    print(f"  all33 列: {feature_columns_all33()}")

    # 診断：kind エッジと ungated SR
    if "edge_it_comm" in causal_all33.columns:
        it_pos = (causal_all33["edge_it_comm"].fillna(0) > 0).astype(float)
        fin_pos = (causal_all33["edge_financial"].fillna(0) > 0).astype(float)
        print("\n--- 診断: ungated × IT因果エッジ符号 ---")
        print(regime_breakdown(combo_net, it_pos, ann=12).to_string(index=False))
        print("\n--- 診断: ungated × 金融因果エッジ符号 ---")
        print(regime_breakdown(combo_net, fin_pos, ann=12).to_string(index=False))

    scale_rep6 = fit_meta_gate(combo_net, feat_rep6, warmup=WARMUP, embargo=EMBARGO)
    scale_all33 = fit_meta_gate(combo_net, feat_all33, warmup=WARMUP, embargo=EMBARGO)
    print(f"\nメタゲート有効月: rep6={scale_rep6.notna().sum()}  "
          f"all33={scale_all33.notna().sum()}/{len(combo_net)}")

    gated_rep6 = MetaGatedStrategy(combo, scale_rep6, name="value+pead_lt|causal_rep6")
    gated_all33 = MetaGatedStrategy(combo, scale_all33, name="value+pead_lt|causal_all33")

    with default_registry() as reg:
        v = judge_grid(
            [combo, gated_rep6, gated_all33], view,
            scope=SCOPE,
            hypothesis="全33業種の因果エッジ＋kind別3種をメタ特徴量に加えると"
                       "代表6業種版よりOOS/DSRが改善する",
            economic_rationale="VALUE→RETの符号は業種種別で異なる（重工業+ vs IT/金融-）。"
                               "横断33業種集約とkind明示が局面識別を精緻化し"
                               "value+PEADの不利因果レジームでの損失を削る",
            registry=reg, costs_bps=15.0, adv=adv, participation=0.1,
        )
    print("\n" + v.report_md)
    html_path = write_html(v, f"data/reports/{v.scope}.html")
    print("HTML:", html_path)

    res = {r.name: r for r in v.results}
    print(f"\n--- IS/OOS（保留 {OOS}〜）---")
    for name in ("value+pead_lt", "value+pead_lt|causal_rep6", "value+pead_lt|causal_all33"):
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
    r6 = res.get("value+pead_lt|causal_rep6")
    a33 = res.get("value+pead_lt|causal_all33")
    if un and a33:
        def _oos_sr(nm):
            s = v.series[nm].dropna()
            s = s[s.index >= pd.Timestamp(OOS)]
            return sharpe_ratio(s) * np.sqrt(12) if s.size >= 8 else np.nan

        oos_un, oos_a33 = _oos_sr("value+pead_lt"), _oos_sr("value+pead_lt|causal_all33")
        oos_r6 = _oos_sr("value+pead_lt|causal_rep6") if r6 else np.nan
        c1, c2, c3 = oos_a33 >= oos_un, a33.dsr >= un.dsr, a33.dsr >= 0.95
        print("\n--- PASS 判定（docs/37 §4）---")
        print(f"  ① all33 OOS SR({oos_a33:+.2f}) ≥ ungated({oos_un:+.2f}): {c1}")
        print(f"  ② all33 DSR({a33.dsr:.2f}) ≥ ungated({un.dsr:.2f}): {c2}")
        print(f"  ③ all33 DSR ≥ 0.95: {c3}")
        if r6:
            print(f"  [副次] all33 OOS SR ≥ rep6({oos_r6:+.2f}): {oos_a33 >= oos_r6}")
            print(f"  [副次] all33 DSR ≥ rep6({r6.dsr:.2f}): {a33.dsr >= r6.dsr}")
        if c1 and c2 and c3:
            print("  ◎ PASS")
        elif c1 and c2:
            print("  △ ①②成立・DSR0.95未達")
        else:
            print("  ・PASS未達 → docs/38 に記録")

    out = ROOT / "docs" / "38-causal-meta-gate-all33-results.md"
    out.write_text(
        "\n".join([
            "# 38 — 全33業種因果メタゲート判定結果",
            "",
            f"> scope=`{SCOPE}`  事前登録: docs/37",
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