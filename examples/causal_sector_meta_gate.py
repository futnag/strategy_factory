"""因果特徴量を value+PEAD メタラベルに統合し、ベースラインと比較。

既存 research_value_pead_meta_gate.py の凍結5特徴量に、
セクター因果エッジ強度・安定性・構造ブレイク近接度を追加。
CPCV + DSR + regime 別パフォーマンスを報告。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"
      .venv\\Scripts\\python.exe examples\\causal_sector_meta_gate.py
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
from invest_system.equities import events  # noqa: E402
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
    backtest, fit_meta_gate, regime_breakdown,
)
from invest_system.research.causal_sector.meta import build_causal_meta_features  # noqa: E402
from invest_system.research.causal_sector.validate import (  # noqa: E402
    causal_regime_cv_report, compare_gated_vs_ungated,
)
from invest_system.validation.dsr import (  # noqa: E402
    deflated_sharpe_ratio, sharpe_ratio,
)

START, END, OOS = "2016-07", "2026-05", "2024-01"
WARMUP, EMBARGO = 36, 1
REPORT_DIR = ROOT / "data" / "reports" / "causal_sector"


def _trailing_sharpe(net: pd.Series, window: int = 12) -> pd.Series:
    def _sr(x):
        sd = np.std(x, ddof=1)
        return float(np.mean(x) / sd) if sd > 0 else np.nan
    return net.rolling(window).apply(_sr, raw=True).shift(1)


def build_base_features(rebal, adj, umask, value_net) -> pd.DataFrame:
    """凍結5特徴量（docs/25 準拠）。"""
    feat = pd.DataFrame(index=rebal)
    try:
        topix = jq.fetch_index_bars(code="0000")
        tcl = topix.dropna(subset=["Date"]).set_index("Date")["C"].sort_index()
        tm = tcl.reindex(rebal, method="ffill")
        tret = tm.pct_change()
        feat["topix_vol"] = tret.rolling(6).std()
        feat["topix_mom"] = tm / tm.shift(6) - 1.0
    except Exception:  # noqa: BLE001
        feat["topix_vol"] = np.nan
        feat["topix_mom"] = np.nan
    feat["value_trail"] = _trailing_sharpe(value_net, 12).reindex(rebal)
    r1 = adj.pct_change()
    feat["dispersion"] = apply_universe_mask(r1, umask).std(axis=1).reindex(rebal)
    from invest_system.equities import flows  # noqa: E402
    inv = flows.load_investor_types()
    fi = flows.net_flow_intensity(inv, investor="foreign", section="TokyoNagoya")
    feat["flow_intensity"] = (
        fi.sort_index().reindex(rebal, method="ffill") if not fi.empty else np.nan
    )
    return feat


def _ann_sr(net: pd.Series) -> float:
    s = net.dropna()
    if len(s) < 6:
        return np.nan
    return float(sharpe_ratio(s) * np.sqrt(12))


def _dsr(net: pd.Series, n_trials: int = 2) -> float:
    s = net.dropna()
    if len(s) < 6:
        return np.nan
    try:
        return deflated_sharpe_ratio(
            sharpe_ratio(s), 0.0, n_trials, len(s), 0.0, 3.0,
        )
    except Exception:  # noqa: BLE001
        return np.nan


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: J_QUANTS_API_KEY required")
        return 1

    print("=" * 72)
    print("因果特徴量 × value+PEAD メタラベル統合")
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
    print("因果特徴量を構築中（代表6業種）…")
    causal_feat = build_causal_meta_features(combo_net.index)
    print(f"  base: {list(base_feat.columns)}")
    print(f"  causal: {list(causal_feat.columns)}")
    cov_base = base_feat.notna().all(axis=1).sum()
    cov_full = base_feat.join(causal_feat).notna().all(axis=1).sum()
    print(f"  特徴カバレッジ: base={cov_base}/{len(combo_net)}  "
          f"full={cov_full}/{len(combo_net)} 月")

    # --- メタゲート比較（base vs base+causal）---
    cmp = compare_gated_vs_ungated(combo_net, base_feat, causal_feat, warmup=WARMUP)
    print("\n--- メタゲート SR/DSR 比較（walk-forward）---")
    print(cmp.to_string(index=False))

    # --- 3モデル: ungated / base_meta / causal_meta ---
    full_feat = base_feat.join(causal_feat, how="left")
    scale_base = fit_meta_gate(combo_net, base_feat, warmup=WARMUP, embargo=EMBARGO)
    scale_causal = fit_meta_gate(combo_net, full_feat, warmup=WARMUP, embargo=EMBARGO)

    gated_base = MetaGatedStrategy(combo, scale_base, name="value+pead_lt|base_meta")
    gated_causal = MetaGatedStrategy(combo, scale_causal, name="value+pead_lt|causal_meta")

    net_ungated = combo_net
    net_base = backtest(gated_base, view, costs_bps=15.0, adv=adv).returns.dropna()
    net_causal = backtest(gated_causal, view, costs_bps=15.0, adv=adv).returns.dropna()

    summary = []
    for label, net in [
        ("ungated", net_ungated),
        ("base_meta", net_base),
        ("causal_meta", net_causal),
    ]:
        oos = net[net.index >= pd.Timestamp(OOS)]
        summary.append({
            "model": label,
            "sr_ann": _ann_sr(net),
            "dsr": _dsr(net),
            "oos_sr_ann": _ann_sr(oos),
            "oos_dsr": _dsr(oos),
            "n_months": len(net),
        })
    summary_df = pd.DataFrame(summary)
    print("\n--- 全期間 / OOS 比較 ---")
    print(summary_df.to_string(index=False))

    # --- CPCV ---
    cpcv_base = causal_regime_cv_report(combo_net, base_feat)
    cpcv_causal = causal_regime_cv_report(combo_net, full_feat)
    print("\n--- CPCV Sharpe 分布 ---")
    print(f"  base_meta  mean_sr={cpcv_base.get('mean_sr')}  dsr={cpcv_base.get('dsr')}")
    print(f"  causal_meta mean_sr={cpcv_causal.get('mean_sr')}  dsr={cpcv_causal.get('dsr')}")

    # --- レジーム別: 統計的(topix_mom) vs 因果的(causal_edge) ---
    stat_regime = (base_feat["topix_mom"] > 0).astype(float)  # 1=bull, 0=bear
    edge = causal_feat["causal_edge"].reindex(combo_net.index)
    causal_regime = (edge.fillna(0) > 0).astype(float)  # 1=edge+, 0=edge weak
    print("\n--- レジーム別 SR（ungated）: 統計的 TOPIX mom (1=bull) ---")
    print(regime_breakdown(net_ungated, stat_regime, ann=12).to_string(index=False))
    print("\n--- レジーム別 SR（causal_meta）: 因果エッジ符号 (1=positive) ---")
    print(regime_breakdown(net_causal, causal_regime, ann=12).to_string(index=False))

    # --- PASS 判定（因果版）---
    sm = summary_df.set_index("model")
    u, b, c = sm.loc["ungated"], sm.loc["base_meta"], sm.loc["causal_meta"]
    c1 = c["oos_sr_ann"] >= u["oos_sr_ann"]
    c2 = c["oos_sr_ann"] >= b["oos_sr_ann"]
    c3 = c["dsr"] >= b["dsr"]
    c4 = c["oos_dsr"] >= 0.95
    print("\n--- 因果メタゲート判定 ---")
    print(f"  ① causal OOS SR({c['oos_sr_ann']:+.2f}) ≥ ungated({u['oos_sr_ann']:+.2f}): {c1}")
    print(f"  ② causal OOS SR ≥ base_meta({b['oos_sr_ann']:+.2f}): {c2}")
    print(f"  ③ causal DSR({c['dsr']:.2f}) ≥ base({b['dsr']:.2f}): {c3}")
    print(f"  ④ causal OOS DSR({c['oos_dsr']:.2f}) ≥ 0.95: {c4}")
    if c1 and c2 and c3 and c4:
        print("  ◎ PASS — 因果メタゲートが OOS で有意に改善。")

    # --- 保存 ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cmp.to_csv(REPORT_DIR / "meta_gate_comparison.csv", index=False)
    summary_df.to_csv(REPORT_DIR / "meta_gate_summary.csv", index=False)
    pd.DataFrame({
        "cpcv_base_sharpes": [cpcv_base.get("cpcv_sharpes")],
        "cpcv_causal_sharpes": [cpcv_causal.get("cpcv_sharpes")],
    }).to_json(REPORT_DIR / "meta_gate_cpcv.json")

    md_lines = [
        "# 因果メタゲート統合結果",
        "",
        "## SR/DSR 比較（walk-forward）",
        cmp.to_string(index=False),
        "",
        "## 全期間 / OOS",
        summary_df.to_string(index=False),
        "",
        "## 判定",
        f"- causal OOS SR ≥ ungated: {c1}",
        f"- causal OOS SR ≥ base_meta: {c2}",
        f"- causal DSR ≥ base_meta: {c3}",
        f"- causal OOS DSR ≥ 0.95: {c4}",
    ]
    (REPORT_DIR / "meta_gate_results.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n保存: {REPORT_DIR}/meta_gate_*.csv, meta_gate_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())