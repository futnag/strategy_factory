"""Phase 4b 小型ユニバース診断（throwaway・K 不変・判定選択に不使用）。

mcap 床 ¥30B への変更で増幅するリスク（廃止バイアス・被覆・容量）を実測し、
docs/22 に併報する材料を stdout に出力する。判定は run_phase4b_judgment.py が別途一度だけ。

実行: .venv\\Scripts\\python.exe examples\\phase4b_preflight_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001  # pragma: no cover
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from invest_system.data import store  # noqa: E402
from invest_system.equities import design_matrix as dm  # noqa: E402
from invest_system.equities import edinet_factors as efa  # noqa: E402
from invest_system.equities import panel as pn  # noqa: E402
from invest_system.equities.fundamentals import fundamentals_panel  # noqa: E402

EDINET_FACTORS = ["fcf_yield", "roic", "asset_growth", "accruals", "gross_profitability",
                  "ebitda_margin", "leverage", "net_share_issuance", "rd_intensity", "cf_to_price"]


def _delist_bias(ac_m: pd.DataFrame, mask: pd.DataFrame) -> dict:
    """流動ユニバース内の等加重月次リターン：last_price vs all_minus100 の差。"""
    base = pn.forward_returns(ac_m)
    lp = pn.impute_delisting(base, ac_m, policy="last_price")
    m100 = pn.impute_delisting(base, ac_m, policy="all_minus100")
    m = mask.reindex(index=ac_m.index, columns=ac_m.columns).fillna(False)
    diffs = []
    for t in ac_m.index:
        cols = m.loc[t][m.loc[t]].index
        if len(cols) < 5:
            continue
        r_lp = lp.loc[t, cols].mean()
        r_m = m100.loc[t, cols].mean()
        if pd.notna(r_lp) and pd.notna(r_m):
            diffs.append(r_lp - r_m)
    arr = np.array(diffs)
    return {"n_months": len(arr), "mean_bias_pct": float(arr.mean() * 100) if len(arr) else float("nan"),
            "max_bias_pct": float(arr.max() * 100) if len(arr) else float("nan")}


def _factor_coverage(fp: dict, mask: pd.DataFrame, months: pd.DatetimeIndex) -> dict:
    """EDINET 因子の月次被覆（流動ユニバース内）。"""
    out = {}
    m = mask.reindex(index=months)
    for name in EDINET_FACTORS:
        w = fp.get(name, pd.DataFrame())
        if w.empty:
            out[name] = float("nan")
            continue
        sub = w.reindex(index=months).where(m)
        out[name] = float(sub.notna().sum().sum() / max(1, m.sum().sum()))
    return out


def _capacity_stats(adv_m: pd.DataFrame, mask: pd.DataFrame, participation: float = 0.05) -> dict:
    """ユニバース内 ADV の分布と participation 制約下の概算容量。"""
    sub = adv_m.where(mask)
    vals = sub.stack(future_stack=True).dropna()
    if vals.empty:
        return {}
    # デシル L/S で最薄建玉が participation×ADV に達する AUM 上限（単純近似）
    p25, p50, p75 = vals.quantile([0.25, 0.5, 0.75])
    cap_p25 = float(participation * p25 * 20)  # デシル≈20銘柄・等加重
    cap_p50 = float(participation * p50 * 20)
    return {"adv_p25_M": p25 / 1e6, "adv_p50_M": p50 / 1e6, "adv_p75_M": p75 / 1e6,
            "cap_est_p25_億": cap_p25 / 1e8, "cap_est_p50_億": cap_p50 / 1e8}


def main() -> int:
    print("=== Phase 4b 小型ユニバース診断（throwaway・K不変）===", flush=True)
    uni_prod = dm.production_universe(preset="production")
    uni_sc = dm.production_universe(preset="smallcap_30b")
    months = uni_sc.index
    added = (uni_sc & ~uni_prod.reindex_like(uni_sc).fillna(False)).sum(axis=1)
    print(f"\n[1] ユニバース規模")
    print(f"  production (¥10B): avg {int(uni_prod.sum(axis=1).mean())} 銘柄/月")
    print(f"  smallcap_30b (¥30B): avg {int(uni_sc.sum(axis=1).mean())} 銘柄/月")
    print(f"  追加銘柄（30Bのみ）: avg +{added.mean():.0f}/月, max +{int(added.max())}")

    ac_m = store.load_wide("AdjC").resample("ME").last().reindex(index=months)
    print(f"\n[2] 廃止バイアス（等加重月次リターン差 last_price−(−100%)）")
    for label, mask in [("production", uni_prod), ("smallcap_30b", uni_sc)]:
        b = _delist_bias(ac_m, mask.reindex(index=months))
        print(f"  {label}: mean +{b['mean_bias_pct']:.3f}%/月, max +{b['max_bias_pct']:.3f}%/月 "
              f"({b['n_months']} months)")

    adv_m = store.load_wide("Va").rolling(60, min_periods=20).mean().resample("ME").last(
    ).reindex(index=months)
    close_m = store.load_wide("C").resample("ME").last().reindex(index=months)
    sh = fundamentals_panel(months, ["ShOutFY", "TrShFY"])
    shares = (sh["ShOutFY"] - sh.get("TrShFY", 0.0).reindex_like(sh["ShOutFY"]).fillna(0.0))
    mcap = (shares * close_m).where(lambda x: x > 0)
    fp = efa.edinet_factor_panels(months, codes=list(close_m.columns), mcap=mcap)

    print(f"\n[3] EDINET 因子被覆（ユニバース内・非NaN率）")
    cov_prod = _factor_coverage(fp, uni_prod.reindex(index=months), months)
    cov_sc = _factor_coverage(fp, uni_sc, months)
    for name in EDINET_FACTORS:
        p, s = cov_prod.get(name, float("nan")), cov_sc.get(name, float("nan"))
        delta = (s - p) * 100 if np.isfinite(p) and np.isfinite(s) else float("nan")
        print(f"  {name:22s} prod={p:.1%}  sc={s:.1%}  Δ={delta:+.1f}pp")

    print(f"\n[4] 容量概算（participation=5%・デシルL/S近似）")
    for label, mask in [("production", uni_prod), ("smallcap_30b", uni_sc)]:
        c = _capacity_stats(adv_m, mask.reindex(index=months))
        if c:
            print(f"  {label}: ADV p25={c['adv_p25_M']:.0f}M p50={c['adv_p50_M']:.0f}M | "
                  f"cap_est p25={c['cap_est_p25_億']:.1f}億 p50={c['cap_est_p50_億']:.1f}億")

    print("\n→ 判定前診断完了。結果は run_phase4b_judgment.py の docs/22 に併記。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())