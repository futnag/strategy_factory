"""日本株クロスセクション・ファンダ・ファクター研究（pillar C・無料枠J-Quants）。

ポイントインタイム整合 → セクター/サイズ中立化（交絡制御）→ ロングショート →
試行数デフレートDSR、までを一気通貫で実行し、正直に結果を報告する。

前提: .env の J_QUANTS_API_KEY（無料枠で可）。初回はAPI取得（数分）、以降は
Parquetキャッシュで高速。市場データはコミットしない（data/ は .gitignore 済）。

実行: .venv\\Scripts\\python.exe examples\\jp_equities_factor_study.py   (Win)
      .venv/bin/python examples/jp_equities_factor_study.py            (Linux)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities.universe import select_universe  # noqa: E402
from invest_system.equities.panel import (  # noqa: E402
    assemble_panel,
    fetch_month_end_snapshots,
    forward_returns,
)
from invest_system.equities.fundamentals import point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore,
    sector_neutralize,
    value_quality_size_factors,
)
from invest_system.equities.backtest import long_short_returns  # noqa: E402
from invest_system.validation.dsr import (  # noqa: E402
    sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
)

# --- 設定（無料枠：2024-03-13〜2026-03-13 の範囲内）-----------------------
# 無料枠は 5 req/分（≈12.5秒/回）。TOP_N 銘柄分の財務取得に TOP_N×12.5秒かかる
# （初回のみ・以降キャッシュ）。Light(60/分)なら J_QUANTS_MIN_INTERVAL=1 で12倍速。
START, END = "2024-06", "2026-02"
TOP_N = int(get_env("J_EQ_TOP_N", "80") or "80")  # 既定80（無料枠で約20分）
QUANTILE = 0.2         # ロング/ショート各20%
COSTS_BPS = 15.0       # 片道15bps（東証の現実的な往復コスト目安）
LAG_DAYS = 1           # 開示当日は使わない（場中開示への保守措置）
MIN_MONTHS = 8         # Sharpe算出に要する最小月数
MIN_NAMES = 12         # 1断面で建玉に要する最小有効銘柄数（小ユニバース対応）
FIELDS = ["ShOutFY", "TrShFY", "FEPS", "Eq", "CFO", "FSales", "FDivAnn",
          "FNP", "TA", "FOP", "EqAR"]
VALUE = ["earnings_yield", "book_to_market", "cf_yield", "sales_yield", "div_yield"]
QUALITY = ["roe", "roa", "op_margin", "equity_ratio"]


def fetch_fundamentals(codes: list[str]) -> pd.DataFrame:
    """ユニバース各銘柄の財務サマリーを取得して長形式に連結（耐障害・進捗表示）。"""
    frames = []
    for i, code in enumerate(codes, 1):
        try:
            st = jq.fetch_statements(code=code)
            if not st.empty:
                frames.append(st)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {code}: {e}")
        if i % 50 == 0:
            print(f"  fundamentals: {i}/{len(codes)} 取得")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def evaluate(name: str, factor: pd.DataFrame, fwd: pd.DataFrame,
             sector: pd.Series) -> tuple[pd.Series, dict]:
    """中立化→z化→ロングショート→指標。返り値: (月次LS系列, 指標dict)。"""
    z = cross_sectional_zscore(sector_neutralize(factor, sector))
    ls = long_short_returns(z, fwd, quantile=QUANTILE, costs_bps=COSTS_BPS,
                            min_names=MIN_NAMES).dropna()
    stats = {"name": name, "n": int(ls.size)}
    if ls.size >= MIN_MONTHS and ls.std(ddof=1) > 0:
        spp = sharpe_ratio(ls)
        stats.update(mean_m=float(ls.mean()), sharpe_pp=spp,
                     sharpe_ann=spp * np.sqrt(12))
    else:
        stats.update(mean_m=np.nan, sharpe_pp=np.nan, sharpe_ann=np.nan)
    return ls, stats


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY を設定してください。")
        return 1

    print(f"=== 日本株ファクター研究  期間 {START}〜{END}  上位{TOP_N}銘柄 ===")
    listed = jq.fetch_listed_info()
    print(f"上場銘柄マスタ: {listed.shape[0]} 件")

    print("月末スナップショット取得中…")
    snaps = fetch_month_end_snapshots(START, END)
    adj = assemble_panel(snaps, "AdjC")
    raw = assemble_panel(snaps, "C")
    turn = assemble_panel(snaps, "Va")
    print(f"パネル: {adj.shape[0]} か月 × {adj.shape[1]} 銘柄")

    universe = select_universe(listed, turn, top_n=TOP_N, min_obs=MIN_MONTHS)
    print(f"ユニバース（流動性上位・普通株）: {len(universe)} 銘柄")
    adj, raw = adj.reindex(columns=universe), raw.reindex(columns=universe)

    print("財務サマリー取得中（初回は数分、以降キャッシュ）…")
    fund = fetch_fundamentals(universe)
    print(f"財務: {fund.shape[0]} 開示行")

    pit = point_in_time(fund, adj.index, FIELDS, lag_days=LAG_DAYS)
    factors = value_quality_size_factors(pit, raw, adj)
    fwd = forward_returns(adj)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"] \
        if "S33" in listed.columns else pd.Series(dtype=object)

    # --- 各単一ファクターを評価 ---
    series: dict[str, pd.Series] = {}
    stats: list[dict] = []
    for name, fac in factors.items():
        ls, st = evaluate(name, fac, fwd, sector)
        series[name] = ls
        stats.append(st)

    # --- 合成（バリュー+クオリティ+サイズの中立化zの平均）---
    comp_terms = []
    for name in VALUE + QUALITY + ["size"]:
        if name in factors:
            comp_terms.append(cross_sectional_zscore(
                sector_neutralize(factors[name], sector)))
    composite = sum(comp_terms) / len(comp_terms)
    comp_ls = long_short_returns(composite, fwd, quantile=QUANTILE,
                                 costs_bps=COSTS_BPS, min_names=MIN_NAMES).dropna()
    series["COMPOSITE"] = comp_ls
    if comp_ls.size >= MIN_MONTHS and comp_ls.std(ddof=1) > 0:
        spp = sharpe_ratio(comp_ls)
        stats.append({"name": "COMPOSITE", "n": int(comp_ls.size),
                      "mean_m": float(comp_ls.mean()), "sharpe_pp": spp,
                      "sharpe_ann": spp * np.sqrt(12)})

    # --- 試行数デフレートDSR ---
    valid = [s for s in stats if not np.isnan(s["sharpe_pp"])]
    sr_pp = np.array([s["sharpe_pp"] for s in valid])
    sr_var = float(np.var(sr_pp, ddof=1)) if len(sr_pp) > 1 else 0.0
    n_trials = len(valid)
    for s in valid:
        ls = series[s["name"]]
        try:
            s["dsr"] = deflated_sharpe_ratio_from_returns(ls.values, sr_var, n_trials)
        except Exception:  # noqa: BLE001
            s["dsr"] = np.nan

    # --- 報告 ---
    valid.sort(key=lambda s: (s["sharpe_ann"] if not np.isnan(s["sharpe_ann"])
                              else -9), reverse=True)
    print("\n=== 結果（セクター中立・コスト{:.0f}bps・LS{:.0%}）==="
          .format(COSTS_BPS, QUANTILE))
    print(f"試行数 n_trials={n_trials}, 試行間SR分散 V[SR]={sr_var:.4f}")
    print(f"{'factor':<16}{'n':>4}{'mean/m':>9}{'SR(ann)':>9}{'DSR':>8}")
    print("-" * 46)
    for s in valid:
        print(f"{s['name']:<16}{s['n']:>4}{s['mean_m']:>9.4f}"
              f"{s['sharpe_ann']:>9.2f}{s['dsr']:>8.2f}")

    survivors = [s for s in valid if not np.isnan(s["dsr"]) and s["dsr"] >= 0.95]
    print("\n--- 判定（DSR≥0.95＝多重検定後も有意）---")
    if survivors:
        for s in survivors:
            print(f"  ★ {s['name']}: SR(ann)={s['sharpe_ann']:.2f}, DSR={s['dsr']:.3f}")
    else:
        best = valid[0] if valid else None
        print("  生存ファクター無し（短い2年標本＋多重検定の壁）。")
        if best:
            print(f"  最良は {best['name']} (SR(ann)={best['sharpe_ann']:.2f}, "
                  f"DSR={best['dsr']:.2f}) だが基準未達。")
        print("  ※ これは健全な結果：偽の α を作らない。履歴延長(Light/5年)や")
        print("    因果フィルタ・メタラベルでの絞り込みが次の一手。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
