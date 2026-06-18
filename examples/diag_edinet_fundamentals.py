"""throwaway 診断：EDINET 新特徴量の被覆率・分布・月次IC・J-Quants 突合（handoff §9）。

**これは診断であって判定ではない。** judge_grid / 永続レジストリ（K）は一切触らない＝
K 不変。新特徴量の予測力を見たいだけの使い捨て計測（IC・被覆率）で、戦略認定ではない。
バックフィル進行中はキャッシュ済み有報のぶんだけ集計され、完走に向けて数値が埋まる。

手元実行: .venv/Scripts/python.exe examples/diag_edinet_fundamentals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.equities import edinet_factors as efa  # noqa: E402
from invest_system.equities import edinet_fundamentals as ef  # noqa: E402
from invest_system.equities import fundamentals as fu  # noqa: E402
from invest_system.equities import panel as pn  # noqa: E402

# J-Quants fins_summary との突合ペア（正準 -> JQ 列）。Eq=純資産=net_assets。
PAIRS = [("net_sales", "Sales"), ("operating_income", "OP"), ("profit", "NP"),
         ("total_assets", "TA"), ("net_assets", "Eq"), ("cfo", "CFO"),
         ("cfi", "CFI"), ("cff", "CFF"), ("cash", "CashEq")]
IC_FACTORS = ["asset_growth", "net_share_issuance", "accruals", "gross_profitability",
              "ebitda_margin", "roic", "leverage", "rd_intensity"]
TOL = 0.01


def coverage_by_year_basis(long: pd.DataFrame) -> None:
    print("=== 被覆率（開示件数：年×基準）===")
    long = long.copy()
    long["yr"] = pd.to_datetime(long["period_end"]).dt.year
    print(long.pivot_table(index="yr", columns="basis", values="docID",
                           aggfunc="count", fill_value=0).to_string())
    print("\n=== フィールド充足率（非null / 全開示, %）===")
    flds = ["net_sales", "operating_income", "gross_profit", "total_assets",
            "net_assets", "cfo", "cfi", "interest_debt", "income_taxes", "rd_expense"]
    fill = (long[flds].notna().mean() * 100).round(1)
    print(fill.to_string())


def reconcile_vs_jquants(long: pd.DataFrame) -> None:
    print("\n=== official EDINET vs J-Quants 突合（|相対差|>1%=WARN）===")
    jq = fu.load_fundamentals()
    jq = jq[jq["CurPerType"] == "FY"].copy()
    jq["key"] = jq["Code"].astype(str).str[:4] + "|" + jq["CurFYEn"].astype(str).str[:10]
    jqi = jq.drop_duplicates("key").set_index("key")
    tally = {cf: [0, 0, 0] for cf, _ in PAIRS}            # [OK, WARN, MISS]
    for _, r in long.iterrows():
        key = str(r["Code"])[:4] + "|" + str(r["period_end"])[:10]
        if key not in jqi.index:
            continue
        jrow = jqi.loc[key]
        for cf, jf in PAIRS:
            ev, jv = r.get(cf), pd.to_numeric(jrow.get(jf), errors="coerce")
            if pd.isna(jv):
                continue
            if ev is None or pd.isna(ev):
                tally[cf][2] += 1
            elif abs(ev - jv) / max(abs(jv), 1.0) <= TOL:
                tally[cf][0] += 1
            else:
                tally[cf][1] += 1
    print(f"{'field':<18}{'OK':>6}{'WARN':>6}{'MISS':>6}")
    for cf, (ok, w, m) in tally.items():
        print(f"{cf:<18}{ok:>6}{w:>6}{m:>6}")


def monthly_ic() -> None:
    print("\n=== 月次クロスセクションIC（Spearman, 翌月リターン）===")
    try:
        adj = pn.load_daily_panel(field="AdjC")
    except Exception as e:  # noqa: BLE001
        print(f"  価格パネル取得不可: {str(e)[:60]} → IC スキップ")
        return
    if adj.empty:
        print("  価格パネルが空 → IC スキップ")
        return
    mclose = adj.resample("ME").last()
    fwd = pn.forward_returns(mclose)                     # 翌月リターン
    months = mclose.index[-60:]                          # 直近 5 年分の月末
    fac = efa.edinet_factor_panels(months, codes=list(adj.columns))
    print(f"{'factor':<20}{'meanIC':>9}{'IR':>7}{'n月':>6}{'平均被覆':>9}")
    for name in IC_FACTORS:
        f = fac.get(name)
        if f is None or f.dropna(how="all").empty:
            print(f"{name:<20}{'--':>9}{'--':>7}{0:>6}{0:>9}  (被覆なし)")
            continue
        ics, cov = [], []
        for t in months:
            if t not in f.index or t not in fwd.index:
                continue
            x, y = f.loc[t], fwd.loc[t]
            common = x.dropna().index.intersection(y.dropna().index)
            cov.append(len(common))
            if len(common) >= 8:
                ic = x[common].rank().corr(y[common].rank())
                if pd.notna(ic):
                    ics.append(ic)
        if ics:
            m, s = np.mean(ics), np.std(ics, ddof=1) if len(ics) > 1 else np.nan
            ir = m / s if s and not np.isnan(s) and s > 0 else np.nan
            print(f"{name:<20}{m:>9.3f}{ir:>7.2f}{len(ics):>6}{int(np.mean(cov)):>9}")
        else:
            print(f"{name:<20}{'--':>9}{'--':>7}{0:>6}{int(np.mean(cov)) if cov else 0:>9}"
                  f"  (被覆<8銘柄/月)")


def main() -> None:
    long = ef.build_edinet_long(verbose=True)
    if long.empty:
        print("EDINET 長形式が空。先に examples/download_edinet.py を実行してください。")
        return
    print(f"対象開示: {len(long)} 件 / 銘柄 {long['Code'].nunique()} / "
          f"期末 {pd.to_datetime(long['period_end']).min():%Y-%m} 〜 "
          f"{pd.to_datetime(long['period_end']).max():%Y-%m}\n")
    coverage_by_year_basis(long)
    reconcile_vs_jquants(long)
    monthly_ic()
    print("\n※ K 不変・throwaway 診断（戦略認定ではない）。被覆はバックフィル完走で埋まる。")


if __name__ == "__main__":
    main()
