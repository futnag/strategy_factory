"""検証スライス：EDINET有報 type=5 のパース値を J-Quants fins_summary と突合（throwaway）。

会計基準を跨ぐ代表銘柄で「要素ID→正準フィールド」マッピングの妥当性を確認する。
判定・レジストリは使わない（K 不変）。相対差 >1% を WARN として表示。
手元実行: .venv/Scripts/python.exe examples/edinet_validate_slice.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import edinet, edinet_taxonomy as tax  # noqa: E402
from invest_system.equities import fundamentals as fu  # noqa: E402

# secCode 上 4 桁 -> ラベル（会計基準の見当。IFRS 多数＋US-GAAP=キヤノン/トヨタ＋JGAAP=任天堂/キーエンス）
TARGETS = {
    "7203": "トヨタ", "8058": "三菱商事", "4502": "武田", "9984": "SBG",
    "6758": "ソニー", "6501": "日立", "7751": "キヤノン", "2502": "アサヒ",
    "7974": "任天堂", "6861": "キーエンス",
}
# 突合フィールド（正準 -> J-Quants fins_summary 列）。
# 注: J-Quants `Eq` は純資産（非支配持分込み total equity）＝正準 net_assets に対応
# （親会社株主持分 equity は NCI の分だけ小さくなる）。
PAIRS = [("net_sales", "Sales"), ("operating_income", "OP"), ("ordinary_income", "OdP"),
         ("profit", "NP"), ("total_assets", "TA"), ("net_assets", "Eq"),
         ("cfo", "CFO"), ("cfi", "CFI"), ("cff", "CFF"), ("cash", "CashEq"),
         ("shares_outstanding", "ShOutFY")]
TOL = 0.01


def latest_annual_docs() -> pd.DataFrame:
    """ローカル by-date ミラーから対象社の最新有報（csvFlag=1）を 1 件ずつ。"""
    rows = []
    for f in sorted(Path("data/edinet/list").glob("*.parquet")):
        df = pd.read_parquet(f)
        if "_empty" in df.columns or "docTypeCode" not in df.columns:
            continue
        a = edinet.annual_report_documents(df)
        if len(a):
            rows.append(a)
    ann = pd.concat(rows, ignore_index=True)
    ann["s4"] = ann["secCode"].astype(str).str[:4]
    ann = ann[ann["s4"].isin(TARGETS) & (ann["csvFlag"].astype(str) == "1")]
    ann = ann.dropna(subset=["periodEnd"])      # periodEnd 欠損の一覧行を除外
    return ann.sort_values("periodEnd").groupby("s4", as_index=False).tail(1)


def jq_fy_row(jq: pd.DataFrame, code4: str, period_end: str):
    pe = pd.to_datetime(period_end).strftime("%Y-%m-%d")
    sub = jq[(jq["Code"].astype(str).str[:4] == code4) & (jq["CurPerType"] == "FY")].copy()
    sub = sub[sub["CurFYEn"].astype(str).str[:10] == pe]
    return sub.sort_values("DiscDate").iloc[-1] if len(sub) else None


def main() -> None:
    jq = fu.load_fundamentals()
    docs = latest_annual_docs()
    print(f"検証対象: {len(docs)} 社（最新有報×1 を J-Quants FY 開示と突合）\n")
    n_ok = n_warn = n_miss = 0
    for _, d in docs.iterrows():
        code4, pe = d["s4"], pd.to_datetime(d["periodEnd"]).strftime("%Y-%m-%d")
        zip_path = edinet.fetch_document(d["docID"], doc_type=5)
        canon = tax.extract_canonical(edinet.read_xbrl_csv(zip_path))
        jrow = jq_fy_row(jq, code4, pe)
        jbasis = jrow["DocType"] if jrow is not None else "NA"
        print(f"=== {TARGETS[code4]}({code4})  FY末 {pe}  "
              f"基準[EDINET]={canon['basis']}  [JQ]={jbasis} ===")
        if jrow is None:
            print("  J-Quants に一致する FY 開示なし\n")
            continue
        for cf, jf in PAIRS:
            ev = canon.get(cf)
            jv = pd.to_numeric(jrow.get(jf), errors="coerce")
            if ev is None and pd.isna(jv):
                continue
            if ev is None:
                n_miss += 1
                print(f"  [MISS] {cf:<18} EDINET=---                JQ={jv:>20,.0f}")
                continue
            if pd.isna(jv):
                print(f"  [JQ--] {cf:<18} EDINET={ev:>20,.0f}  JQ=---")
                continue
            rel = abs(ev - jv) / max(abs(jv), 1.0)
            ok = rel <= TOL
            n_ok += ok
            n_warn += (not ok)
            print(f"  [{'OK ' if ok else 'WARN'}] {cf:<18} "
                  f"EDINET={ev:>20,.0f}  JQ={jv:>20,.0f}  rel={rel * 100:6.2f}%")
        print()
    print(f"突合サマリ: OK={n_ok}  WARN={n_warn}  MISS(EDINET未取得)={n_miss}")
    print("※ throwaway 診断（K 不変・判定ではない）。WARN は基準差・定義差・マッピング欠落の候補。")


if __name__ == "__main__":
    main()
