"""EDINET 書類一覧 API の実地 PoC（docs/08 §7 の確認 5 点を実測）。

直近の営業日と、数年前の単発日（縦覧期間境界の探索）の一覧を取得し、以下を実測する：
  point1 様式(formCode)分布            … 府令060 の一般/特例/変更報告書の様式
  point2 csvFlag の実態                … TOB/大量保有/有報 別の CSV 提供範囲
  point3 縦覧期間境界                  … 古い日付で legalStatus=0・フィールド null 化が始まる日
  point4 変更報告書の docTypeCode      … 府令060 配下の docTypeCode 分布（350/360 以外）
  point5 大量保有 CSV のタグ位置       … --fetch-doc で CSV を1件取得し要素名を確認

jquants_markets_probe.py と同型（ライブ応答を真とする）。一覧 API は 1 日 1 リクエスト・
ページングなし。--fetch-doc 指定時のみ書類本体を 1 件だけダウンロードする。

usage:
  python examples/edinet_probe.py --days 7
  python examples/edinet_probe.py --days 5 --back-years 1,2,3,4,5,6 --fetch-doc
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import edinet as ed  # noqa: E402


def recent_bdays(n: int, end=None) -> list[str]:
    """直近 n 営業日（既定は前営業日まで）の YYYYMMDD。"""
    end = (pd.Timestamp(end) if end
           else pd.Timestamp.today().normalize() - pd.Timedelta(days=1))
    return [d.strftime("%Y%m%d") for d in pd.bdate_range(end=end, periods=n)]


def yearly_probe_dates(years: list[int]) -> list[str]:
    """today から y 年前の平日（縦覧境界探索用の単発日）。"""
    today = pd.Timestamp.today().normalize()
    out = []
    for y in years:
        d = today - pd.DateOffset(years=y)
        while d.weekday() >= 5:                 # 土日は前倒し
            d -= pd.Timedelta(days=1)
        out.append(d.strftime("%Y%m%d"))
    return out


def collect(dates: list[str], label: str) -> pd.DataFrame:
    frames = []
    for d in dates:
        try:
            df = ed.fetch_documents_list(d)
        except Exception as e:                  # noqa: BLE001
            print(f"  [warn] {label} {d}: {str(e)[:80]}")
            continue
        frames.append(df.assign(list_date=d))
        print(f"  {label} {d}: {len(df)} docs")
    return pd.concat(frames, ignore_index=True) if frames else ed.parse_documents([])


def _counts(s: pd.Series) -> dict:
    return dict(Counter(s.dropna()))


def report_recent(df: pd.DataFrame) -> None:
    print("\n=== 府令別件数（直近窓） ===")
    print("  040 第三者TOB:", int((df["ordinanceCode"] == "040").sum()),
          "/ 050 自社株TOB:", int((df["ordinanceCode"] == "050").sum()),
          "/ 060 大量保有:", int((df["ordinanceCode"] == "060").sum()),
          "/ 全:", len(df))

    print("\n=== フィールド充足率（本文を取らずに対象会社を特定できるか） ===")
    for c in ["secCode", "submitDateTime", "subjectEdinetCode",
              "issuerEdinetCode", "parentDocID", "filerName"]:
        rate = df[c].notna().mean() if len(df) else 0.0
        print(f"  {c}: {rate:.1%}")

    print("\n=== csvFlag/xbrlFlag の実態（point 2） ===")
    cats = [("TOB(府令040)", df["ordinanceCode"] == "040"),
            ("大量保有(府令060)", df["ordinanceCode"] == "060"),
            ("有報(docType120/130)", df["docTypeCode"].isin(["120", "130"]))]
    for name, mask in cats:
        sub = df[mask]
        if len(sub):
            print(f"  {name}: n={len(sub)} csvFlag={_counts(sub['csvFlag'])} "
                  f"xbrlFlag={_counts(sub['xbrlFlag'])} pdfFlag={_counts(sub['pdfFlag'])}")
        else:
            print(f"  {name}: n=0（窓内に該当なし）")

    print("\n=== 府令060（大量保有）の docTypeCode/formCode 分布（point 1・4） ===")
    sub = df[df["ordinanceCode"] == "060"]
    print("  docTypeCode:", _counts(sub["docTypeCode"]))
    print("  formCode   :", _counts(sub["formCode"]))

    print("\n=== 府令040（TOB）の docTypeCode 分布 ===")
    sub = df[df["ordinanceCode"] == "040"]
    print("  docTypeCode:", _counts(sub["docTypeCode"]))

    print("\n=== legalStatus 分布（縦覧区分・直近窓） ===")
    print("  ", _counts(df["legalStatus"]))


def report_boundary(hist: list[tuple[str, pd.DataFrame]]) -> None:
    print("\n=== 縦覧期間境界の探索（point 3：古い日付で null 化が始まるか） ===")
    print("  date      docs  filerName充足  legalStatus分布")
    for d, df in hist:
        fill = df["filerName"].notna().mean() if len(df) else float("nan")
        print(f"  {d}  {len(df):5d}    {fill:6.1%}     {_counts(df['legalStatus'])}")


def _decode_csv(raw: bytes) -> str:
    """EDINET CSV は UTF-16 TSV のことが多い。複数エンコーディングを順に試す。"""
    for enc in ("utf-16", "utf-16-le", "cp932", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except Exception:                       # noqa: BLE001
            continue
    return raw.decode("utf-8", "ignore")


def probe_large_holding_csv(df: pd.DataFrame) -> None:
    print("\n=== 大量保有 CSV の要素確認（point 5：保有割合・共同保有者・保有目的） ===")
    cand = df[(df["ordinanceCode"] == "060") & (df["csvFlag"] == "1")]
    if not len(cand):
        print("  csvFlag=1 の大量保有書類が窓内に無し（--days を増やすか別期間で再試行）")
        return
    doc_id = str(cand["docID"].iloc[0])
    print(f"  対象 docID={doc_id}（type=5 CSV を取得）")
    try:
        path = ed.fetch_document(doc_id, doc_type=5)
    except Exception as e:                       # noqa: BLE001
        print(f"  [warn] 取得失敗: {str(e)[:100]}")
        return
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        print("  ZIP 内:", names)
        for nm in names:
            if nm.lower().endswith(".csv"):
                lines = _decode_csv(z.read(nm)).splitlines()
                print(f"  --- {nm}: {len(lines)} 行（先頭 30 行）---")
                for ln in lines[:30]:
                    print("   ", ln[:160])
                break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="直近の取得営業日数")
    ap.add_argument("--back-years", type=str, default="1,2,3,4,5,6,7",
                    help="縦覧境界探索の遡及年（カンマ区切り）")
    ap.add_argument("--fetch-doc", action="store_true",
                    help="大量保有 CSV を1件取得し要素名を確認（point 5）")
    args = ap.parse_args()

    ed._api_key()                                # キー欠如は早期に明示エラー
    print(f"=== EDINET PoC: 直近 {args.days} 営業日 ===")
    rec = collect(recent_bdays(args.days), "recent")
    if len(rec):
        report_recent(rec)

    years = [int(x) for x in args.back_years.split(",") if x.strip()]
    print(f"\n=== 過去日（{years} 年前の平日・各1日） ===")
    hist = []
    for d in yearly_probe_dates(years):
        try:
            hist.append((d, ed.fetch_documents_list(d)))
            print(f"  hist {d}: {len(hist[-1][1])} docs")
        except Exception as e:                   # noqa: BLE001
            print(f"  [warn] {d}: {str(e)[:80]}")
    report_boundary(hist)

    if args.fetch_doc:
        probe_large_holding_csv(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
