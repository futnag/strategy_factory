"""EDINET 有報三表を「提出日アンカーの as-of パネル」に統合（GKX Phase 2・#6）。

ローカルにバックフィルした有報 type=5（examples/download_edinet.py）を 1 件ずつパースし、
正準フィールド（生ライン項目）の長形式を作る。提出日（submitDateTime）を PIT アンカーに、
既存の fundamentals.point_in_time（DiscDate≤t−lag の最新開示のみ採用・FYE 非依存）へ
そのまま流して as-of パネルにする＝J-Quants fins_summary と同一の PIT 機構を再利用する。

規律（handoff §2,§5）:
- 提出日アンカー厳守（submitDateTime を DiscDate に）。決算期末・報告義務日は使わない。
- 会計基準タグ（IFRS/JGAAP/USGAAP）を basis 列で保持。基準別 null は呼び出し側で扱う。
- 会計基準移行（JGAAP→IFRS）は遡及再表示で BS が不連続＝add_basis_transition で
  basis_changed を立て、#7 の YoY 系（資産成長・アクルーアル・純株式発行）を NaN 化する。
- コードクロスウォーク：EDINET secCode も J-Quants Code も 5 桁＝そのまま結合できる。
- 取得物は data/edinet/（gitignore）。長形式キャッシュも同所（コミット不可）。

データ完走を待たずに動く：現在キャッシュ済みの zip だけからでもパネルを組める（増分構築）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..data.sources import edinet as ed
from ..data.sources import edinet_taxonomy as tax
from .fundamentals import point_in_time

# as-of パネルに載る正準フィールド（生ライン項目）。raw のみ＝比率は #7 で自前計算。
CANONICAL_FIELDS: list[str] = (list(tax.FIELD_MAP) + list(tax.FIELD_MAP_COMMON)
                               + list(tax.FIELD_MAP_ENTITY) + ["interest_debt"])

_LONG_CACHE = ed._CACHE / "fundamentals_long.parquet"
_META_COLS = ["docID", "secCode", "submitDateTime", "periodEnd", "docTypeCode"]


def _annual_doc_metadata(list_dir: Optional[Path] = None) -> pd.DataFrame:
    """by-date 一覧ミラーから有報メタ（docID→Code/提出日時/期末/種別）を集約。"""
    list_dir = Path(list_dir) if list_dir else ed._CACHE / "list"
    rows = []
    for p in sorted(list_dir.glob("*.parquet")):
        df = pd.read_parquet(p)
        if "_empty" in df.columns or "docTypeCode" not in df.columns:
            continue
        a = ed.annual_report_documents(df)
        if len(a):
            rows.append(a[_META_COLS])
    if not rows:
        return pd.DataFrame(columns=_META_COLS)
    meta = pd.concat(rows, ignore_index=True).dropna(subset=["docID"])
    return meta.drop_duplicates("docID").set_index("docID")


def build_edinet_long(rebuild: bool = False, docs_dir: Optional[Path] = None,
                      cache_path: Optional[Path] = None,
                      verbose: bool = False) -> pd.DataFrame:
    """キャッシュ済み有報 zip をパースして正準フィールドの長形式を作る（増分・冪等）。

    返り値の列：Code, DiscDate(=submitDateTime), period_end, docID, docTypeCode,
    basis, ＋ CANONICAL_FIELDS。rebuild=False なら既存キャッシュに無い zip だけ追加。
    """
    docs_dir = Path(docs_dir) if docs_dir else ed._CACHE / "docs"
    cache_path = Path(cache_path) if cache_path else _LONG_CACHE
    meta = _annual_doc_metadata()

    have = pd.DataFrame()
    if cache_path.exists() and not rebuild:
        have = pd.read_parquet(cache_path)
    done = set(have["docID"]) if not have.empty else set()

    new_rows = []
    for p in sorted(docs_dir.glob("*_5.zip")):
        did = p.stem.rsplit("_", 1)[0]
        if did in done or did not in meta.index:
            continue
        try:
            canon = tax.extract_canonical(ed.read_xbrl_csv(p))
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[warn] {did}: {str(e)[:60]}")
            continue
        m = meta.loc[did]
        row = {"docID": did, "Code": str(m["secCode"]),
               "DiscDate": m["submitDateTime"], "period_end": m["periodEnd"],
               "docTypeCode": str(m["docTypeCode"])}
        row.update(canon)
        new_rows.append(row)

    if new_rows:
        add = pd.DataFrame(new_rows)
        out = pd.concat([have, add], ignore_index=True) if not have.empty else add
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path)
        if verbose:
            print(f"[edinet_long] +{len(add)} 件（累計 {len(out)}）→ {cache_path}")
    else:
        out = have
    return out.reset_index(drop=True) if not out.empty else out


def add_basis_transition(long: pd.DataFrame) -> pd.DataFrame:
    """各銘柄を period_end 昇順で見て、会計基準が前期から変わった開示に basis_changed=True。

    移行年をまたぐ成長率・アクルーアル・純株式発行は遡及再表示で壊れるため、#7 で
    basis_changed を使って当該 YoY を NaN 化する。
    """
    if long.empty:
        return long.assign(prev_basis=pd.Series(dtype="object"),
                           basis_changed=pd.Series(dtype="bool"))
    df = long.sort_values(["Code", "period_end"]).copy()
    df["prev_basis"] = df.groupby("Code")["basis"].shift(1)
    df["basis_changed"] = df["prev_basis"].notna() & (df["basis"] != df["prev_basis"])
    return df


def edinet_fundamentals_panel(rebal_dates, fields: list[str],
                              codes: Optional[Iterable] = None, lag_days: int = 1
                              ) -> dict[str, pd.DataFrame]:
    """EDINET 三表の as-of パネル（提出日アンカー）。fundamentals_panel と同じ機構・FYE 非依存。

    返り値 {field: DataFrame(index=rebal, columns=Code)}。codes=None で全銘柄。
    fields は CANONICAL_FIELDS のほか、長形式に付与済みの派生列（#7）も指定可。
    """
    long = build_edinet_long()
    if long.empty:
        rebal = pd.DatetimeIndex(sorted(pd.to_datetime(list(rebal_dates)))).normalize()
        return {f: pd.DataFrame(index=rebal, dtype="float64") for f in fields}
    if codes is not None:
        want = {str(c) for c in codes}
        long = long[long["Code"].isin(want)]
    return point_in_time(long, rebal_dates, fields, date_col="DiscDate",
                         code_col="Code", lag_days=lag_days)
