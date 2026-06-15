"""EDINET v2 パース/フィルタ/キャッシュ（ネットワーク不要）の検証。"""
import pandas as pd

from invest_system.data.sources import edinet as ed


def _sample() -> list[dict]:
    """第三者TOB届出 / 自社株TOB届出 / 大量保有 / 訂正大量保有 の 4 件。"""
    return [
        {"seqNumber": 1, "docID": "S1", "secCode": "72030", "ordinanceCode": "040",
         "docTypeCode": "240", "subjectEdinetCode": "E9", "filerName": "買収者X",
         "submitDateTime": "2026-06-01 09:30", "csvFlag": "1", "legalStatus": "1"},
        {"seqNumber": 2, "docID": "S2", "secCode": "45020", "ordinanceCode": "050",
         "docTypeCode": "240", "submitDateTime": "2026-06-01 10:00"},
        {"seqNumber": 3, "docID": "S3", "secCode": "63010", "ordinanceCode": "060",
         "docTypeCode": "350", "issuerEdinetCode": "E1", "filerName": "エフィッシモ",
         "submitDateTime": "2026-06-01 15:45", "csvFlag": "0"},
        {"seqNumber": 4, "docID": "S4", "ordinanceCode": "060", "docTypeCode": "360",
         "submitDateTime": "2026-06-01 16:10"},
    ]


def test_parse_keeps_codes_as_strings():
    df = ed.parse_documents(_sample())
    # 府令/書類種別/証券コードは先頭ゼロ・区分値を保つため文字列のまま（"040"→40 にしない）
    assert df["ordinanceCode"].iloc[0] == "040"
    assert df["docTypeCode"].iloc[0] == "240"
    assert df["secCode"].iloc[0] == "72030"
    assert df["ordinanceCode"].dtype == "string"
    assert df["csvFlag"].dtype == "string"


def test_parse_datetime_and_seqnumber():
    df = ed.parse_documents(_sample())
    assert df["submitDateTime"].iloc[0] == pd.Timestamp("2026-06-01 09:30")  # PIT アンカー
    assert pd.api.types.is_datetime64_any_dtype(df["submitDateTime"])
    assert str(df["seqNumber"].dtype) == "Int64"
    assert df["seqNumber"].iloc[0] == 1


def test_parse_schema_stable_and_empty():
    # 列が欠けるレコードでも全列が揃う（日跨ぎ by-date 突合の頑健性）
    df = ed.parse_documents([{"docID": "X", "ordinanceCode": "060"}])
    for c in ed._ALL_COLS:
        assert c in df.columns
    # 空 results → 0 行・全列付き
    empty = ed.parse_documents([])
    assert len(empty) == 0
    assert list(empty.columns) == ed._ALL_COLS


def test_tob_documents_excludes_self_tender_by_default():
    df = ed.parse_documents(_sample())
    tob = ed.tob_documents(df)
    assert list(tob["docID"]) == ["S1"]          # 040/240 のみ（050 自社株TOB は除外）
    incl = ed.tob_documents(df, include_self=True)
    assert set(incl["docID"]) == {"S1", "S2"}


def test_tob_documents_doc_type_filter():
    df = ed.parse_documents(_sample())
    # docType を 270(公開買付報告書) に絞るとサンプル（240 のみ）には該当なし
    assert len(ed.tob_documents(df, doc_types=(ed.DOC_TOB_REPORT,))) == 0


def test_large_holding_documents_by_ordinance_060():
    df = ed.parse_documents(_sample())
    lh = ed.large_holding_documents(df)
    # 350(新規)と 360(訂正/変更)を府令060 で一括捕捉（docTypeCode に依存せず取りこぼさない）
    assert set(lh["docID"]) == {"S3", "S4"}


def test_filters_on_empty():
    e = ed.parse_documents([])
    assert ed.tob_documents(e).empty
    assert ed.large_holding_documents(e).empty


def test_date_normalization():
    assert ed._ymd("2026-06-01") == "20260601"
    assert ed._ymd("20260601") == "20260601"
    assert ed._ymd(pd.Timestamp("2026-06-01")) == "20260601"
    assert ed._iso("20260601") == "2026-06-01"
    assert ed._iso("2026-06-01") == "2026-06-01"


def test_edinet_catalog_wired_and_differential_update(tmp_path):
    from invest_system.data.catalog import EDINET_DATASETS, Dataset, _plain_date
    from invest_system.data.updater import DataUpdater, scan_cache_dates

    ds = EDINET_DATASETS["edinet_docs"]
    assert ds.cadence == "daily" and ds.cache_subdir == "list"
    # キャッシュ名 {YYYYMMDD}.parquet を _plain_date が日付化（差分更新の取得済み復元）
    d = tmp_path / "list"
    d.mkdir()
    (d / "20240101.parquet").touch()
    assert scan_cache_dates(ds, tmp_path) == {"20240101"}

    # 専用 updater（base=tmp_path）で欠損日のみ取得（fetch はモック）。01 は既取得 →
    # 02/03 のみ取得＝既存 J-Quants 夜間とは別ベースで差分更新が機能する
    calls: list[str] = []
    mock = {"edinet_docs": Dataset("edinet_docs", "daily", "list",
                                   lambda x: calls.append(x), _plain_date)}
    up = DataUpdater(datasets=mock, refresh_datasets={}, base=str(tmp_path),
                     manifest_path=str(tmp_path / "m.json"), start="2024-01-01")
    up.update(until="2024-01-03", verbose=False)
    assert calls == ["20240102", "20240103"]


def test_list_cache_roundtrip_and_empty_marker(tmp_path):
    cache = tmp_path / "20260601.parquet"
    ed._save_list(ed.parse_documents(_sample()), cache)
    back = ed._read_list_cache(cache)
    assert len(back) == 4
    assert back["ordinanceCode"].iloc[0] == "040"        # Parquet 往復で文字列保持
    # 空（書類なしの日）→ マーカー保存 → 全列付き空 df で復元（再取得を防ぐ）
    empty_cache = tmp_path / "20260607.parquet"
    ed._save_list(ed.parse_documents([]), empty_cache)
    back_e = ed._read_list_cache(empty_cache)
    assert len(back_e) == 0
    for c in ed._ALL_COLS:
        assert c in back_e.columns
