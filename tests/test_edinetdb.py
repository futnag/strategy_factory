"""補助ソース edinetdb.jp クライアント（ネットワーク不要・合成フィクスチャ）。

パース・基準正規化・突合（reconcile）・日次クォータ（100/日）・コードクロスウォークを検証。
"""
import json

import pandas as pd
import pytest

from invest_system.data.sources import edinet as ed
from invest_system.data.sources import edinetdb as edb


def test_norm_basis():
    assert edb._norm_basis("IFRS") == "IFRS"
    assert edb._norm_basis("US GAAP") == "USGAAP"
    assert edb._norm_basis("USGAAP") == "USGAAP"
    assert edb._norm_basis("JP GAAP") == "JGAAP"
    assert edb._norm_basis("JGAAP") == "JGAAP"
    assert edb._norm_basis(None) is None


def test_parse_financials_maps_canonical():
    payload = {"data": [
        {"fiscal_year": 2024, "accounting_standard": "IFRS", "revenue": 100.0,
         "operating_income": 10.0, "net_income": 7.0, "total_assets": 500.0,
         "cf_operating": 9.0, "cf_investing": -4.0, "shares_issued": 1000.0,
         "split_adjustment_factor": 2.0, "roe_official": 0.12},
        {"fiscal_year": 2025, "accounting_standard": "IFRS", "revenue": 110.0,
         "net_income": 8.0, "total_assets": 520.0},
    ], "meta": {}}
    df = edb.parse_financials(payload, edinet_code="E0001")
    assert list(df["fiscal_year"]) == [2024, 2025]
    assert str(df["fiscal_year"].dtype) == "Int64"
    assert df.loc[0, "net_sales"] == 100.0          # revenue -> net_sales
    assert df.loc[0, "profit"] == 7.0               # net_income -> profit
    assert df.loc[0, "cfo"] == 9.0 and df.loc[0, "cfi"] == -4.0
    assert df.loc[0, "shares_outstanding"] == 1000.0
    assert df.loc[0, "split_adjustment_factor"] == 2.0
    assert "roe_official" not in df.columns         # 加工済み比率は取り込まない
    assert df.loc[0, "basis"] == "IFRS"
    assert df.loc[0, "edinet_code"] == "E0001"


def test_parse_financials_empty():
    assert edb.parse_financials({"data": []}).empty
    assert edb.parse_financials({}).empty


def test_reconcile_ok_warn_miss():
    ours = pd.DataFrame([{"period_end": "2025-03-31", "net_sales": 100.0,
                          "total_assets": 50.0, "cfo": None}])
    edbl = pd.DataFrame([{"fiscal_year": 2025, "net_sales": 100.0,
                          "total_assets": 51.0, "cfo": 9.0}])
    rec = edb.reconcile(ours, edbl, fields=["net_sales", "total_assets", "cfo"], tol=0.01)
    flags = dict(zip(rec["field"], rec["flag"]))
    assert flags["net_sales"] == "OK"               # 完全一致
    assert flags["total_assets"] == "WARN"          # 50 vs 51 = 2% > 1%
    assert flags["cfo"] == "MISS"                    # 公式側 None・edinetdb 在り


def test_daily_quota_counts_and_persists(tmp_path):
    qpath = tmp_path / "quota.json"
    q = edb.DailyQuota(path=qpath, cap=2)
    assert q.remaining() == 2
    q.consume(); q.consume()
    assert q.remaining() == 0
    with pytest.raises(edb.QuotaExceeded):
        q.consume()
    # 別インスタンス（同日）でも消費済みカウントを引き継ぐ＝再実行で上限を超えない
    q2 = edb.DailyQuota(path=qpath, cap=2)
    assert q2.remaining() == 0


def test_seccode_to_edinet_crosswalk(tmp_path):
    rec = [{"docID": "S1", "secCode": "72030", "edinetCode": "E02144",
            "ordinanceCode": "010", "docTypeCode": "120"}]
    (tmp_path / "list").mkdir()
    ed._save_list(ed.parse_documents(rec), tmp_path / "list" / "20250620.parquet")
    m = edb.seccode_to_edinet(list_dir=tmp_path / "list")
    assert m["72030"] == "E02144"
