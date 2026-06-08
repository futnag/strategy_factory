"""差分更新エンジンの検証。ネットワーク不要（取得はモック）。"""
import pandas as pd

from invest_system.data.catalog import Dataset, _suffix_date
from invest_system.data.updater import (
    DataUpdater, Manifest, candidate_dates, missing_dates, scan_cache_dates,
)


def test_candidate_dates_daily_and_weekly():
    daily = candidate_dates("daily", "2024-01-01", "2024-01-05")
    assert daily == ["20240101", "20240102", "20240103", "20240104", "20240105"]
    weekly = candidate_dates("weekly", "2024-01-01", "2024-01-31")
    # 2024-01 の金曜：5,12,19,26
    assert weekly == ["20240105", "20240112", "20240119", "20240126"]
    assert candidate_dates("daily", "2024-02-01", "2024-01-01") == []


def test_missing_dates():
    cands = ["20240101", "20240102", "20240103"]
    assert missing_dates(cands, {"20240102"}) == ["20240101", "20240103"]


def test_scan_cache_dates(tmp_path):
    d = tmp_path / "margin_weekly"
    d.mkdir()
    (d / "date_20260529.parquet").touch()
    (d / "date_20260522.parquet").touch()
    (d / "junk.parquet").touch()                 # 日付化できない名は無視
    ds = Dataset("w", "weekly", "margin_weekly", lambda x: None, _suffix_date)
    assert scan_cache_dates(ds, tmp_path) == {"20260529", "20260522"}


def test_manifest_roundtrip(tmp_path):
    m = Manifest(tmp_path / "m.json")
    m.mark("ds", "20240101")
    m.mark("ds", "20240101")                      # 重複は無視
    m.mark("ds", "20240102")
    m.save()
    assert Manifest(tmp_path / "m.json").fetched("ds") == {"20240101", "20240102"}


def test_updater_fetches_only_missing_idempotent(tmp_path):
    calls = []
    ds = Dataset("mock", "daily", "mock", lambda d: calls.append(d), _suffix_date)
    up = DataUpdater(datasets={"mock": ds}, refresh_datasets={}, base=str(tmp_path),
                     manifest_path=str(tmp_path / "m.json"), start="2024-01-01")
    rep = up.update(until="2024-01-05", verbose=False)
    assert rep["mock"] == {"missing": 5, "fetched": 5}
    assert len(calls) == 5
    # 2回目は欠損0（マニフェストで既取得を把握）＝差分のみ取得
    rep2 = up.update(until="2024-01-05", verbose=False)
    assert rep2["mock"]["fetched"] == 0
    assert len(calls) == 5


def test_update_default_targets_maintained_only(tmp_path):
    seen = {"a": [], "b": []}
    dsets = {
        "a": Dataset("a", "daily", "a", lambda d: seen["a"].append(d), _suffix_date,
                     maintained=True),
        "b": Dataset("b", "daily", "b", lambda d: seen["b"].append(d), _suffix_date,
                     maintained=False),
    }
    up = DataUpdater(datasets=dsets, refresh_datasets={}, base=str(tmp_path),
                     manifest_path=str(tmp_path / "m.json"), start="2024-01-01")
    up.update(until="2024-01-03", verbose=False)   # 既定= maintained のみ
    assert len(seen["a"]) == 3 and seen["b"] == []


def test_catalog_has_bydate_mirrors():
    from invest_system.data.catalog import DATASETS
    assert "fins_summary" in DATASETS
    fs = DATASETS["fins_summary"]
    assert fs.cadence == "daily" and fs.cache_subdir == "fins_summary"
    assert fs.maintained                                  # 案A：全件ミラー対象
    assert DATASETS["daily_quotes"].maintained            # 株価も全件ミラーへ変更


def test_save_parquet_empty_marker(tmp_path):
    from invest_system.data.sources.jquants import _save_parquet
    cache = tmp_path / "x.parquet"
    _save_parquet(pd.DataFrame(), cache)                  # 空(無開示日)もマーカー保存可
    assert pd.read_parquet(cache).empty
    _save_parquet(pd.DataFrame({"a": [1, 2]}), cache)     # 非空はそのまま
    assert len(pd.read_parquet(cache)) == 2


def test_refresh_dataset_invoked(tmp_path):
    from invest_system.data.catalog import RefreshSpec
    calls = []
    spec = RefreshSpec("idx", lambda s, u: (calls.append((s, u)), 42)[1])
    up = DataUpdater(datasets={}, refresh_datasets={"idx": spec},
                     base=str(tmp_path), manifest_path=str(tmp_path / "m.json"),
                     start="2016-06-13")
    rep = up.update(until="2026-06-06", verbose=False)
    assert rep["idx"] == {"refreshed_rows": 42}
    assert calls == [("2016-06-13", "2026-06-06")]
