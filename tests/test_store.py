"""Processed (Silver) 層 store.py の検証。ネットワーク不要・合成 by-date。"""
import numpy as np
import pandas as pd

from invest_system.data.store import (
    health_check,
    load_wide,
    materialize_all,
    materialize_long,
    materialize_wide,
    rebuild_adjusted,
)


def _raw_day(invdir, ymd, codes, closes, adjf=None):
    n = len(codes)
    pd.DataFrame({
        "Date": [pd.Timestamp(ymd)] * n, "Code": codes,
        "O": closes, "H": closes, "L": closes, "C": closes,
        "Vo": [100.0] * n, "Va": [1e6] * n,
        "AdjFactor": adjf if adjf is not None else [1.0] * n,
        "UL": closes, "LL": closes,
    }).to_parquet(invdir / f"{ymd.replace('-', '')}.parquet")


def _empty(invdir, ymd):
    pd.DataFrame({"_empty": pd.Series([], dtype="bool")}).to_parquet(
        invdir / f"{ymd.replace('-', '')}.parquet")


def test_materialize_wide_and_alias(tmp_path):
    raw = tmp_path / "jquants" / "daily"; raw.mkdir(parents=True)
    _raw_day(raw, "2024-01-04", ["7203", "6758"], [100., 200.])
    _raw_day(raw, "2024-01-05", ["7203", "6758"], [110., 190.])
    _empty(raw, "2024-01-06")                              # 祝日マーカー
    rep = materialize_wide(base=str(tmp_path))
    assert rep["appended_dates"] == 2
    cl = load_wide("close", base=str(tmp_path))
    assert cl.loc["2024-01-05", "7203"] == 110.0
    assert sorted(cl.columns) == ["6758", "7203"]
    assert load_wide("C", base=str(tmp_path)).equals(cl)   # 別名解決
    assert load_wide("turnover", base=str(tmp_path)).loc["2024-01-04", "6758"] == 1e6


def test_materialize_incremental_and_idempotent(tmp_path):
    raw = tmp_path / "jquants" / "daily"; raw.mkdir(parents=True)
    _raw_day(raw, "2024-01-04", ["7203"], [100.])
    _raw_day(raw, "2024-01-05", ["7203"], [110.])
    assert materialize_wide(base=str(tmp_path))["appended_dates"] == 2
    # 新規 1 日 → 増分 1（新規上場 6758 は列追加）
    _raw_day(raw, "2024-01-09", ["7203", "6758"], [120., 50.])
    assert materialize_wide(base=str(tmp_path))["appended_dates"] == 1
    cl = load_wide("close", base=str(tmp_path))
    assert cl.shape == (3, 2) and cl.loc["2024-01-09", "6758"] == 50.0
    assert np.isnan(cl.loc["2024-01-04", "6758"])          # 上場前は NaN
    # 再実行は冪等（新規なし）
    assert materialize_wide(base=str(tmp_path))["appended_dates"] == 0


def test_rebuild_adjusted_split(tmp_path):
    raw = tmp_path / "jquants" / "daily"; raw.mkdir(parents=True)
    _raw_day(raw, "2024-01-04", ["7203"], [100.], adjf=[1.0])
    _raw_day(raw, "2024-01-05", ["7203"], [50.], adjf=[0.5])   # 1:2 分割（ex-date）
    _raw_day(raw, "2024-01-09", ["7203"], [51.], adjf=[1.0])
    materialize_all(base=str(tmp_path))
    adj = load_wide("adj_close", base=str(tmp_path))
    assert adj.loc["2024-01-04", "7203"] == 50.0           # 100×0.5（過去を後方調整）
    assert adj.loc["2024-01-05", "7203"] == 50.0
    assert adj.loc["2024-01-09", "7203"] == 51.0           # 最新は無調整
    assert load_wide("AdjC", base=str(tmp_path)).equals(adj)  # 別名


def test_health_check_alignment(tmp_path):
    raw = tmp_path / "jquants" / "daily"; raw.mkdir(parents=True)
    _raw_day(raw, "2024-01-04", ["7203", "6758"], [100., 200.])
    _raw_day(raw, "2024-01-05", ["7203", "6758"], [110., 190.])
    materialize_all(base=str(tmp_path))
    hc = health_check(base=str(tmp_path))
    close_row = hc[hc["field"] == "close"].iloc[0]
    assert close_row["days"] == 2 and close_row["codes"] == 2
    assert bool(hc["aligned_to_close"].all())              # 全フィールド日付整合


def test_materialize_long_partitioned(tmp_path):
    raw = tmp_path / "jquants" / "daily"; raw.mkdir(parents=True)
    _raw_day(raw, "2023-12-29", ["7203"], [100.])
    _raw_day(raw, "2024-01-04", ["7203", "6758"], [110., 50.])
    rep = materialize_long(base=str(tmp_path))
    assert rep["rows"] == 3 and rep["years"] == 2
    out = tmp_path / "processed" / "equities" / "long"
    assert (out / "year=2023").exists() and (out / "year=2024").exists()  # Hive分割
    df24 = pd.read_parquet(out / "year=2024")
    assert set(df24["Code"]) == {"7203", "6758"} and len(df24) == 2
