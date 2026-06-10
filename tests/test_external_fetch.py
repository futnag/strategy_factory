"""外部価格の無人差分更新（検証付き追記）の検証。ネット不要（fetch 注入）。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.data.external import _PRICE_FILES
from invest_system.data.external_fetch import (
    merge_new_dates, update_external_prices, validate_overlap,
)


def _frame(start, n, base=100.0, cols=True):
    idx = pd.date_range(start, periods=n, freq="B")
    close = base + np.arange(n, dtype=float)
    df = pd.DataFrame({"open": close - 0.5, "high": close + 1.0,
                       "low": close - 1.0, "close": close,
                       "volume": np.full(n, 10.0)}, index=idx)
    df.index.name = "date"
    if cols:
        df["change_pct"] = df["close"].pct_change() * 100
    return df


def test_validate_overlap_accepts_close_and_rejects_unit_mismatch():
    old = _frame("2026-01-05", 40)["close"]
    ok, diff, n = validate_overlap(old, old * 1.001)
    assert ok and n >= 5 and diff == pytest.approx(0.001, rel=1e-6)
    ok2, diff2, _ = validate_overlap(old, old * 100.0)   # 単位違い（セント/ドル等）
    assert not ok2 and diff2 > 0.02
    ok3, _, n3 = validate_overlap(old, _frame("2030-01-01", 10)["close"])
    assert not ok3 and n3 == 0                            # 重複なし＝照合不能


def test_merge_new_dates_appends_only_future_and_is_idempotent():
    old = _frame("2026-01-05", 10)
    new = _frame("2026-01-05", 15)
    new.loc[:, "close"] = new["close"] + 99.0             # 重複日は値が違っても
    merged, n_new = merge_new_dates(old, new)
    assert n_new == 5                                      # 追記は新規日だけ
    pd.testing.assert_frame_equal(merged.iloc[:10], old)   # 既存履歴は不変
    merged2, n2 = merge_new_dates(merged, new)
    assert n2 == 0 and merged2.equals(merged)              # 冪等


def test_merge_recomputes_change_pct_across_boundary():
    old = _frame("2026-01-05", 5)
    new = _frame("2026-01-05", 7)
    merged, _ = merge_new_dates(old, new)
    expect = (merged["close"].iloc[5] / merged["close"].iloc[4] - 1) * 100
    assert merged["change_pct"].iloc[5] == pytest.approx(round(expect, 2))


def test_update_external_prices_writes_and_rejects(tmp_path):
    pdir = tmp_path / "investers"
    pdir.mkdir(parents=True)
    old = _frame("2026-01-05", 30)
    old.to_parquet(pdir / _PRICE_FILES["gold"])
    old.to_parquet(pdir / _PRICE_FILES["wti"])

    def fake(symbol, period="3mo"):
        if symbol == "GC=F":                               # 正常：5日延長
            return _frame("2026-01-05", 35, cols=False)
        return _frame("2026-01-05", 35, base=10000.0, cols=False)  # 単位違い

    rep = update_external_prices(["gold", "wti"], base=str(tmp_path),
                                 fetch=fake).set_index("key")
    assert rep.loc["gold", "status"] == "OK" and rep.loc["gold", "n_new"] == 5
    assert str(rep.loc["wti", "status"]).startswith("REJECT")
    saved = pd.read_parquet(pdir / _PRICE_FILES["gold"])
    assert len(saved) == 35
    untouched = pd.read_parquet(pdir / _PRICE_FILES["wti"])
    pd.testing.assert_frame_equal(untouched, old, check_freq=False)  # 不採用＝不変


def test_update_external_prices_survives_fetch_error(tmp_path):
    (tmp_path / "investers").mkdir(parents=True)
    _frame("2026-01-05", 30).to_parquet(
        tmp_path / "investers" / _PRICE_FILES["gold"])

    def boom(symbol, period="3mo"):
        raise RuntimeError("network down")

    rep = update_external_prices(["gold"], base=str(tmp_path), fetch=boom)
    assert str(rep.iloc[0]["status"]).startswith("FAIL:")
