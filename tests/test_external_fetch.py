"""外部価格の無人差分更新（検証付き追記）の検証。ネット不要（fetch 注入）。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.data.external import _PRICE_FILES
from invest_system.data.external_fetch import (
    _utc_today, merge_new_dates, update_external_prices, validate_overlap,
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


def test_merge_cutoff_excludes_forming_bar():
    # ほぼ24h銘柄は取得時点で当日足が形成中＝cutoff（当日）以降は追記しない。
    old = _frame("2026-01-05", 10)
    new = _frame("2026-01-05", 15)
    cutoff = new.index[13]                                 # 14本目以降は「未確定」
    merged, n_new = merge_new_dates(old, new, cutoff=cutoff)
    assert n_new == 3                                      # 11..13本目のみ追記
    assert merged.index.max() == new.index[12]
    # 翌日（cutoff が進む）に残りが確定足として追記される＝凍結ではなく繰延
    merged2, n2 = merge_new_dates(merged, new, cutoff=new.index[-1] + pd.Timedelta(days=1))
    assert n2 == 2 and merged2.index.max() == new.index[-1]


def test_update_external_prices_default_cutoff_blocks_today(tmp_path):
    # 既定 cutoff=UTC当日：fetch が「今日の部分足」を返しても保存されない。
    pdir = tmp_path / "investers"
    pdir.mkdir(parents=True)
    today = _utc_today()
    idx = pd.date_range(end=today, periods=40, freq="D")   # 末尾＝今日（形成中）
    close = 100.0 + np.arange(40, dtype=float)
    full = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": np.full(40, 1.0)}, index=idx)
    full.index.name = "date"
    full.iloc[:35].to_parquet(pdir / _PRICE_FILES["gold"])

    rep = update_external_prices(["gold"], base=str(tmp_path),
                                 fetch=lambda s, period="3mo": full)
    assert rep.iloc[0]["status"] == "OK"
    assert int(rep.iloc[0]["n_new"]) == 4                  # 今日の1本だけ遮断
    saved = pd.read_parquet(pdir / _PRICE_FILES["gold"])
    assert saved.index.max() < today                       # 部分足は凍結されない


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
