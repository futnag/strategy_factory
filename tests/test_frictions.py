"""日本市場の執行フリクション（値幅制限・貸借銘柄マスク）の検証。ネットワーク不要。"""
import pandas as pd

from invest_system.equities.frictions import (
    limit_lock_flags, short_notional_coverage, shortable_mask,
)


def _panels():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    close = pd.DataFrame({"A": [100., 110, 120], "B": [100., 90, 80]}, index=idx)
    high = pd.DataFrame({"A": [100., 110, 125], "B": [101., 95, 81]}, index=idx)
    low = pd.DataFrame({"A": [99., 101, 118], "B": [98., 90, 79]}, index=idx)
    ul = pd.DataFrame({"A": [0., 1, 1], "B": [0., 0, 0]}, index=idx)
    ll = pd.DataFrame({"A": [0., 0, 0], "B": [0., 1, 0]}, index=idx)
    vol = pd.DataFrame({"A": [1e5, 1e5, 1e5], "B": [1e5, 1e5, 0.]}, index=idx)
    return idx, close, high, low, ul, ll, vol


def test_limit_lock_pinned_close_blocks_buy():
    idx, close, high, low, ul, ll, vol = _panels()
    no_buy, no_sell = limit_lock_flags(close, high, low, ul, ll)
    # d1: A は UL=1 かつ close==high（ストップ高引け）→ 買い不能・売りは可
    assert bool(no_buy.loc[idx[1], "A"]) and not bool(no_sell.loc[idx[1], "A"])
    # d2: A は UL=1 だが close<high（日中タッチのみ・引けは剥がれ）→ 引け執行可
    assert not bool(no_buy.loc[idx[2], "A"])


def test_limit_lock_pinned_close_blocks_sell():
    idx, close, high, low, ul, ll, vol = _panels()
    no_buy, no_sell = limit_lock_flags(close, high, low, ul, ll)
    # d1: B は LL=1 かつ close==low（ストップ安引け）→ 売り不能・買いは可
    assert bool(no_sell.loc[idx[1], "B"]) and not bool(no_buy.loc[idx[1], "B"])
    assert not bool(no_sell.loc[idx[0], "B"])


def test_limit_lock_zero_volume_blocks_both():
    idx, close, high, low, ul, ll, vol = _panels()
    no_buy, no_sell = limit_lock_flags(close, high, low, ul, ll, volume=vol)
    # d2: B は出来高 0（終日約定なし）→ 両側不能
    assert bool(no_buy.loc[idx[2], "B"]) and bool(no_sell.loc[idx[2], "B"])
    # close 欠損や NaN フラグはブロックしない（既定 False）
    assert not bool(no_buy.loc[idx[0], "A"])


def _weekly():
    # 週次レコード：A は貸借（IssType=2）、B は信用のみ（IssType=1）、C は IssType
    # 欠損だが制度売残あり（代替判定で True）
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-05", "2024-01-05", "2024-01-05",
                                "2024-01-12"]),
        "Code": ["A", "B", "C", "A"],
        "IssType": [2, 1, None, 2],
        "ShrtStdVol": [100.0, 0.0, 50.0, 120.0],
    })


def test_shortable_mask_pit_lag_and_types():
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-09", "2024-01-31"])
    m = shortable_mask(_weekly(), dates, lag_days=3, tolerance_days=60)
    # 1/5 時点：lag 3日 → 1/2 以前のレコードなし → 全 False（公表前は使わない）
    assert not m.loc["2024-01-05"].any()
    # 1/9 時点：1/5 レコードが可視。A=貸借 True / B=信用のみ False / C=代替判定 True
    assert bool(m.loc["2024-01-09", "A"]) and bool(m.loc["2024-01-09", "C"])
    assert not bool(m.loc["2024-01-09", "B"])
    # 1/31 時点：直近（1/12）の A は True。C は 1/12 にレコードが無いが、銘柄ごとの
    # LOCF で 1/5 の判定（True）が tolerance 内なら生きる（行レベル ffill だと落ちる）
    assert bool(m.loc["2024-01-31", "A"]) and bool(m.loc["2024-01-31", "C"])
    assert not bool(m.loc["2024-01-31", "B"])


def test_shortable_mask_tolerance_expires():
    dates = pd.DatetimeIndex(["2024-06-28"])
    m = shortable_mask(_weekly(), dates, lag_days=3, tolerance_days=60)
    # 最終レコード 1/12 から 60日超 → 鮮度切れで False（貸借区分の喪失を失効）
    assert not m.loc["2024-06-28"].any()


def test_short_notional_coverage():
    w = pd.Series({"A": 0.5, "B": -0.3, "C": -0.2})
    row = pd.Series({"A": True, "B": True, "C": False})
    assert short_notional_coverage(w, row) == 0.6
    assert short_notional_coverage(pd.Series({"A": 1.0}), row) == 1.0
