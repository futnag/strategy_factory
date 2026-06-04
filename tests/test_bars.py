"""情報主導型バーと tick rule の正しさを検証。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.data.bars import (
    apply_tick_rule,
    dollar_bars,
    dollar_imbalance_bars,
    tick_bars,
    volume_bars,
)


def _trades(prices, volumes=None, sides=None, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="s")
    data = {"price": prices, "volume": volumes if volumes is not None else [1.0] * len(prices)}
    if sides is not None:
        data["side"] = sides
    return pd.DataFrame(data, index=idx)


def test_dollar_bars_counts_and_ohlc():
    trades = _trades([100, 101, 102, 101, 100, 101])
    bars = dollar_bars(trades, threshold=200)
    assert len(bars) == 3
    b0 = bars.iloc[0]
    assert b0["open"] == 100 and b0["close"] == 101
    assert b0["high"] == 101 and b0["low"] == 100
    assert b0["dollar"] == pytest.approx(201.0)
    assert b0["n_ticks"] == 2


def test_tick_bars_fixed_count():
    trades = _trades([100, 101, 102, 101, 100, 101])
    bars = tick_bars(trades, threshold=2)
    assert len(bars) == 3
    assert (bars["n_ticks"] == 2).all()


def test_volume_bars_fixed_volume():
    trades = _trades([100, 101, 102, 101, 100, 101], volumes=[1.0] * 6)
    bars = volume_bars(trades, threshold=3)
    assert len(bars) == 2
    assert (bars["n_ticks"] == 3).all()


def test_threshold_must_be_positive():
    trades = _trades([100, 101])
    with pytest.raises(ValueError):
        dollar_bars(trades, threshold=0)


def test_apply_tick_rule_carries_zero_forward():
    signs = apply_tick_rule([100, 101, 101, 100, 100, 102])
    assert list(signs) == [1, 1, 1, -1, -1, 1]


def test_dollar_imbalance_bars_positive_flow():
    trades = _trades([100, 101, 102, 103, 104, 105], sides=["buy"] * 6)
    bars = dollar_imbalance_bars(trades, threshold=250)
    assert len(bars) == 2
    assert (bars["imbalance"] > 0).all()


def test_dollar_imbalance_bars_negative_flow():
    trades = _trades([100, 100, 100, 100], sides=["sell"] * 4)
    bars = dollar_imbalance_bars(trades, threshold=250)
    assert len(bars) == 1                      # 末尾の未確定ブロックは出力しない
    assert bars.iloc[0]["imbalance"] < 0


def test_bars_index_tz_naive_from_tzaware_trades():
    # parse_trades は tz-aware を返すが、バーは tz-naive UTC に正規化される
    idx = pd.date_range("2020-01-01", periods=6, freq="s", tz="UTC")
    trades = pd.DataFrame({"price": [100, 101, 102, 101, 100, 101],
                           "volume": [1.0] * 6, "side": ["buy"] * 6}, index=idx)
    assert dollar_bars(trades, threshold=200).index.tz is None
    assert dollar_imbalance_bars(trades, threshold=200).index.tz is None


def test_dollar_imbalance_tick_sign_method():
    # side 無し → tick rule で符号付け（価格上昇 = 買い圧）
    trades = _trades([100, 101, 102, 103])
    bars = dollar_imbalance_bars(trades, threshold=250, sign_method="tick")
    assert len(bars) >= 1
    assert (bars["imbalance"] > 0).all()
