"""bitbank データ取込（parse / fetch の DI）を検証。"""
import pandas as pd

from invest_system.data.sources.bitbank import fetch_trades, parse_trades


def test_parse_trades_schema_and_values():
    raw = [
        {"timestamp": 1700000000000, "price": 100.0, "amount": 0.5, "side": "buy"},
        {"timestamp": 1700000001000, "price": 101.0, "amount": 0.3, "side": "sell"},
    ]
    df = parse_trades(raw)
    assert list(df.columns) == ["price", "volume", "side"]
    assert df.shape == (2, 3)
    assert df["price"].tolist() == [100.0, 101.0]
    assert df["volume"].tolist() == [0.5, 0.3]
    assert df["side"].tolist() == ["buy", "sell"]
    assert df.index.is_monotonic_increasing
    assert str(df.index.tz) == "UTC"


def test_parse_trades_sorts_by_time():
    raw = [
        {"timestamp": 1700000005000, "price": 102.0, "amount": 1.0, "side": "buy"},
        {"timestamp": 1700000000000, "price": 100.0, "amount": 1.0, "side": "sell"},
    ]
    df = parse_trades(raw)
    assert df["price"].tolist() == [100.0, 102.0]


def test_parse_trades_empty():
    df = parse_trades([])
    assert list(df.columns) == ["price", "volume", "side"]
    assert df.shape == (0, 3)


class _FakeExchange:
    """ネットワークを使わずに ccxt 形式の応答を返すスタブ。"""

    def fetch_trades(self, symbol, since=None, limit=None):
        return [
            {"timestamp": 1700000000000, "price": 100.0, "amount": 1.0,
             "side": "buy", "id": "1", "info": {}},
            {"timestamp": 1700000002000, "price": 102.0, "amount": 2.0,
             "side": "sell", "id": "2", "info": {}},
        ]


def test_fetch_trades_with_injected_exchange():
    df = fetch_trades("BTC/JPY", exchange=_FakeExchange())
    assert df.shape == (2, 3)
    assert df["price"].tolist() == [100.0, 102.0]
    assert df["side"].tolist() == ["buy", "sell"]
