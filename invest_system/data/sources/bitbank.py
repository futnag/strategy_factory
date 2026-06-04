"""bitbank データ取込（L1）。

統合ナレッジベース §3.2 / 設計書 D1（執行・研究データとも bitbank）。
ネットワーク取得は ccxt を遅延 import。生データ→標準スキーマ変換は純関数
``parse_trades`` として分離し、ネットワーク無しでテスト可能にする。

ライブ取得には ccxt が必要： pip install ccxt （または pip install ".[live]"）
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

_COLUMNS = ["price", "volume", "side"]


def parse_trades(raw: list) -> pd.DataFrame:
    """ccxt 形式の約定リスト（dictの配列）を標準 trades DataFrame に変換。

    各要素は {'timestamp': ms, 'price': float, 'amount': float, 'side': 'buy'|'sell'}。
    返り値：index=DatetimeIndex(UTC, name='timestamp'), columns=[price, volume, side]。
    時刻昇順にソート。
    """
    if not raw:
        empty_idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return pd.DataFrame(columns=_COLUMNS, index=empty_idx)
    df = pd.DataFrame(raw)
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    side = df["side"] if "side" in df.columns else pd.Series([None] * len(df))
    out = pd.DataFrame(
        {
            "price": df["price"].astype(float).to_numpy(),
            "volume": df["amount"].astype(float).to_numpy(),
            "side": side.to_numpy(),
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )
    return out.sort_index()


def fetch_trades(symbol: str = "BTC/JPY", since: Optional[int] = None,
                 limit: int = 1000, *, exchange=None) -> pd.DataFrame:
    """bitbank から約定を取得して標準 trades DataFrame を返す（ネットワーク）。

    ``exchange`` を渡せば差し替え可能（テスト/モック用 DI）。未指定なら
    ccxt.bitbank を生成する。
    """
    if exchange is None:
        try:
            import ccxt  # 遅延 import
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ccxt が必要です: pip install ccxt （または pip install \".[live]\"）"
            ) from e
        exchange = ccxt.bitbank()
    raw = exchange.fetch_trades(symbol, since=since, limit=limit)
    norm = [
        {"timestamp": t["timestamp"], "price": t["price"],
         "amount": t["amount"], "side": t.get("side")}
        for t in raw
    ]
    return parse_trades(norm)
