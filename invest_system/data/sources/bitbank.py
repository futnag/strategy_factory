"""bitbank データ取込（L1）。

統合ナレッジベース §3.2 / 設計書 D1（執行・研究データとも bitbank）。
ネットワーク取得は ccxt を遅延 import。生データ→標準スキーマ変換は純関数
``parse_trades`` として分離し、ネットワーク無しでテスト可能にする。

ライブ取得には ccxt が必要： pip install ccxt （または pip install ".[live]"）
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Optional

import pandas as pd

_COLUMNS = ["price", "volume", "side"]
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
_PUBLIC_BASE = "https://public.bitbank.cc"


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


# --- 公開ローソク足（APIキー不要） ---------------------------------------
def parse_candlesticks(ohlcv: list) -> pd.DataFrame:
    """bitbank candlestick の ohlcv 配列を標準 OHLCV DataFrame に変換（純関数）。

    各要素は [open, high, low, close, volume, timestamp_ms]（文字列可）。
    返り値：index=DatetimeIndex(UTC・tz-naive, name='timestamp'), columns=open/high/low/close/volume。
    （パイプライン全体が tz-naive UTC 前提のため、足データも tz-naive で返す。）
    """
    if not ohlcv:
        empty = pd.DatetimeIndex([], name="timestamp")
        return pd.DataFrame(columns=_OHLCV_COLUMNS, index=empty)
    ts = pd.to_datetime([int(row[5]) for row in ohlcv], unit="ms")
    out = pd.DataFrame(
        {
            "open": [float(row[0]) for row in ohlcv],
            "high": [float(row[1]) for row in ohlcv],
            "low": [float(row[2]) for row in ohlcv],
            "close": [float(row[3]) for row in ohlcv],
            "volume": [float(row[4]) for row in ohlcv],
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )
    return out.sort_index()


def fetch_candlesticks(pair: str = "btc_jpy", candle_type: str = "4hour",
                       periods=("2024", "2025"), *, base_url: str = _PUBLIC_BASE,
                       pause: float = 0.3, timeout: int = 30) -> pd.DataFrame:
    """bitbank 公開APIからローソク足を取得（APIキー不要）。

    candle_type が 4hour/8hour/12hour/1day/1week/1month のとき periods は 'YYYY'、
    1min/5min/15min/30min/1hour のとき 'YYYYMMDD' を渡す。複数 period を連結し、
    重複時刻を除いて時刻順に返す。市場データは公開のため認証不要。
    """
    frames = []
    for period in periods:
        url = f"{base_url}/{pair}/candlestick/{candle_type}/{period}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.load(resp)
        if payload.get("success") != 1:
            raise RuntimeError(f"bitbank API error ({period}): {payload}")
        ohlcv = payload["data"]["candlestick"][0]["ohlcv"]
        frames.append(parse_candlesticks(ohlcv))
        time.sleep(pause)
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="first")]
