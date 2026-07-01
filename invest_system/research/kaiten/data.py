"""Silver wide パネルから銘柄別 OHLCV と PIT ユニバースを構築。"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from invest_system.data.external import load_external_prices
from invest_system.data.sources import jquants as jq
from invest_system.data.store import load_wide
from invest_system.equities.universe import filter_common_stocks, point_in_time_universe

from .synthetic import synthetic_data


def _symbol_frame(
    code: str,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """wide 列から単一銘柄の long OHLCV を組立。"""
    if code not in close.columns:
        return None
    sub = pd.DataFrame({
        "Date": close.index,
        "Open": open_[code].values if code in open_.columns else pd.NA,
        "High": high[code].values if code in high.columns else pd.NA,
        "Low": low[code].values if code in low.columns else pd.NA,
        "Close": close[code].values,
        "Volume": volume[code].values if code in volume.columns else pd.NA,
    })
    sub = sub.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    return sub if not sub.empty else None


def load_from_wide(cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """processed/equities/wide から銘柄辞書と PIT ユニバースマスクを返す。"""
    base = cfg["data_base"]
    start, end = cfg.get("start_date"), cfg.get("end_date")

    open_ = load_wide("adj_open", start=start, end=end, base=base)
    high = load_wide("adj_high", start=start, end=end, base=base)
    low = load_wide("adj_low", start=start, end=end, base=base)
    close = load_wide("adj_close", start=start, end=end, base=base)
    volume = load_wide("volume", start=start, end=end, base=base)
    turnover = load_wide("turnover", start=start, end=end, base=base)

    if close.empty or turnover.empty:
        return {}, pd.DataFrame()

    codes = [str(c) for c in close.columns]
    if cfg.get("filter_common_stocks", True):
        listed = jq.fetch_listed_info()
        common = set(filter_common_stocks(listed)["Code"].astype(str))
        codes = [c for c in codes if c in common]
        turnover = turnover.reindex(columns=[c for c in turnover.columns if str(c) in common])

    universe_mask = point_in_time_universe(
        turnover,
        top_n=int(cfg["universe_top_n"]),
        lookback=int(cfg["universe_lookback"]),
        min_obs=int(cfg.get("universe_min_obs", 30)),
    )
    active = [str(c) for c in universe_mask.columns if universe_mask[c].any()]
    min_len = int(cfg["trend_ma_period"]) + 5

    data: dict[str, pd.DataFrame] = {}
    for code in active:
        df = _symbol_frame(code, open_, high, low, close, volume)
        if df is not None and len(df) >= min_len:
            data[code] = df

    return data, universe_mask


def load_from_external(cfg: dict) -> tuple[dict[str, pd.DataFrame], None]:
    """investers/ の単一先物・指数 OHLCV を銘柄辞書に変換（ユニバースなし）。"""
    key = str(cfg.get("external_key", "nk225_fut"))
    base = cfg["data_base"]
    start, end = cfg.get("start_date"), cfg.get("end_date")
    fields = ("open", "high", "low", "close", "volume")
    panels = {
        f: load_external_prices([key], field=f, start=start, end=end, base=base)
        for f in fields
    }
    close = panels["close"]
    if close.empty or key not in close.columns:
        return {}, None

    df = pd.DataFrame({
        "Date": close.index,
        "Open": panels["open"][key] if key in panels["open"].columns else pd.NA,
        "High": panels["high"][key] if key in panels["high"].columns else pd.NA,
        "Low": panels["low"][key] if key in panels["low"].columns else pd.NA,
        "Close": close[key],
        "Volume": panels["volume"][key] if key in panels["volume"].columns else pd.NA,
    })
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    min_len = int(cfg["trend_ma_period"]) + 5
    if len(df) < min_len:
        return {}, None
    return {key: df}, None


def load_symbols(cfg: dict) -> tuple[dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """CONFIG に従いデータを読み込む。戻り値は (symbol->OHLCV, universe_mask|None)。"""
    source = cfg.get("data_source", "wide")
    if source == "synthetic":
        return synthetic_data(), None
    if source == "external":
        return load_from_external(cfg)

    data, mask = load_from_wide(cfg)
    if not data and cfg.get("use_synthetic_if_empty", True):
        print("[INFO] wide データが空のため合成データで動作確認します。")
        return synthetic_data(), None
    return data, mask if not mask.empty else None