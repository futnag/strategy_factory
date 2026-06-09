"""外部クロスアセット・マクロデータの正準ローダ（マクロ特徴量用）。

二系統の追加データを、英語キーの wide パネルとして束ねて分析に供給する：
- 価格（正本）= `data/investers/`（investing.com 由来 OHLCV・2010〜・日本語ファイル名）
- マクロ     = `data/supplemental/macro_extended.parquet`（FRED/yfinance 由来）のうち
  価格と重複しない系列のみ（金利・政策金利・CPI・VIX・GBPJPY）。VIX 重複は1本に正規化。

設計は `equities/panel.load_daily_panel` / `fundamentals.load_fundamentals` を踏襲（純関数・
ネット不要・キャッシュ読取のみ）。暦日の系列は `asof_align` で JP リバランス日に ≤t-lag で
as-of 結合し、先読みなしで `AsOfView` の追加パネル/特徴量にできる（DP5・point_in_time と同思想）。

市場データは ToS によりコミットしない（`data/` は gitignore 済）。本モジュールは読取専用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# 価格（investers/・OHLCV）：正準キー → 実ファイル名（日本語・スペース込み）
_PRICE_FILES: dict[str, str] = {
    # FX
    "usdjpy": "USD_JPY 過去データ.parquet",
    "eurjpy": "EUR_JPY 過去データ.parquet",
    "audjpy": "AUD_JPY 過去データ.parquet",
    # JP 指数 / 先物
    "nk225": "日経平均株価 過去データ.parquet",
    "nk225_fut": "日経225先物 先物の過去データ.parquet",
    "topix": "TOPIX 過去データ.parquet",
    "topix_fut": "TOPIX先物 先物の過去データ.parquet",
    # US 指数
    "sp500": "S&P500 過去データ.parquet",
    "nasdaq_comp": "ナスダック総合 過去データ.parquet",
    "dow": "NYダウ平均株価 過去データ.parquet",
    # US CFD（Cash）
    "us500": "US 500 Cash 過去データ.parquet",
    "us_tech100": "US Tech 100 Cash 過去データ.parquet",
    "us30": "US 30 Cash 過去データ.parquet",
    # エネルギー
    "wti": "原油先物 WTI 過去データ.parquet",
    "natgas": "天然ガス先物 過去データ.parquet",
    # 金属
    "gold": "金先物 過去データ.parquet",
    "silver": "銀先物 過去データ.parquet",
    "copper": "銅先物 過去データ.parquet",
    "platinum": "白金先物 過去データ.parquet",
    "palladium": "パラジウム先物 過去データ.parquet",
    "nickel": "ニッケル先物 過去データ.parquet",
    "zinc": "亜鉛先物 過去データ.parquet",
    "aluminum": "アルミニウム 過去データ.parquet",
}

_PRICE_CLASS: dict[str, str] = {
    "usdjpy": "fx", "eurjpy": "fx", "audjpy": "fx",
    "nk225": "jp_index", "topix": "jp_index",
    "nk225_fut": "jp_future", "topix_fut": "jp_future",
    "sp500": "us_index", "nasdaq_comp": "us_index", "dow": "us_index",
    "us500": "us_cfd", "us_tech100": "us_cfd", "us30": "us_cfd",
    "wti": "energy", "natgas": "energy",
    "gold": "metal", "silver": "metal", "copper": "metal", "platinum": "metal",
    "palladium": "metal", "nickel": "metal", "zinc": "metal", "aluminum": "metal",
}

# マクロ（supplemental/macro_extended.parquet）：正準キー → 元カラム（価格重複は除外）
_MACRO_COLS: dict[str, str] = {
    "jp_10y": "japan_10y_yield",
    "jp_policy": "japan_policy_rate",
    "us_10y": "us_10y_yield",
    "us_ff": "us_federal_funds_rate",
    "jp_cpi": "japan_cpi",
    "us_cpi": "us_cpi",
    "us_core_cpi": "us_core_cpi",
    "vix": "vix",          # vix_yf / vix_dup4 は採らない＝重複正規化
    "gbpjpy": "gbp_jpy",
}


def _slice(panel: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is not None:
        panel = panel.loc[pd.Timestamp(start):]
    if end is not None:
        panel = panel.loc[:pd.Timestamp(end)]
    return panel


def load_external_prices(keys: Optional[Iterable[str]] = None, field: str = "close",
                         start=None, end=None, base: str = "data") -> pd.DataFrame:
    """investers/ の OHLCV から wide パネル（index=date, col=正準キー, 値=field）。

    keys=None で全銘柄。field は open/high/low/close/volume/change_pct。各日の実値＝先読みなし。
    """
    pdir = Path(base) / "investers"
    want = list(_PRICE_FILES) if keys is None else [str(k) for k in keys]
    series: dict[str, pd.Series] = {}
    for k in want:
        fn = _PRICE_FILES.get(k)
        if fn is None:
            continue
        fp = pdir / fn
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if field not in df.columns:
            continue
        s = pd.to_numeric(df[field], errors="coerce")
        s.index = pd.to_datetime(df.index).normalize().astype("datetime64[ns]")
        series[k] = s[~s.index.duplicated(keep="last")]
    if not series:
        return pd.DataFrame()
    return _slice(pd.DataFrame(series).sort_index(), start, end)


def load_macro(keys: Optional[Iterable[str]] = None, start=None, end=None,
               base: str = "data") -> pd.DataFrame:
    """supplemental/macro_extended.parquet から「価格と重複しないマクロ」を正準キーで返す。

    金利(jp_10y/jp_policy/us_10y/us_ff)・CPI(jp_cpi/us_cpi/us_core_cpi)・vix・gbpjpy。
    月次系列は元マージ時に日次 ffill 済。VIX 重複は vix 1本へ正規化。
    """
    fp = Path(base) / "supplemental" / "macro_extended.parquet"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    df.index = pd.to_datetime(df.index).normalize().astype("datetime64[ns]")
    want = list(_MACRO_COLS) if keys is None else [str(k) for k in keys]
    out: dict[str, pd.Series] = {}
    for k in want:
        src = _MACRO_COLS.get(k)
        if src and src in df.columns:
            out[k] = pd.to_numeric(df[src], errors="coerce")
    if not out:
        return pd.DataFrame()
    return _slice(pd.DataFrame(out).sort_index(), start, end)


def asof_align(ext, rebal_dates, lag_days: int = 1, ffill: bool = True) -> pd.DataFrame:
    """暦日の外部系列を JP リバランス日に as-of 結合（≤ t-lag の最新値・先読みなし）。

    各 t で「cutoff = t - lag_days 以前の最新観測」を採用（merge_asof backward）。US 系列は
    時差で t-1 に確実既知＝lag 既定1。月次CPI等も同経路で安全に日次化。返り値 index=rebal_dates。
    ffill=True で休日/週末の欠損を直近値で繰越（過去値のみ参照＝先読みなし）。
    """
    df = ext.to_frame() if isinstance(ext, pd.Series) else ext.copy()
    cols = list(df.columns)
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize().astype("datetime64[ns]")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if ffill:
        df = df.ffill()
    rebal = pd.DatetimeIndex(sorted(pd.to_datetime(list(rebal_dates)))) \
        .normalize().astype("datetime64[ns]")
    right = df.reset_index()
    right.columns = ["_d"] + cols
    left = pd.DataFrame({"asof": rebal})
    left["_cut"] = left["asof"] - pd.Timedelta(days=int(lag_days))
    m = pd.merge_asof(left.sort_values("_cut"), right.sort_values("_d"),
                      left_on="_cut", right_on="_d", direction="backward")
    return m.set_index("asof")[cols].reindex(rebal)


def list_external(base: str = "data") -> pd.DataFrame:
    """利用可能な正準キー一覧（kind/class/行数/期間/source）。MISSING で未取得を可視化。"""
    rows = []
    pdir = Path(base) / "investers"
    for k, fn in _PRICE_FILES.items():
        fp = pdir / fn
        if fp.exists():
            df = pd.read_parquet(fp)
            idx = pd.to_datetime(df.index)
            rows.append((k, "price", _PRICE_CLASS.get(k, ""), len(df),
                         idx.min().date() if len(idx) else None,
                         idx.max().date() if len(idx) else None,
                         ",".join(map(str, df.columns))))
        else:
            rows.append((k, "price", _PRICE_CLASS.get(k, ""), 0, None, None, "MISSING"))
    mfp = Path(base) / "supplemental" / "macro_extended.parquet"
    if mfp.exists():
        mdf = pd.read_parquet(mfp)
        midx = pd.to_datetime(mdf.index)
        for k, src in _MACRO_COLS.items():
            present = src in mdf.columns
            rows.append((k, "macro", "macro",
                         int(mdf[src].notna().sum()) if present else 0,
                         midx.min().date() if present and len(midx) else None,
                         midx.max().date() if present and len(midx) else None,
                         src if present else "MISSING"))
    return pd.DataFrame(rows, columns=["key", "kind", "class", "rows",
                                       "start", "end", "source"])
