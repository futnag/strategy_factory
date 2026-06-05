"""J-Quants（日本株）データ取得：V2 APIキー認証 ＋ ローカル Parquet キャッシュ。

J-Quants API V2 はトークン方式を廃し、ダッシュボード発行の APIキーを
リクエストヘッダー `x-api-key` で送る方式（auth_user/auth_refresh は不要）。
ベースURL https://api.jquants.com/v2。

APIキーは .env（gitignore 済）/環境変数 J_QUANTS_API_KEY から読み、コード・git・
ログに書かない。データは data/jquants/（gitignore 済）に Parquet でキャッシュして
再取得を避ける（レート制限・無料枠・容量への配慮）。無料プランは約12週の遅延あり。

V2 エンドポイント（ライブ確認済み）：
  上場銘柄  /equities/master      列: Code, CoName, CoNameEn, S17/S17Nm,
                                     S33/S33Nm, ScaleCat, Mkt/MktNm, ...
  日次株価  /equities/bars/daily   （date=YYYYMMDD, code=...）
            列: Date, Code, O/H/L/C, UL/LL, Vo(出来高), Va(売買代金),
                AdjFactor, AdjO/AdjH/AdjL/AdjC, AdjVo
  財務      /fins/summary         決算短信サマリー（無料枠でも取得可）
            列: DiscDate, Code, Sales, OP(営業利益), OdP(経常), NP(純利益),
                EPS, BPS, Eq(純資産), EqAR(自己資本比率), TA(総資産), ...
            ※ 詳細 BS/PL/CF の /fins/details は Premium 限定（本実装では不使用）
  レスポンスは {"data": [...], "pagination_key": ...}（列名は上記の略称）。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from ...config import get_env

_BASE = "https://api.jquants.com/v2"
_CACHE = Path("data/jquants")

# レート制限対応。公式上限（1分あたり）: Free=5, Light=60, Standard=120,
# Premium=500。Free は約12秒に1回。さらに「大幅超過して撃ち続けると約5分
# 完全遮断」されるため、呼び出し間隔の下限＋429時は長め(60秒〜)の待機で自衛する。
# 環境変数で調整可:
#   J_QUANTS_MIN_INTERVAL（秒, 既定12.5=Free安全側。Lightなら1, Standardなら0.5等）
#   J_QUANTS_MAX_RETRIES（429/5xx の最大リトライ回数）
_MIN_INTERVAL = float(get_env("J_QUANTS_MIN_INTERVAL", "12.5") or "12.5")
_MAX_RETRIES = int(get_env("J_QUANTS_MAX_RETRIES", "5") or "5")
_RETRY_CODES = {429, 500, 502, 503, 504}
_last_call = [0.0]


def _throttle() -> None:
    """直前の呼び出しから _MIN_INTERVAL 秒空ける（レート上限の遵守）。"""
    dt = time.monotonic() - _last_call[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = time.monotonic()

# 数値化する列（V2略称＋V1フルネームの両対応で頑健に）。
# V2略称はライブ /equities/bars/daily で確認済み:
#   O/H/L/C(始高安終) UL/LL(制限値幅) Vo(出来高) Va(売買代金)
#   AdjFactor(調整係数) AdjO/AdjH/AdjL/AdjC(調整後OHLC) AdjVo(調整後出来高)
_NUMERIC = [
    # V2 略称
    "O", "H", "L", "C", "UL", "LL", "Vo", "Va", "AdjFactor",
    "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo",
    # V1 フルネーム（後方互換）
    "Open", "High", "Low", "Close", "Volume", "TurnoverValue",
    "UpperLimit", "LowerLimit", "AdjustmentFactor",
    "AdjustmentOpen", "AdjustmentHigh", "AdjustmentLow",
    "AdjustmentClose", "AdjustmentVolume",
    # 財務サマリー /fins/summary（V2略称）: 売上 営業/経常/純利益 EPS BPS
    # 純資産Eq 自己資本比率EqAR 総資産TA（ライブで実在分を確定）
    "Sales", "OP", "OdP", "NP", "EPS", "BPS", "Eq", "EqAR", "TA",
    # 財務 V1 フルネーム（後方互換）
    "NetSales", "OperatingProfit", "OrdinaryProfit", "Profit",
    "EarningsPerShare", "BookValuePerShare", "Equity",
    "EquityToAssetRatio", "TotalAssets",
]
_DATE_COLS = ["Date", "DiscDate", "DisclosedDate", "CurrentPeriodEndDate",
              "CurrentFiscalYearEndDate"]


def _request(url: str, headers: dict, timeout: int = 60) -> dict:
    """GET＋JSON。429/5xx は指数バックオフで自動リトライ（ペーシング付き）。"""
    last_err = None
    for attempt in range(_MAX_RETRIES + 1):
        _throttle()
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in _RETRY_CODES and attempt < _MAX_RETRIES:
                ra = e.headers.get("Retry-After") if e.headers else None
                # 429 は小刻み再試行が逆効果（遮断延長）。60秒〜と長めに待つ。
                wait = (float(ra) if ra and str(ra).strip().isdigit()
                        else min(300.0, 60.0 * 2 ** attempt))
                time.sleep(wait)
                continue
            detail = e.read().decode("utf-8", "ignore")[:300]
            raise RuntimeError(f"J-Quants API HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            last_err = e
            if attempt < _MAX_RETRIES:
                time.sleep(min(30.0, 3.0 * 2 ** attempt))
                continue
            raise RuntimeError(f"J-Quants API network error: {e}") from e
    raise RuntimeError(f"J-Quants API failed after retries: {last_err}")


def _api_key(api_key: Optional[str] = None) -> str:
    key = api_key or get_env("J_QUANTS_API_KEY")
    if not key:
        raise RuntimeError(
            "J-Quants APIキーがありません。.env に J_QUANTS_API_KEY を設定してください"
            "（V2 はダッシュボード発行の APIキー方式）。")
    return key


def _get_paginated(path: str, params: dict, api_key: str) -> list:
    """V2 共通レスポンス {"data":[...], "pagination_key":...} をページ送りで全取得。"""
    headers = {"x-api-key": api_key}
    records: list = []
    pk = None
    while True:
        q = dict(params)
        if pk:
            q["pagination_key"] = pk
        url = f"{_BASE}{path}?{urllib.parse.urlencode(q)}"
        res = _request(url, headers)
        records.extend(res.get("data", []))
        pk = res.get("pagination_key")
        if not pk:
            break
        time.sleep(0.2)
    return records


# --- パース（純関数・ネットワーク不要、V1/V2 列名に頑健） ------------------
def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in _NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_daily_quotes(records: list) -> pd.DataFrame:
    """日次株価レコード→DataFrame。略称(C/AdjC..)・フル(Close..)双方を数値化。"""
    if not records:
        return pd.DataFrame(columns=["Date", "Code", "C", "AdjC"])
    return _coerce(pd.DataFrame(records))


def parse_listed_info(records: list) -> pd.DataFrame:
    """上場銘柄マスタ→DataFrame。"""
    if not records:
        return pd.DataFrame(columns=["Code"])
    return _coerce(pd.DataFrame(records))


def parse_statements(records: list) -> pd.DataFrame:
    """財務サマリー(/fins/summary)→DataFrame。V2略称(DiscDate/Sales/EPS..)を数値化。"""
    if not records:
        return pd.DataFrame(columns=["DiscDate", "Code"])
    return _coerce(pd.DataFrame(records))


def adjusted_close_col(df: pd.DataFrame) -> str:
    """調整後終値の列名を V2/V1 から解決（AdjC → AdjustmentClose → C → Close）。"""
    for c in ("AdjC", "AdjustmentClose", "C", "Close"):
        if c in df.columns:
            return c
    raise KeyError("adjusted/close column not found in daily quotes")


# --- 取得（Parquet キャッシュ付き） ---------------------------------------
def _cached(cache: Path, refresh: bool):
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    return None


def fetch_listed_info(api_key: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    cache = _CACHE / "equities_master.parquet"
    hit = _cached(cache, refresh)
    if hit is not None:
        return hit
    df = parse_listed_info(_get_paginated("/equities/master", {}, _api_key(api_key)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def fetch_daily_quotes(date: str, api_key: Optional[str] = None,
                       refresh: bool = False) -> pd.DataFrame:
    """指定日(YYYYMMDD or YYYY-MM-DD)の全銘柄日次株価。日付単位で Parquet キャッシュ。"""
    d = str(date).replace("-", "")
    cache = _CACHE / "daily" / f"{d}.parquet"
    hit = _cached(cache, refresh)
    if hit is not None:
        return hit
    df = parse_daily_quotes(
        _get_paginated("/equities/bars/daily", {"date": d}, _api_key(api_key)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def fetch_statements(code: Optional[str] = None, date: Optional[str] = None,
                     api_key: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    """財務サマリー（/fins/summary, code別 or date別）。キャッシュ付き。

    決算短信サマリー（売上/利益/EPS/BPS/純資産等）で、無料プランでも
    2年/12週遅延の範囲で取得可能。詳細BS/PL/CF(/fins/details)は Premium 限定
    のため本クライアントでは扱わない（標準的なバリュー/クオリティには不要）。
    code/date のどちらか一方は必須。
    """
    key = (code or (str(date).replace("-", "") if date else None) or "all")
    cache = _CACHE / "statements" / f"{key}.parquet"
    hit = _cached(cache, refresh)
    if hit is not None:
        return hit
    params = {}
    if code:
        params["code"] = code
    if date:
        params["date"] = str(date).replace("-", "")
    df = parse_statements(_get_paginated("/fins/summary", params, _api_key(api_key)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df
