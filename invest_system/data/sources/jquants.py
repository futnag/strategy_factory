"""J-Quants（日本株）データ取得：認証＋ローカル Parquet キャッシュ。

認証情報は .env（gitignore 済）/環境変数から読み、コード・git・ログに書かない。
データは data/jquants/（gitignore 済）に Parquet でキャッシュし再取得を避ける
（API レート制限・無料枠・容量への配慮）。無料プランはデータに約12週の遅延あり。

認証フロー（J-Quants API v1）：
  リフレッシュトークン → /token/auth_refresh → idToken(24h) → 各データ取得の Bearer。
  リフレッシュトークンが無ければ mail+password → /token/auth_user で取得。
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

_BASE = "https://api.jquants.com/v1"
_CACHE = Path("data/jquants")
_NUMERIC_DQ = ["Open", "High", "Low", "Close", "Volume", "TurnoverValue",
               "AdjustmentOpen", "AdjustmentHigh", "AdjustmentLow",
               "AdjustmentClose", "AdjustmentVolume"]


def _request(url: str, method: str = "GET", headers: Optional[dict] = None,
             body: Optional[bytes] = None, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:500]
        raise RuntimeError(f"J-Quants API HTTP {e.code}: {detail}") from e


# --- 認証 -----------------------------------------------------------------
def get_id_token(refresh_token: Optional[str] = None, mail: Optional[str] = None,
                 password: Optional[str] = None, use_cache: bool = True) -> str:
    """idToken（24h 有効）を取得。ローカルにキャッシュして再認証を避ける。"""
    cache_file = _CACHE / ".id_token.json"
    if use_cache and cache_file.exists():
        try:
            meta = json.loads(cache_file.read_text())
            if time.time() - meta.get("ts", 0) < 23 * 3600:
                return meta["id_token"]
        except Exception:
            pass

    rt = refresh_token or get_env("J_QUANTS_REFRESH_TOKEN")
    if not rt:
        mail = mail or get_env("J_QUANTS_MAILADDRESS")
        password = password or get_env("J_QUANTS_PASSWORD")
        if not (mail and password):
            raise RuntimeError(
                "J-Quants 認証情報がありません。.env に J_QUANTS_REFRESH_TOKEN "
                "（または J_QUANTS_MAILADDRESS と J_QUANTS_PASSWORD）を設定してください。")
        res = _request(f"{_BASE}/token/auth_user", method="POST",
                       headers={"Content-Type": "application/json"},
                       body=json.dumps({"mailaddress": mail, "password": password}).encode())
        rt = res["refreshToken"]

    res = _request(
        f"{_BASE}/token/auth_refresh?refreshtoken={urllib.parse.quote(rt)}",
        method="POST", body=b"")
    id_token = res["idToken"]
    if use_cache:
        _CACHE.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"id_token": id_token, "ts": time.time()}))
    return id_token


def _get_paginated(path: str, params: dict, key: str, id_token: str) -> list:
    records: list = []
    pk = None
    while True:
        q = dict(params)
        if pk:
            q["pagination_key"] = pk
        url = f"{_BASE}{path}?{urllib.parse.urlencode(q)}"
        res = _request(url, headers={"Authorization": f"Bearer {id_token}"})
        records.extend(res.get(key, []))
        pk = res.get("pagination_key")
        if not pk:
            break
        time.sleep(0.2)
    return records


# --- パース（純関数・ネットワーク不要） -----------------------------------
def parse_daily_quotes(records: list) -> pd.DataFrame:
    """日次株価レコードを DataFrame に変換。調整後終値等を数値化。"""
    if not records:
        return pd.DataFrame(columns=["Date", "Code", "Close", "AdjustmentClose"])
    df = pd.DataFrame(records)
    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"])
    for col in _NUMERIC_DQ:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_listed_info(records: list) -> pd.DataFrame:
    """上場銘柄情報（コード・名称・業種・市場区分等）を DataFrame に変換。"""
    if not records:
        return pd.DataFrame(columns=["Code", "CompanyName"])
    df = pd.DataFrame(records)
    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def parse_statements(records: list) -> pd.DataFrame:
    """財務諸表レコードを DataFrame に変換。開示日等を日付化。"""
    if not records:
        return pd.DataFrame(columns=["DisclosedDate", "LocalCode"])
    df = pd.DataFrame(records)
    for col in ("DisclosedDate", "CurrentPeriodEndDate", "CurrentFiscalYearEndDate"):
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# --- 取得（Parquet キャッシュ付き） ---------------------------------------
def _cached(cache: Path, refresh: bool):
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    return None


def fetch_listed_info(id_token: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    cache = _CACHE / "listed_info.parquet"
    hit = _cached(cache, refresh)
    if hit is not None:
        return hit
    records = _get_paginated("/listed/info", {}, "info", id_token or get_id_token())
    df = parse_listed_info(records)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def fetch_daily_quotes(date: str, id_token: Optional[str] = None,
                       refresh: bool = False) -> pd.DataFrame:
    """指定日(YYYY-MM-DD)の全銘柄日次株価。日付単位で Parquet キャッシュ。"""
    cache = _CACHE / "daily_quotes" / f"{date}.parquet"
    hit = _cached(cache, refresh)
    if hit is not None:
        return hit
    records = _get_paginated("/prices/daily_quotes", {"date": date},
                             "daily_quotes", id_token or get_id_token())
    df = parse_daily_quotes(records)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def fetch_statements(code: Optional[str] = None, date: Optional[str] = None,
                     id_token: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    """財務諸表（code 別 or date 別）。キャッシュ付き。"""
    key = code or date or "all"
    cache = _CACHE / "statements" / f"{key}.parquet"
    hit = _cached(cache, refresh)
    if hit is not None:
        return hit
    params = {}
    if code:
        params["code"] = code
    if date:
        params["date"] = date
    records = _get_paginated("/fins/statements", params, "statements",
                             id_token or get_id_token())
    df = parse_statements(records)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df
