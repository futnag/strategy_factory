"""セクター日足データの柔軟な読み込み。

対応形式:
  1. ロング形式 CSV/Parquet（date, sector, close, volume 等）
  2. セクター別ファイルのディレクトリ
  3. Date×sector の wide Parquet/CSV
  4. 本リポジトリの features/returns.parquet + S33 マスタ（既存ローダー）
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

_DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


def get_data_root() -> Path:
    """データルートを返す。

    worktree では gitignore の ``data/`` が main と共有されないため、
    環境変数 ``INVEST_DATA_ROOT`` で main 側のパスを指定できる。
    """
    env = os.environ.get("INVEST_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_DATA_ROOT


def set_data_root(path: Union[str, Path]) -> Path:
    """データルートを上書き（テスト・ノートブック用）。"""
    global DATA_ROOT
    DATA_ROOT = Path(path).expanduser().resolve()
    return DATA_ROOT


DATA_ROOT = get_data_root()

# カラム名の別名マッピング（柔軟対応）
_DATE_ALIASES = ("date", "Date", "datetime", "timestamp", "trade_date")
_SECTOR_ALIASES = ("sector", "Sector", "s33", "S33", "industry", "code")
_CLOSE_ALIASES = ("close", "Close", "adj_close", "AdjC", "price", "c")
_VOLUME_ALIASES = ("volume", "Volume", "Vo", "vol")
_HIGH_ALIASES = ("high", "High", "H")
_LOW_ALIASES = ("low", "Low", "L")


def _resolve_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    for a in aliases:
        if a in df.columns:
            return a
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _normalize_long(df: pd.DataFrame) -> pd.DataFrame:
    """ロング形式を統一スキーマに正規化。"""
    date_col = _resolve_col(df, _DATE_ALIASES)
    sector_col = _resolve_col(df, _SECTOR_ALIASES)
    close_col = _resolve_col(df, _CLOSE_ALIASES)
    if date_col is None or sector_col is None or close_col is None:
        raise ValueError(
            f"必須列が見つかりません（date/sector/close）。columns={list(df.columns)}"
        )
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col])
    out["sector"] = df[sector_col].astype(str).str.zfill(4)
    out["close"] = pd.to_numeric(df[close_col], errors="coerce")
    vol_col = _resolve_col(df, _VOLUME_ALIASES)
    if vol_col:
        out["volume"] = pd.to_numeric(df[vol_col], errors="coerce")
    hi = _resolve_col(df, _HIGH_ALIASES)
    lo = _resolve_col(df, _LOW_ALIASES)
    if hi:
        out["high"] = pd.to_numeric(df[hi], errors="coerce")
    if lo:
        out["low"] = pd.to_numeric(df[lo], errors="coerce")
    return out.dropna(subset=["date", "sector", "close"]).sort_values(["sector", "date"])


def load_sector_daily(
    source: Union[str, Path],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """セクター日足をロング形式で読み込む。

    Parameters
    ----------
    source : str | Path
        - 単一ファイル（.csv / .parquet）
        - ディレクトリ（各ファイルに sector 列、またはファイル名が sector コード）
        - "project" : 本リポジトリの features から S33 集約
    """
    source = Path(source) if str(source) != "project" else "project"

    if source == "project":
        return _load_from_project_features(start=start, end=end)

    if source.is_file():
        df = _read_table(source)
        if _resolve_col(df, _SECTOR_ALIASES) is None:
            # wide 形式の可能性
            if _resolve_col(df, _DATE_ALIASES) or df.index.name in _DATE_ALIASES:
                return _wide_to_long(df)
        return _normalize_long(df)

    if source.is_dir():
        return _load_from_directory(source)

    raise FileNotFoundError(f"データソースが見つかりません: {source}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"未対応の拡張子: {path.suffix}")


def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Date×sector wide → ロング形式。"""
    date_col = _resolve_col(df, _DATE_ALIASES)
    if date_col:
        w = df.set_index(date_col)
    else:
        w = df.copy()
    w.index = pd.to_datetime(w.index)
    stacked = w.stack(future_stack=True).reset_index()
    stacked.columns = ["date", "sector", "close"]
    stacked["sector"] = stacked["sector"].astype(str).str.zfill(4)
    return stacked.dropna(subset=["close"]).sort_values(["sector", "date"])


def _load_from_directory(directory: Path) -> pd.DataFrame:
    """ディレクトリ内の CSV/Parquet を結合。"""
    frames = []
    for p in sorted(directory.glob("*")):
        if p.suffix.lower() not in (".csv", ".parquet", ".pq"):
            continue
        df = _read_table(p)
        if _resolve_col(df, _SECTOR_ALIASES) is None:
            sector_code = p.stem.replace("s33_", "").replace("sector_", "")
            df = df.copy()
            df["sector"] = sector_code
        frames.append(_normalize_long(df))
    if not frames:
        raise ValueError(f"読み込み可能なファイルがありません: {directory}")
    return pd.concat(frames, ignore_index=True)


def _sector_map_from_listed() -> pd.Series:
    """Code → S33（listed info から取得）。

    worktree では ``data/jquants/`` が空のことがあるため、まず ``DATA_ROOT`` 直下の
    キャッシュ Parquet を読む（API キー不要）。
    """
    cache = DATA_ROOT / "jquants" / "equities_master.parquet"
    if cache.exists():
        listed = pd.read_parquet(cache)
    else:
        from invest_system.data.sources import jquants as jq
        listed = jq.fetch_listed_info()
    if listed.empty or "S33" not in listed.columns:
        return pd.Series(dtype=object)
    return listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]


def _aggregate_returns_by_sector(
    wide: pd.DataFrame, smap: pd.Series, *, min_stocks: int = 5,
) -> pd.DataFrame:
    """Date×Code wide → Date×S33 中央値集約。"""
    common = wide.columns.intersection(smap.index)
    sub = wide[common]
    sec = smap.reindex(common)
    stacked = sub.stack(future_stack=True)
    codes = stacked.index.get_level_values(1)
    sectors = sec.reindex(codes).values
    stacked.index = pd.MultiIndex.from_arrays(
        [stacked.index.get_level_values(0), sectors], names=["Date", "S33"],
    )
    agg = stacked.groupby(["Date", "S33"]).median()
    counts = stacked.groupby(["Date", "S33"]).count()
    agg = agg.where(counts >= min_stocks)
    return agg.unstack("S33").sort_index()


def _load_from_project_features(
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """features/returns.parquet から S33 等加重指数を構築（close 代理）。"""
    ret_path = DATA_ROOT / "features" / "returns.parquet"
    if not ret_path.exists():
        raise FileNotFoundError(
            f"プロジェクトデータが見つかりません: {ret_path}\n"
            "data/ に J-Quants データを配置するか、CSV/Parquet を直接指定してください。"
        )
    smap = _sector_map_from_listed()
    ret_w = pd.read_parquet(ret_path)
    sector_ret = _aggregate_returns_by_sector(ret_w, smap)
    # 累積指数（close 代理）: 各セクターの等加重リターンから構築
    close = (1.0 + sector_ret.fillna(0.0)).cumprod() * 100.0
    stacked = close.stack(future_stack=True).reset_index()
    stacked.columns = ["date", "sector", "close"]
    stacked["sector"] = stacked["sector"].astype(str).str.zfill(4)
    out = stacked.dropna(subset=["close"]).sort_values(["sector", "date"])
    if start:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out


def list_sectors(daily: pd.DataFrame) -> list[str]:
    """利用可能なセクターコード一覧。"""
    return sorted(daily["sector"].unique().tolist())


def generate_synthetic_sector_daily(
    sectors: Optional[list[str]] = None,
    n_days: int = 3000,
    *,
    seed: int = 42,
    start: str = "2010-01-04",
) -> pd.DataFrame:
    """テスト・デモ用の合成セクター日足（マルコフ・スイッチング風）。"""
    rng = np.random.default_rng(seed)
    sectors = sectors or ["3050", "3250", "3300", "3450", "3600", "3650", "5250", "7050"]
    dates = pd.bdate_range(start, periods=n_days, freq="B")
    rows = []
    for sector in sectors:
        # セクターごとに異なるボラ・レジーム持続
        base_vol = rng.uniform(0.008, 0.025)
        state = 0
        persist = rng.integers(60, 200)
        counter = 0
        log_price = np.log(100.0)
        for d in dates:
            if counter >= persist:
                state = 1 - state
                persist = rng.integers(40, 180)
                counter = 0
            counter += 1
            mu = 0.0002 if state == 0 else -0.0003
            vol = base_vol * (0.7 if state == 0 else 1.5)
            ret = rng.normal(mu, vol)
            log_price += ret
            rows.append({
                "date": d,
                "sector": sector,
                "close": np.exp(log_price),
                "volume": rng.integers(1_000_000, 50_000_000),
                "high": np.exp(log_price + abs(rng.normal(0, vol * 0.5))),
                "low": np.exp(log_price - abs(rng.normal(0, vol * 0.5))),
            })
    return pd.DataFrame(rows)