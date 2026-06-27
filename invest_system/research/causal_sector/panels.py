"""セクター日次パネル構築とデータ品質レポート。

銘柄レベルの features/ と macro_extended を S33 業種別に集約する。
すべて ≤t の情報のみ（リターンは当日、B/M は月次開示を日次 ffill）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import SECTOR_PROFILES, SectorProfile

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

# モジュールキャッシュ（大きな wide parquet の再読込を避ける）
_CACHE: dict[str, object] = {}


def _cached_wide(name: str, path: Path) -> pd.DataFrame:
    if name not in _CACHE:
        _CACHE[name] = pd.read_parquet(path)
    return _CACHE[name]


def _sector_map() -> pd.Series:
    """Code → S33（文字列5桁）。"""
    from invest_system.data.sources import jquants as jq
    listed = jq.fetch_listed_info()
    if listed.empty or "S33" not in listed.columns:
        return pd.Series(dtype=object)
    return listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]


def _stationarize_macro(macro: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """価格→対数リターン、金利→差分、VIX→変化（diag_causal_pcmci と同規約）。"""
    out = pd.DataFrame(index=macro.index)
    price_like = {"usd_jpy", "sp500", "nasdaq", "wti_crude", "copper", "gold",
                  "silver", "eur_jpy", "aud_jpy", "gbp_jpy"}
    diff_like = {"japan_10y_yield", "us_10y_yield", "japan_policy_rate",
                 "us_federal_funds_rate", "vix", "vix_yf", "vix_dup4"}
    for c in cols:
        if c not in macro.columns:
            continue
        s = macro[c].astype(float)
        if c in price_like:
            out[c] = np.log(s).diff()
        elif c in diff_like:
            out[c] = s.diff()
        else:
            out[c] = s.pct_change()
    return out.replace([np.inf, -np.inf], np.nan)


def _aggregate_by_sector(wide: pd.DataFrame, smap: pd.Series,
                         how: str = "median", min_stocks: int = 5) -> pd.DataFrame:
    """Date×Code wide → Date×S33 集約（ベクトル化・全業種一括）。"""
    common = wide.columns.intersection(smap.index)
    sub = wide[common]
    sec = smap.reindex(common)
    # stack → groupby(Date, S33) で一括集約
    stacked = sub.stack(future_stack=True)
    codes = stacked.index.get_level_values(1)
    sectors = sec.reindex(codes).values
    stacked.index = pd.MultiIndex.from_arrays(
        [stacked.index.get_level_values(0), sectors], names=["Date", "S33"])
    agg = stacked.groupby(["Date", "S33"]).agg(how)
    counts = stacked.groupby(["Date", "S33"]).count()
    agg = agg.where(counts >= min_stocks)
    return agg.unstack("S33").sort_index()


def _sector_short_ratio(s33: str) -> pd.Series:
    """業種別空売り比率（jquants short_ratio）。"""
    p = DATA_ROOT / "jquants" / "short_ratio" / f"s33_{s33}.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    if "Date" not in df.columns:
        return pd.Series(dtype=float)
    col = "ShrtWithResVa"
    if col not in df.columns:
        cols = [c for c in df.columns if "Shrt" in c or "Va" in c]
        col = cols[0] if cols else None
    if col is None:
        return pd.Series(dtype=float)
    s = df.set_index("Date")[col].astype(float).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def _sector_book_to_market_monthly(smap: pd.Series) -> pd.DataFrame:
    """phase4b X の book_to_market を月次×S33 に集約し日次へ ffill。"""
    p = DATA_ROOT / "phase4b" / "X.parquet"
    if not p.exists():
        return pd.DataFrame()
    X = pd.read_parquet(p)
    if "book_to_market" not in X.columns:
        return pd.DataFrame()
    bm = X["book_to_market"].unstack("Code")
    common = bm.columns.intersection(smap.index)
    bm = bm[common]
    sec = smap.reindex(common)
    monthly = bm.T.groupby(sec).median().T
    monthly.index = pd.to_datetime(monthly.index)
    return monthly


def build_sector_daily_panel(s33: str, *,
                             start: Optional[str] = None,
                             end: Optional[str] = None,
                             profile: Optional[SectorProfile] = None) -> pd.DataFrame:
    """1業種の日次因果分析パネルを構築。

    列:
      RET      業種等加重中央値リターン（当日）
      VALUE    業種 B/M 中央値（月次→日次 ffill、≤t）
      VOL      20日実現ボラ（≤t）
      MOM      12-1モメンタム（≤t）
      SHORT_RATIO  業種空売り比率
      + プロファイルの外生ドライバ（定常化済み）
    """
    profile = profile or SECTOR_PROFILES.get(s33)
    if profile is None:
        raise ValueError(f"unknown S33: {s33}")

    smap = _sector_map()
    if smap.empty:
        raise RuntimeError("listed info / S33 unavailable")

    ret_path = DATA_ROOT / "features" / "returns.parquet"
    mom_path = DATA_ROOT / "features" / "momentum_12_1.parquet"
    vol_path = DATA_ROOT / "features" / "vol_20.parquet"
    macro_path = DATA_ROOT / "supplemental" / "macro_extended.parquet"

    if "_sector_ret" not in _CACHE:
        ret_w = _cached_wide("returns", ret_path)
        _CACHE["_sector_ret"] = _aggregate_by_sector(ret_w, smap)
    if mom_path.exists() and "_sector_mom" not in _CACHE:
        _CACHE["_sector_mom"] = _aggregate_by_sector(_cached_wide("mom", mom_path), smap)
    if vol_path.exists() and "_sector_vol" not in _CACHE:
        _CACHE["_sector_vol"] = _aggregate_by_sector(_cached_wide("vol", vol_path), smap)

    ret_all = _CACHE["_sector_ret"]
    ret_s = ret_all[s33] if s33 in ret_all.columns else None
    if ret_s is None:
        raise ValueError(f"no return data for S33={s33}")

    panel = pd.DataFrame(index=ret_s.index)
    panel["RET"] = ret_s

    bm_m = _sector_book_to_market_monthly(smap)
    if not bm_m.empty and s33 in bm_m.columns:
        panel["VALUE"] = bm_m[s33].reindex(panel.index, method="ffill")

    if "_sector_mom" in _CACHE and s33 in _CACHE["_sector_mom"].columns:
        panel["MOM"] = _CACHE["_sector_mom"][s33].reindex(panel.index)
    if "_sector_vol" in _CACHE and s33 in _CACHE["_sector_vol"].columns:
        panel["VOL"] = _CACHE["_sector_vol"][s33].reindex(panel.index)

    sr = _sector_short_ratio(s33)
    if not sr.empty:
        panel["SHORT_RATIO"] = sr.reindex(panel.index, method="ffill")

    if macro_path.exists():
        macro = pd.read_parquet(macro_path).sort_index()
        macro = macro[~macro.index.duplicated(keep="last")]
        drv = _stationarize_macro(macro, list(profile.drivers))
        panel = panel.join(drv[list(profile.drivers)], how="left")

    if start:
        panel = panel.loc[panel.index >= pd.Timestamp(start)]
    if end:
        panel = panel.loc[panel.index <= pd.Timestamp(end)]
    return panel


def sector_data_quality_report(s33s: Optional[list[str]] = None) -> pd.DataFrame:
    """セクター別データ品質サマリー（欠損率・銘柄数・期間）。"""
    smap = _sector_map()
    ret_w = _cached_wide("returns", DATA_ROOT / "features" / "returns.parquet")
    s33s = s33s or sorted(smap.dropna().unique())

    # 全業種パネルを一括構築（キャッシュ活用）
    ret_all = _aggregate_by_sector(ret_w, smap)
    bm_m = _sector_book_to_market_monthly(smap)
    rows = []
    for s33 in s33s:
        prof = SECTOR_PROFILES.get(s33)
        name = prof.name if prof else s33
        codes = smap[smap == s33].index
        n_stocks = len(codes.intersection(ret_w.columns))
        try:
            if s33 not in ret_all.columns:
                raise ValueError("no sector column")
            ret_s = ret_all[s33].dropna()
            miss_ret = ret_all[s33].isna().mean()
            miss_val = bm_m[s33].isna().mean() if (not bm_m.empty and s33 in bm_m.columns) else 1.0
            rows.append({
                "S33": s33, "name": name, "n_stocks": n_stocks,
                "start": ret_s.index.min(), "end": ret_s.index.max(), "n_days": len(ret_s),
                "RET_miss": miss_ret, "VALUE_miss": miss_val,
                "VOL_miss": np.nan, "drivers_miss": np.nan,
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"S33": s33, "name": name, "n_stocks": n_stocks,
                         "error": str(e)[:80]})
    return pd.DataFrame(rows)