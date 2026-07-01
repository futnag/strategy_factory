"""レジームゲートと銘柄ブラックリスト（PIT・先読みなし）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from invest_system.data.feature_store import load_feature


def load_regime(cfg: dict) -> pd.DataFrame:
    """市場レジーム（data/features/regime.parquet）。欠損時は空 DataFrame。"""
    base = cfg.get("data_base", "data")
    path = cfg.get("regime_feature_path")
    if path:
        fp = Path(path)
        if not fp.exists():
            return pd.DataFrame()
        df = pd.read_parquet(fp)
    else:
        df = load_feature("regime", base=base)
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def regime_allows(date: pd.Timestamp, regime: pd.DataFrame, cfg: dict) -> bool:
    """高ボラ局面などで新規エントリーを抑制。"""
    if not cfg.get("use_regime_filter", False) or regime.empty:
        return True
    d = pd.Timestamp(date).normalize()
    if d not in regime.index:
        prior = regime.index[regime.index <= d]
        if prior.empty:
            return False
        d = prior[-1]
    row = regime.loc[d]
    vol_max = float(cfg.get("regime_vol_max", 1.0))
    if "vol_regime" in row.index and pd.notna(row["vol_regime"]):
        if float(row["vol_regime"]) > vol_max:
            return False
    if cfg.get("regime_require_trend_up", False):
        if row.get("trend_up", 0.0) < 0.5:
            return False
    return True


def load_sector_map(cfg: dict) -> dict[str, str]:
    """Code → S33Nm（listed マスタ・静的参照）。"""
    if not cfg.get("exclude_sectors") and not cfg.get("use_symbol_blacklist", False):
        return {}
    try:
        from invest_system.data.sources import jquants as jq
        listed = jq.fetch_listed_info()
        listed = listed.drop_duplicates("Code", keep="last")
        return dict(zip(listed["Code"].astype(str), listed["S33Nm"]))
    except Exception:  # noqa: BLE001
        return {}


def load_blacklist(cfg: dict) -> set[str]:
    """ブラックリスト銘柄コード集合。"""
    if not cfg.get("use_symbol_blacklist", False):
        return set()
    codes: set[str] = set()
    for c in cfg.get("symbol_blacklist", []) or []:
        codes.add(str(c))
    path = cfg.get("symbol_blacklist_path")
    if path and Path(path).exists():
        df = pd.read_csv(path, encoding="utf-8-sig")
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        codes.update(df[col].astype(str).tolist())
    return codes


def symbol_allowed(
    symbol: str,
    cfg: dict,
    *,
    blacklist: set[str],
    sector_map: dict[str, str],
) -> bool:
    sym = str(symbol)
    if sym in blacklist:
        return False
    exclude = cfg.get("exclude_sectors") or []
    if exclude and sector_map:
        sec = sector_map.get(sym)
        if sec in exclude:
            return False
    return True


def build_blacklist_from_trades(
    trades: pd.DataFrame,
    *,
    min_trades: int = 5,
    max_target_rate: float = 0.0,
    max_total_pnl: float = 0.0,
) -> pd.DataFrame:
    """トレード履歴からブラックリスト候補を生成（探索期間のみで使うこと）。"""
    if trades.empty:
        return pd.DataFrame(columns=["symbol", "n_trades", "total_pnl", "pct_target", "reason"])
    reason = trades.assign(
        base=trades["reason"].str.replace(r"\(.*\)", "", regex=True)
    )
    agg = trades.groupby("symbol").agg(
        n_trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
    )
    target_rate = (
        reason.groupby("symbol")["base"]
        .apply(lambda s: (s == "target").mean())
        .rename("pct_target")
    )
    out = agg.join(target_rate).reset_index()
    mask = (
        (out["n_trades"] >= min_trades)
        & (out["total_pnl"] <= max_total_pnl)
        & (out["pct_target"] <= max_target_rate)
    )
    bl = out.loc[mask].copy()
    bl["reason"] = "low_target_hist_loss"
    return bl.sort_values("total_pnl").reset_index(drop=True)