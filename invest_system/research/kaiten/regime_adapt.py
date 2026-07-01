"""レジーム状態に応じた動的パラメータ調整（PIT・先読みなし）。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from .indicators import add_indicators

# vol_regime: 0=低 / 1=中 / 2=高（探索前の理論ベース初期値）
DEFAULT_VOL_PROFILES: dict[int, dict[str, Any]] = {
    0: {
        "bb_sigma": 2.5,
        "risk_per_trade_pct": 0.012,
        "max_positions": 10,
        "limit_valid_days": 2,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.07,
        "max_hold_days": 5,
    },
    1: {
        "bb_sigma": 2.5,
        "risk_per_trade_pct": 0.01,
        "max_positions": 8,
        "limit_valid_days": 1,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.07,
        "max_hold_days": 5,
    },
    2: {
        "bb_sigma": 3.0,
        "risk_per_trade_pct": 0.006,
        "max_positions": 4,
        "limit_valid_days": 1,
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.05,
        "max_hold_days": 3,
    },
}

REGIME_ADAPT_MODES = (
    "none",
    "gate",
    "sizing",
    "exit",
    "entry",
    "sizing_exit",
    "sizing_entry",
    "exit_entry",
    "full",
)


def lookup_regime_row(date: pd.Timestamp, regime: pd.DataFrame) -> pd.Series | None:
    """日付のレジーム行（欠損時は直近過去値）。"""
    if regime.empty:
        return None
    d = pd.Timestamp(date).normalize()
    if d not in regime.index:
        prior = regime.index[regime.index <= d]
        if prior.empty:
            return None
        d = prior[-1]
    return regime.loc[d]


def vol_bucket(regime_row: pd.Series | None) -> int:
    if regime_row is None or "vol_regime" not in regime_row.index:
        return 1
    v = regime_row["vol_regime"]
    if pd.isna(v):
        return 1
    return int(v)


def mode_uses_sizing(mode: str | None) -> bool:
    return mode in ("sizing", "sizing_exit", "sizing_entry", "full")


def mode_uses_exit(mode: str | None) -> bool:
    return mode in ("exit", "sizing_exit", "exit_entry", "full")


def mode_uses_entry(mode: str | None) -> bool:
    return mode in ("entry", "sizing_entry", "exit_entry", "full")


def resolve_regime_cfg(
    base_cfg: dict,
    regime_row: pd.Series | None,
    *,
    mode: str | None,
    layer: str,
) -> dict:
    """レジーム行からその層に必要なパラメータだけ上書きした cfg を返す。

    layer: "sizing" | "exit" | "entry"
    """
    if not mode or mode in ("none", "gate"):
        return base_cfg
    if layer == "sizing" and not mode_uses_sizing(mode):
        return base_cfg
    if layer == "exit" and not mode_uses_exit(mode):
        return base_cfg
    if layer == "entry" and not mode_uses_entry(mode):
        return base_cfg

    profiles = base_cfg.get("regime_vol_profiles") or DEFAULT_VOL_PROFILES
    bucket = vol_bucket(regime_row)
    prof = profiles.get(bucket, profiles.get(1, {}))
    out = deepcopy(base_cfg)
    for key in (
        "bb_sigma", "risk_per_trade_pct", "max_positions", "limit_valid_days",
        "stop_loss_pct", "take_profit_pct", "max_hold_days",
    ):
        if key in prof:
            out[key] = prof[key]

    if regime_row is not None and base_cfg.get("regime_trend_down_scale", True):
        if float(regime_row.get("trend_up", 1.0)) < 0.5:
            out["max_positions"] = max(1, int(out["max_positions"] * 0.5))
            out["risk_per_trade_pct"] = float(out["risk_per_trade_pct"]) * 0.5
    return out


def prepare_data_with_regime_indicators(
    data: dict[str, pd.DataFrame],
    cfg: dict,
) -> dict[str, pd.DataFrame]:
    """entry 層用: vol_regime 別の limit_next / trend_ok を列として付与。"""
    profiles = cfg.get("regime_vol_profiles") or DEFAULT_VOL_PROFILES
    out: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        base = df.copy()
        for bucket in sorted(profiles):
            sub = resolve_regime_cfg(
                cfg, pd.Series({"vol_regime": bucket, "trend_up": 1.0}),
                mode="entry", layer="entry",
            )
            ind = add_indicators(base, sub)
            suffix = f"_v{bucket}"
            base[f"limit_next{suffix}"] = ind["limit_next"]
            base[f"trend_ok{suffix}"] = ind["trend_ok"]
        if not mode_uses_entry(cfg.get("regime_adapt_mode")):
            ind = add_indicators(base, cfg)
            base["limit_next"] = ind["limit_next"]
            base["trend_ok"] = ind["trend_ok"]
        out[sym] = base
    return out


def entry_signal_for_row(
    row: pd.Series,
    regime_row: pd.Series | None,
    cfg: dict,
) -> tuple[float | None, bool]:
    """その日のエントリーシグナル（limit, trend_ok）。"""
    mode = cfg.get("regime_adapt_mode")
    if mode_uses_entry(mode):
        bucket = vol_bucket(regime_row)
        lim_col = f"limit_next_v{bucket}"
        ok_col = f"trend_ok_v{bucket}"
        lim = row.get(lim_col, row.get("limit_next"))
        ok = bool(row.get(ok_col, row.get("trend_ok", False)))
    else:
        lim = row.get("limit_next")
        ok = bool(row.get("trend_ok", False))
    if pd.isna(lim) or lim <= 0:
        return None, ok
    return float(lim), ok