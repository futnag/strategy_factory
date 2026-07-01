"""パラメータグリッド探索（データは1回だけ読み込み）。"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import CONFIG
from .data import load_symbols
from .engine import run_backtest


def summarize(eq: pd.DataFrame, tr: pd.DataFrame, cfg: dict) -> dict:
    """1ランの主要指標を辞書で返す。"""
    init = cfg["initial_capital"]
    row: dict = {
        "external_key": cfg.get("external_key"),
        "instrument_label": cfg.get("instrument_label"),
        "entry_method": cfg["entry_method"],
        "atr_mult": cfg.get("atr_mult"),
        "bb_sigma": cfg.get("bb_sigma"),
        "trend_ma_period": cfg["trend_ma_period"],
        "limit_valid_days": cfg.get("limit_valid_days"),
        "take_profit_pct": cfg["take_profit_pct"],
        "stop_loss_pct": cfg["stop_loss_pct"],
        "max_hold_days": cfg["max_hold_days"],
        "use_trend_filter": cfg.get("use_trend_filter"),
        "use_ma_rise_filter": cfg.get("use_ma_rise_filter"),
        "ma_rise_periods": str(cfg.get("ma_rise_periods")),
        "ma_rise_days": cfg.get("ma_rise_days"),
        "use_ma_order_filter": cfg.get("use_ma_order_filter"),
        "ma_order_periods": str(cfg.get("ma_order_periods")),
        "n_trades": 0,
        "total_ret": np.nan,
        "cagr": np.nan,
        "max_dd": np.nan,
        "sharpe": np.nan,
        "win_rate": np.nan,
        "profit_factor": np.nan,
        "avg_hold_days": np.nan,
        "pct_timeout": np.nan,
        "pct_target": np.nan,
        "pct_stop": np.nan,
    }
    if eq.empty:
        return row

    final = eq["Equity"].iloc[-1]
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    row["total_ret"] = final / init - 1
    row["cagr"] = (final / init) ** (1 / years) - 1 if final > 0 else -1.0
    roll_max = eq["Equity"].cummax()
    row["max_dd"] = float((eq["Equity"] / roll_max - 1).min())

    daily = eq["Equity"].pct_change().dropna()
    if daily.std(ddof=0) > 0:
        row["sharpe"] = float(
            daily.mean() / daily.std(ddof=0) * np.sqrt(cfg["annualization_days"])
        )

    if tr.empty:
        return row

    n = len(tr)
    wins = tr[tr["pnl"] > 0]
    losses = tr[tr["pnl"] <= 0]
    row["n_trades"] = n
    row["win_rate"] = len(wins) / n
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    row["profit_factor"] = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )
    row["avg_hold_days"] = float(tr["hold_days"].mean())

    reasons = tr["reason"].str.replace(r"\(.*\)", "", regex=True)
    vc = reasons.value_counts(normalize=True)
    row["pct_timeout"] = float(vc.get("timeout", 0.0))
    row["pct_target"] = float(vc.get("target", 0.0))
    row["pct_stop"] = float(vc.get("stop", 0.0))
    return row


def run_single(
    cfg: dict,
    data: dict[str, pd.DataFrame],
    universe_mask: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eq, tr = run_backtest(data, cfg, universe_mask=universe_mask)
    return eq, tr, summarize(eq, tr, cfg)


def sweep_grid(
    grid: dict[str, Iterable],
    base_cfg: Optional[dict] = None,
    *,
    data: Optional[dict] = None,
    universe_mask: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """grid の直積を走らせ結果 DataFrame を返す。data 未指定なら1回読込。"""
    base = deepcopy(CONFIG if base_cfg is None else base_cfg)
    if data is None:
        data, universe_mask = load_symbols(base)
    if not data:
        raise RuntimeError("有効なデータがありません")

    keys = list(grid.keys())
    combos = list(product(*(grid[k] for k in keys)))
    rows: list[dict] = []
    for i, vals in enumerate(combos):
        cfg = deepcopy(base)
        for k, v in zip(keys, vals):
            cfg[k] = v
        if cfg["entry_method"] == "atr":
            label = f"atr{cfg['atr_mult']}_ma{cfg['trend_ma_period']}"
        else:
            label = f"bb{cfg['bb_sigma']}_ma{cfg['trend_ma_period']}"
        if verbose:
            print(f"[{i + 1}/{len(combos)}] {label} ...", flush=True)
        _, _, sm = run_single(cfg, data, universe_mask)
        rows.append(sm)

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["profit_factor", "sharpe", "total_ret"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)