"""イベント駆動ポートフォリオバックテスト（指値・損切・利確・共有資金）。"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .filters import (
    load_blacklist,
    load_regime,
    load_sector_map,
    regime_allows,
    symbol_allowed,
)
from .indicators import add_indicators
from .regime_adapt import (
    entry_signal_for_row,
    lookup_regime_row,
    mode_uses_entry,
    mode_uses_exit,
    mode_uses_sizing,
    prepare_data_with_regime_indicators,
    resolve_regime_cfg,
)


def _is_futures(cfg: dict) -> bool:
    return cfg.get("instrument_type") == "futures"


def _mult(cfg: dict) -> float:
    return float(cfg.get("contract_multiplier", 1.0))


def _margin_per_contract(price: float, cfg: dict) -> float:
    return price * _mult(cfg) * float(cfg.get("margin_rate", 0.11))


def _position_mark_value(pos: "Position", mark: float, cfg: dict) -> float:
    if _is_futures(cfg):
        return pos.margin + pos.shares * (mark - pos.entry_price) * _mult(cfg)
    return pos.shares * mark


class Position:
    __slots__ = ("symbol", "shares", "entry_price", "entry_i",
                 "stop_price", "target_price", "cost_entry", "margin", "max_hold_days")

    def __init__(self, symbol: str, shares: float, entry_price: float,
                 entry_i: int, cfg: dict, *, max_hold_days: int | None = None):
        self.symbol = symbol
        self.shares = shares
        self.entry_price = entry_price
        self.entry_i = entry_i
        self.stop_price = entry_price * (1 - cfg["stop_loss_pct"])
        self.target_price = entry_price * (1 + cfg["take_profit_pct"])
        self.max_hold_days = int(
            max_hold_days if max_hold_days is not None else cfg["max_hold_days"]
        )
        if _is_futures(cfg):
            self.margin = _margin_per_contract(entry_price, cfg) * shares
            self.cost_entry = float(cfg.get("commission_per_contract", 0)) * shares
        else:
            self.margin = entry_price * shares
            self.cost_entry = entry_price * shares * cfg["commission_pct"]


def gap_adjust(level: float, open_: float, *, is_target: bool) -> float:
    """寄付が水準を飛び越えている場合は寄値約定にする。"""
    if is_target:
        return max(level, open_) if open_ >= level else level
    return min(level, open_) if open_ <= level else level


def limit_buy_fill(open_: float, limit_price: float, slippage_pct: float) -> float:
    """買い指値の約定価格。場中タッチ＝指値どおり（指値注文は指値より不利に約定しない）。

    寄付ギャップ（open < 指値）＝寄値執行にスリッページ、ただし指値を上限とする。
    """
    return min(open_ * (1 + slippage_pct), limit_price)


def resolve_exit(
    pos: Position, row: pd.Series, day_i: int, cfg: dict,
) -> tuple[Optional[float], Optional[str]]:
    """手仕舞い価格と理由。約定しなければ (None, None)。"""
    high, low, close, open_ = row["High"], row["Low"], row["Close"], row["Open"]
    hit_stop = low <= pos.stop_price
    hit_tgt = high >= pos.target_price
    held = day_i - pos.entry_i
    max_hold = pos.max_hold_days

    if hit_stop and hit_tgt:
        mode = cfg["same_day_priority"]
        if mode == "target":
            return gap_adjust(pos.target_price, open_, is_target=True), "target(same-day)"
        if mode == "auto":
            if abs(open_ - pos.target_price) <= abs(open_ - pos.stop_price):
                return gap_adjust(pos.target_price, open_, is_target=True), "target(auto)"
            return gap_adjust(pos.stop_price, open_, is_target=False), "stop(auto)"
        return gap_adjust(pos.stop_price, open_, is_target=False), "stop(same-day)"

    if hit_stop:
        return gap_adjust(pos.stop_price, open_, is_target=False), "stop"
    if hit_tgt:
        return gap_adjust(pos.target_price, open_, is_target=True), "target"
    if held >= max_hold:
        return close, "timeout"
    return None, None


def _in_universe(
    universe_mask: Optional[pd.DataFrame], sym: str, d: pd.Timestamp,
) -> bool:
    if universe_mask is None:
        return True
    if sym not in universe_mask.columns or d not in universe_mask.index:
        return False
    return bool(universe_mask.at[d, sym])


def _try_open(
    symbol: str, fill_price: float, day_i: int, cash: float,
    positions: dict, cfg: dict,
) -> Optional[Position]:
    if len(positions) >= cfg["max_positions"]:
        return None
    equity_now = cash + sum(
        _position_mark_value(p, p.entry_price, cfg) for p in positions.values()
    )
    risk_amount = equity_now * cfg["risk_per_trade_pct"]
    mult = _mult(cfg)
    if _is_futures(cfg):
        per_unit_risk = fill_price * cfg["stop_loss_pct"] * mult
    else:
        per_unit_risk = fill_price * cfg["stop_loss_pct"]
    if per_unit_risk <= 0:
        return None
    units = risk_amount / per_unit_risk

    if _is_futures(cfg):
        gross_now = sum(p.shares * p.entry_price * mult for p in positions.values())
        notional_per = fill_price * mult
    else:
        gross_now = sum(p.shares * p.entry_price for p in positions.values())
        notional_per = fill_price
    max_gross = equity_now * cfg["leverage"]
    room = max(0.0, max_gross - gross_now)
    units = min(units, room / notional_per if notional_per > 0 else 0)

    if _is_futures(cfg):
        margin_per = _margin_per_contract(fill_price, cfg)
        comm = float(cfg.get("commission_per_contract", 0))
        affordable = cash / (margin_per + comm) if margin_per + comm > 0 else 0
    else:
        affordable = cash / (fill_price * (1 + cfg["commission_pct"]))
    units = min(units, affordable)
    units = int(np.floor(units))
    if units < 1:
        if _is_futures(cfg) and affordable >= 1:
            units = 1
        else:
            return None
    hold_days = cfg["max_hold_days"] if mode_uses_exit(cfg.get("regime_adapt_mode")) else None
    return Position(symbol, units, fill_price, day_i, cfg, max_hold_days=hold_days)


def _open_cash_cost(pos: Position, cfg: dict) -> float:
    if _is_futures(cfg):
        return pos.margin + pos.cost_entry
    return pos.shares * pos.entry_price + pos.cost_entry


def _close_position(
    pos: Position, exit_price: float, exit_date: pd.Timestamp, exit_i: int,
    cfg: dict, trades: list, reason: str = "exit",
    limit_price: float | None = None,
) -> float:
    exit_price_eff = exit_price * (1 - cfg["slippage_pct"])
    if limit_price is not None:
        # 指値売り（利確）は指値より不利に約定しない：場中タッチ＝指値どおり、
        # 寄付ギャップ＝寄値−スリップ（下限は指値）。
        exit_price_eff = max(exit_price_eff, limit_price)
    if _is_futures(cfg):
        mult = _mult(cfg)
        pnl = pos.shares * (exit_price_eff - pos.entry_price) * mult
        pnl -= float(cfg.get("commission_per_contract", 0)) * pos.shares
        cash_return = pos.margin + pnl
        basis = pos.margin + pos.cost_entry
        ret_pct = pnl / basis if basis > 0 else 0.0
        qty_label = pos.shares
    else:
        gross = pos.shares * exit_price_eff
        commission = gross * cfg["commission_pct"]
        cash_return = gross - commission
        entry_notional = pos.shares * pos.entry_price
        pnl = cash_return - entry_notional - pos.cost_entry
        basis = entry_notional + pos.cost_entry
        ret_pct = pnl / basis if basis > 0 else 0.0
        qty_label = pos.shares

    trades.append({
        "symbol": pos.symbol,
        "entry_i": pos.entry_i,
        "exit_i": exit_i,
        "hold_days": exit_i - pos.entry_i,
        "entry_price": round(pos.entry_price, 2),
        "exit_price": round(exit_price_eff, 2),
        "shares": qty_label,
        "contracts": qty_label if _is_futures(cfg) else np.nan,
        "pnl": round(pnl, 2),
        "ret_pct": round(ret_pct, 4),
        "reason": reason,
        "exit_date": exit_date,
    })
    return cash_return


def _prepare_symbol_data(data: dict[str, pd.DataFrame], cfg: dict) -> dict[str, pd.DataFrame]:
    mode = cfg.get("regime_adapt_mode")
    if mode_uses_entry(mode):
        return prepare_data_with_regime_indicators(data, cfg)
    return {sym: add_indicators(df, cfg) for sym, df in data.items()}


def _regime_passes(d: pd.Timestamp, regime_df: pd.DataFrame, cfg: dict) -> bool:
    """ゲートモード時のみ高ボラをブロック。適応モードは全レジームで取引可。"""
    mode = cfg.get("regime_adapt_mode")
    if mode and mode != "gate":
        return True
    return regime_allows(d, regime_df, cfg)


def run_backtest(
    data: dict[str, pd.DataFrame],
    cfg: dict,
    universe_mask: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ポートフォリオ共有資金で回転手法をシミュレート。"""
    prepared = _prepare_symbol_data(data, cfg)
    all_dates = sorted(set().union(*[set(df["Date"]) for df in prepared.values()]))
    by_sym: dict[str, pd.DataFrame] = {}
    for sym, df in prepared.items():
        by_sym[sym] = df.set_index("Date")

    regime_df = load_regime(cfg)
    blacklist = load_blacklist(cfg)
    sector_map = load_sector_map(cfg)
    adapt_mode = cfg.get("regime_adapt_mode")

    cash = float(cfg["initial_capital"])
    positions: dict[str, Position] = {}
    # pending: limit_price, place_i, limit_valid_days
    pending: dict[str, tuple[float, int, int]] = {}
    trades: list[dict] = []
    equity_curve: list[tuple] = []

    def portfolio_value(on_date: pd.Timestamp) -> float:
        val = cash
        for sym, pos in positions.items():
            df = by_sym[sym]
            px = df.at[on_date, "Close"] if on_date in df.index else pos.entry_price
            val += _position_mark_value(pos, px, cfg)
        return val

    for day_i, d in enumerate(all_dates):
        regime_row = lookup_regime_row(d, regime_df)
        day_sizing_cfg = resolve_regime_cfg(
            cfg, regime_row, mode=adapt_mode, layer="sizing",
        )

        for sym in list(positions.keys()):
            pos = positions[sym]
            df = by_sym[sym]
            if d not in df.index or day_i <= pos.entry_i:
                continue
            exit_price, reason = resolve_exit(pos, df.loc[d], day_i, cfg)
            if exit_price is not None:
                lim_px = pos.target_price if reason.startswith("target") else None
                cash += _close_position(pos, exit_price, d, day_i, cfg, trades,
                                        reason=reason, limit_price=lim_px)
                del positions[sym]

        for sym in list(pending.keys()):
            limit_price, place_i, valid_days = pending[sym]
            if day_i - place_i > valid_days:
                del pending[sym]
                continue
            df = by_sym[sym]
            if d not in df.index:
                continue
            if sym in positions and not cfg["allow_multiple_per_symbol"]:
                del pending[sym]
                continue
            row = df.loc[d]
            if row["Low"] <= limit_price:
                fill = limit_buy_fill(float(row["Open"]), limit_price,
                                      cfg["slippage_pct"])
                open_cfg = resolve_regime_cfg(
                    cfg, regime_row, mode=adapt_mode, layer="exit",
                )
                if mode_uses_sizing(adapt_mode):
                    open_cfg = resolve_regime_cfg(
                        open_cfg, regime_row, mode=adapt_mode, layer="sizing",
                    )
                pos = _try_open(sym, fill, day_i, cash, positions, open_cfg)
                if pos is not None:
                    cash -= _open_cash_cost(pos, open_cfg)
                    positions[sym] = pos
                del pending[sym]

        for sym, df in by_sym.items():
            if d not in df.index:
                continue
            if sym in positions and not cfg["allow_multiple_per_symbol"]:
                continue
            if sym in pending:
                continue
            if not _in_universe(universe_mask, sym, d):
                continue
            if not _regime_passes(d, regime_df, cfg):
                continue
            if not symbol_allowed(sym, cfg, blacklist=blacklist, sector_map=sector_map):
                continue
            row = df.loc[d]
            lim, trend_ok = entry_signal_for_row(row, regime_row, cfg)
            if not trend_ok:
                continue
            if lim is not None:
                valid = int(day_sizing_cfg["limit_valid_days"])
                pending[sym] = (lim, day_i, valid)

        equity_curve.append((d, portfolio_value(d)))

    last_d = all_dates[-1]
    for sym in list(positions.keys()):
        pos = positions[sym]
        df = by_sym[sym]
        px = df.at[last_d, "Close"] if last_d in df.index else pos.entry_price
        cash += _close_position(pos, px, last_d, len(all_dates) - 1, cfg, trades,
                                reason="eod_final")
        del positions[sym]

    eq = pd.DataFrame(equity_curve, columns=["Date", "Equity"]).set_index("Date")
    tr = pd.DataFrame(trades)
    return eq, tr