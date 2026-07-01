"""回転手法バックテストの単体テスト（先読み排除・同日両ヒット・e2e）。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from invest_system.research.kaiten import (
    CONFIG,
    Position,
    add_indicators,
    build_blacklist_from_trades,
    filtered_equity_config,
    futures_config,
    futures_entry_methods,
    gap_adjust,
    load_from_external,
    load_regime,
    resolve_exit,
    run_backtest,
    synthetic_data,
)
from invest_system.research.kaiten.engine import limit_buy_fill
from invest_system.research.kaiten.filters import regime_allows, symbol_allowed
from invest_system.research.kaiten.regime_adapt import (
    entry_signal_for_row,
    resolve_regime_cfg,
    vol_bucket,
)
from invest_system.research.kaiten.metrics import performance_report, plot_equity
from invest_system.research.kaiten.run import main


def _base_cfg() -> dict:
    return deepcopy(CONFIG)


def _pos(entry: float = 100.0, cfg: dict | None = None) -> Position:
    cfg = cfg or _base_cfg()
    return Position("TEST", 10, entry, entry_i=0, cfg=cfg)


def test_gap_adjust_target_favorable():
    assert gap_adjust(110.0, 115.0, is_target=True) == 115.0


def test_gap_adjust_stop_adverse():
    assert gap_adjust(90.0, 85.0, is_target=False) == 85.0


def test_same_day_priority_changes_exit_price():
    """同日に損切・利確両方タッチしたバーで mode により約定価格が変わる。"""
    cfg = _base_cfg()
    pos = _pos(100.0, cfg)
    row = pd.Series({"Open": 100.0, "High": 110.0, "Low": 88.0, "Close": 102.0})

    cfg_stop = {**cfg, "same_day_priority": "stop"}
    px_stop, _ = resolve_exit(pos, row, day_i=5, cfg=cfg_stop)

    cfg_tgt = {**cfg, "same_day_priority": "target"}
    px_tgt, _ = resolve_exit(pos, row, day_i=5, cfg=cfg_tgt)

    assert px_stop is not None and px_tgt is not None
    assert px_stop != px_tgt
    assert px_stop == pytest.approx(pos.stop_price)
    assert px_tgt == pytest.approx(pos.target_price)


def test_same_day_auto_picks_closer_to_open():
    cfg = _base_cfg()
    pos = _pos(100.0, cfg)
    row = pd.Series({"Open": 101.0, "High": 110.0, "Low": 88.0, "Close": 102.0})

    cfg_auto = {**cfg, "same_day_priority": "auto"}
    px, reason = resolve_exit(pos, row, day_i=5, cfg=cfg_auto)
    assert reason == "target(auto)"
    assert px == pytest.approx(pos.target_price)


def test_no_exit_on_entry_day_unit():
    """resolve_exit 単体は当日反応するが、エンジンは entry_i 当日をスキップする。"""
    cfg = _base_cfg()
    pos = Position("X", 1, 100.0, entry_i=3, cfg=cfg)
    row = pd.Series({"Open": 100.0, "High": 120.0, "Low": 80.0, "Close": 105.0})
    assert resolve_exit(pos, row, day_i=3, cfg=cfg)[0] is not None


def test_engine_skips_exit_on_entry_day():
    """エンジンは entry_i と同じ day_i では手仕舞いしない。"""
    cfg = _base_cfg()
    cfg.update({
        "use_trend_filter": False,
        "trend_ma_period": 3,
        "entry_method": "bb",
        "bb_period": 3,
        "bb_sigma": 0.01,
        "take_profit_pct": 0.05,
        "stop_loss_pct": 0.50,
        "max_hold_days": 10,
        "initial_capital": 1_000_000,
        "risk_per_trade_pct": 0.5,
        "max_positions": 1,
    })
    dates = pd.bdate_range("2024-06-01", periods=6)
    # 2日目に深い low で指値約定、当日 high で利確圏 — 当日は決済されない
    df = pd.DataFrame({
        "Date": dates,
        "Open": [100, 100, 100, 100, 100, 100],
        "High": [101, 101, 108, 108, 108, 108],
        "Low":  [99, 99, 95, 100, 100, 100],
        "Close": [100, 100, 100, 107, 107, 107],
        "Volume": [1e6] * 6,
    })
    data = {"72030": df}
    eq, tr = run_backtest(data, cfg)
    if not tr.empty:
        # エントリー当日の日中イグジットは禁止（eod_final は最終日清算で hold_days=0 あり得る）
        intraday = tr[~tr["reason"].isin(["eod_final"])]
        if not intraday.empty:
            assert all(intraday["hold_days"] >= 1)


def test_ma_perfect_order_filter():
    """パーフェクトオーダー: 短期 MA > 長期 MA のときのみ trend_ok。"""
    n = 60
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = np.linspace(1000, 1500, n)  # 明確な上昇
    df = pd.DataFrame({
        "Date": dates,
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 1e6,
    })
    cfg = _base_cfg()
    cfg.update({
        "use_trend_filter": False,
        "use_ma_rise_filter": False,
        "use_ma_order_filter": True,
        "ma_order_periods": [5, 20],
        "trend_ma_period": 30,
        "entry_method": "bb",
    })
    out = add_indicators(df, cfg)
    assert out["trend_ok"].iloc[-1]


def test_ma_rise_filter_blocks_flat_trend():
    """MA 連続上昇フィルターは横ばい相場でエントリーを抑制する。"""
    n = 80
    dates = pd.bdate_range("2024-01-01", periods=n)
    # 前半上昇 → 後半フラット（MA 上昇が止まる）
    close = np.concatenate([
        np.linspace(1000, 1200, n // 2),
        np.full(n // 2, 1200.0),
    ])
    df = pd.DataFrame({
        "Date": dates,
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 1e6,
    })
    cfg = _base_cfg()
    cfg.update({
        "use_trend_filter": False,
        "use_ma_rise_filter": True,
        "use_ma_order_filter": False,
        "ma_rise_periods": [10],
        "ma_rise_days": 5,
        "trend_ma_period": 20,
        "entry_method": "bb",
        "bb_period": 10,
        "bb_sigma": 0.5,
    })
    out = add_indicators(df, cfg)
    # 上昇途中では True、フラット後半では False になるはず
    assert out["trend_ok"].iloc[30]
    assert not bool(out["trend_ok"].iloc[-1])


def test_entry_method_atr_and_bb_smoke():
    data = synthetic_data(n_symbols=2, n_days=400, seed=1)
    cfg = _base_cfg()
    cfg.update({"trend_ma_period": 60, "atr_mult": 1.2, "use_trend_filter": True})
    for method in ("atr", "bb"):
        c = {**cfg, "entry_method": method}
        eq, tr = run_backtest(data, c)
        assert not eq.empty


def test_exit_reasons_not_all_same():
    """合成データで stop/target/timeout が複数理由に分かれること。"""
    data = synthetic_data(n_symbols=5, n_days=1200, seed=7)
    cfg = _base_cfg()
    cfg.update({
        "trend_ma_period": 60,
        "atr_mult": 1.5,
        "atr_period": 10,
        "take_profit_pct": 0.05,
        "stop_loss_pct": 0.08,
        "max_hold_days": 5,
        "use_trend_filter": True,
    })
    _, tr = run_backtest(data, cfg)
    if tr.empty:
        pytest.skip("合成データでトレード無し — パラメータ要調整")
    reasons = set(tr["reason"].str.replace(r"\(.*\)", "", regex=True).unique())
    assert len(reasons) >= 2


def test_synthetic_e2e_outputs(tmp_path: Path):
    """合成データ end-to-end: メトリクス・CSV・PNG。"""
    cfg = _base_cfg()
    cfg.update({
        "data_source": "synthetic",
        "trend_ma_period": 60,
        "atr_mult": 1.5,
        "output_dir": str(tmp_path),
        "plot_path": str(tmp_path / "equity_curve.png"),
        "trades_csv_path": str(tmp_path / "trades.csv"),
    })
    eq, tr = main(cfg)
    assert eq is not None and not eq.empty
    assert (tmp_path / "equity_curve.png").exists()
    if not tr.empty:
        assert (tmp_path / "trades.csv").exists()


def test_performance_report_runs(capsys):
    cfg = _base_cfg()
    dates = pd.bdate_range("2024-01-01", periods=50)
    eq = pd.DataFrame({"Equity": np.linspace(1e6, 1.1e6, 50)}, index=dates)
    tr = pd.DataFrame({
        "pnl": [1000, -500, 2000],
        "hold_days": [2, 3, 4],
        "reason": ["target", "stop", "timeout"],
    })
    performance_report(eq, tr, cfg)
    out = capsys.readouterr().out
    assert "プロフィットF" in out
    assert "timeout" in out


@pytest.mark.skipif(
    not Path("data/investers/日経225先物 先物の過去データ.parquet").exists(),
    reason="先物 parquet 無し",
)
def test_futures_external_smoke():
    """日経225先物データで先物エンジンが完走すること。"""
    cfg = futures_config("nk225_fut")
    cfg.update({"start_date": "2020-01-01", "end_date": "2024-12-31"})
    data, mask = load_from_external(cfg)
    assert mask is None
    assert "nk225_fut" in data
    eq, tr = run_backtest(data, cfg)
    assert not eq.empty
    assert cfg["instrument_type"] == "futures"
    assert cfg["entry_method"] == "bb"
    assert cfg["bb_sigma"] == 3.0
    if not tr.empty:
        assert "contracts" in tr.columns


def test_regime_filter_blocks_high_vol():
    regime = pd.DataFrame({
        "vol_regime": [0.0, 1.0, 2.0],
        "trend_up": [1.0, 1.0, 1.0],
    }, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    cfg = {"use_regime_filter": True, "regime_vol_max": 1.0}
    assert regime_allows(pd.Timestamp("2024-01-01"), regime, cfg)
    assert regime_allows(pd.Timestamp("2024-01-02"), regime, cfg)
    assert not regime_allows(pd.Timestamp("2024-01-03"), regime, cfg)


def test_symbol_blacklist_and_sector_exclude():
    cfg = {
        "use_symbol_blacklist": True,
        "symbol_blacklist": ["11110"],
        "exclude_sectors": ["医薬品"],
    }
    sector = {"11110": "銀行業", "22220": "医薬品", "33330": "食料品"}
    assert not symbol_allowed("11110", cfg, blacklist={"11110"}, sector_map=sector)
    assert not symbol_allowed("22220", cfg, blacklist=set(), sector_map=sector)
    assert symbol_allowed("33330", cfg, blacklist=set(), sector_map=sector)


def test_build_blacklist_from_trades():
    tr = pd.DataFrame({
        "symbol": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "pnl": [-100, -100, -100, -100, -100, 50, 50, 50, 50, 50],
        "reason": ["timeout"] * 10,
    })
    bl = build_blacklist_from_trades(tr, min_trades=5, max_target_rate=0.0)
    assert list(bl["symbol"]) == ["A"]


@pytest.mark.skipif(
    not Path("data/features/regime.parquet").exists(),
    reason="regime feature 無し",
)
def test_filtered_config_smoke():
    cfg = filtered_equity_config()
    regime = load_regime(cfg)
    assert not regime.empty
    assert cfg["regime_adapt_mode"] == "gate"


def test_resolve_regime_cfg_sizing_high_vol():
    base = _base_cfg()
    base["regime_adapt_mode"] = "sizing"
    row = pd.Series({"vol_regime": 2.0, "trend_up": 1.0})
    out = resolve_regime_cfg(base, row, mode="sizing", layer="sizing")
    assert out["max_positions"] == 4
    assert out["risk_per_trade_pct"] == 0.006


def test_resolve_regime_cfg_trend_down_scales():
    base = _base_cfg()
    base["regime_adapt_mode"] = "sizing"
    row = pd.Series({"vol_regime": 1.0, "trend_up": 0.0})
    out = resolve_regime_cfg(base, row, mode="sizing", layer="sizing")
    assert out["max_positions"] == 4
    assert out["risk_per_trade_pct"] == 0.005


def test_entry_signal_for_row_uses_vol_bucket():
    cfg = _base_cfg()
    cfg["regime_adapt_mode"] = "entry"
    row = pd.Series({
        "limit_next_v0": 90.0, "trend_ok_v0": True,
        "limit_next_v2": 80.0, "trend_ok_v2": False,
    })
    lim, ok = entry_signal_for_row(row, pd.Series({"vol_regime": 0.0}), cfg)
    assert lim == 90.0 and ok is True
    lim2, ok2 = entry_signal_for_row(row, pd.Series({"vol_regime": 2.0}), cfg)
    assert lim2 == 80.0 and ok2 is False


def test_vol_bucket_defaults():
    assert vol_bucket(None) == 1
    assert vol_bucket(pd.Series({"vol_regime": 0.0})) == 0


def test_futures_config_entry_variants():
    assert futures_entry_methods("nk225_fut") == ["bb", "atr"]
    assert futures_entry_methods("topix_fut") == ["bb"]
    atr = futures_config("nk225_fut", entry="atr")
    assert atr["entry_method"] == "atr"
    assert atr["atr_mult"] == 2.0
    assert atr["max_hold_days"] == 3
    topix = futures_config("topix_fut")
    assert topix["bb_sigma"] == 2.5
    assert topix["limit_valid_days"] == 7


# ---------------------------------------------------------------- 指値約定とスリッページ
def test_limit_buy_fill_touch_no_slippage():
    """場中タッチの買い指値はスリッページなし＝指値どおり（指値より不利は構造的に不可能）。"""
    assert limit_buy_fill(105.0, 100.0, 0.001) == 100.0


def test_limit_buy_fill_gap_open_capped_at_limit():
    """寄付ギャップは寄値+スリップで約定、ただし指値を上限とする。"""
    assert limit_buy_fill(95.0, 100.0, 0.001) == pytest.approx(95.0 * 1.001)
    assert limit_buy_fill(99.95, 100.0, 0.01) == 100.0


def _fill_test_cfg() -> dict:
    cfg = _base_cfg()
    cfg.update({
        "use_trend_filter": False,
        "use_ma_rise_filter": False,
        "use_ma_order_filter": False,
        "entry_method": "bb",
        "bb_period": 3,
        "bb_sigma": 0.01,
        "take_profit_pct": 0.07,
        "stop_loss_pct": 0.05,
        "max_hold_days": 10,
        "risk_per_trade_pct": 0.5,
        "max_positions": 1,
        "leverage": 1.0,
    })
    return cfg


def _ohlc(dates, o, h, l, c):  # noqa: E741
    return pd.DataFrame({"Date": dates, "Open": o, "High": h, "Low": l,
                         "Close": c, "Volume": [1e6] * len(dates)})


def test_entry_touch_fills_at_limit_exactly():
    """場中タッチのエントリーは指値ちょうど（スリッページなし）で約定する。"""
    cfg = _fill_test_cfg()
    dates = pd.bdate_range("2024-06-03", periods=6)
    # day3 の指値 = MA3(100,100,100) − 0.01σ = 100.0。day4 は Open>指値・Low<指値＝場中タッチ。
    df = _ohlc(dates,
               o=[100, 100, 100, 101, 100, 100],
               h=[100, 100, 100, 101, 100, 100],
               l=[100, 100, 100, 99.5, 100, 100],
               c=[100, 100, 100, 100, 100, 100])
    _, tr = run_backtest({"X": df}, cfg)
    assert len(tr) == 1
    assert tr.iloc[0]["entry_price"] == pytest.approx(100.0, abs=1e-9)


def test_target_touch_exit_fills_at_target_exactly():
    """利確（指値売り）の場中タッチはスリッページ控除なしで target ちょうどに約定する。"""
    cfg = _fill_test_cfg()
    dates = pd.bdate_range("2024-06-03", periods=6)
    # day4: 寄付ギャップ約定（entry = 90×(1+slip)）→ day5: High が target を跨ぐ（Open は下）。
    df = _ohlc(dates,
               o=[100, 100, 100, 90, 92, 99],
               h=[100, 100, 100, 90.5, 97.5, 99],
               l=[100, 100, 100, 89, 91, 99],
               c=[100, 100, 100, 90, 92, 99])
    _, tr = run_backtest({"X": df}, cfg)
    assert len(tr) == 1
    t = tr.iloc[0]
    entry = 90.0 * (1 + cfg["slippage_pct"])
    assert t["entry_price"] == pytest.approx(entry, abs=0.01)
    assert str(t["reason"]).startswith("target")
    assert t["exit_price"] == pytest.approx(entry * 1.07, abs=0.01)


def test_stop_exit_keeps_slippage():
    """損切り（ストップ→成行相当）はスリッページを控除したまま＝保守を維持する。"""
    cfg = _fill_test_cfg()
    dates = pd.bdate_range("2024-06-03", periods=6)
    df = _ohlc(dates,
               o=[100, 100, 100, 90, 88, 99],
               h=[100, 100, 100, 90.5, 89, 99],
               l=[100, 100, 100, 89, 84, 99],
               c=[100, 100, 100, 90, 88, 99])
    _, tr = run_backtest({"X": df}, cfg)
    assert len(tr) == 1
    t = tr.iloc[0]
    entry = 90.0 * (1 + cfg["slippage_pct"])
    stop = entry * (1 - cfg["stop_loss_pct"])
    assert str(t["reason"]).startswith("stop")
    assert t["exit_price"] == pytest.approx(stop * (1 - cfg["slippage_pct"]), abs=0.01)


@pytest.mark.skipif(
    not Path("data/processed/equities/wide/adj_close.parquet").exists(),
    reason="wide パネル無し",
)
def test_load_wide_smoke():
    """実データ（wide）でクラッシュせず完走すること。"""
    cfg = _base_cfg()
    cfg.update({
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",
        "trend_ma_period": 60,
        "atr_mult": 1.5,
        "universe_top_n": 50,
        "output_dir": "output/kaiten_test",
        "plot_path": "output/kaiten_test/equity_curve.png",
        "trades_csv_path": "output/kaiten_test/trades.csv",
    })
    Path("output/kaiten_test").mkdir(parents=True, exist_ok=True)
    eq, _ = main(cfg)
    assert eq is not None and not eq.empty