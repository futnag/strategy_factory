"""Phase 2 台帳・照合部品（T+1約定・円建て損益・キルスイッチ）の検証。ネット不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.production import (
    apply_actual_fills, drawdown_status, next_open_fills, yen_positions_pnl,
)


def _open_panel():
    idx = pd.date_range("2026-05-25", periods=8, freq="B")
    return pd.DataFrame({"A": [100, 101, 102, np.nan, 104, 105, 106, 107],
                         "B": [50, 51, 52, 53, 54, 55, 56, 57]},
                        index=idx, dtype="float64")


def test_next_open_fills_t_plus_one_and_holiday_rollover():
    op = _open_panel()
    after = op.index[2]                          # 決定日
    f = next_open_fills(["A", "B", "Z"], after, op).set_index("key")
    assert f.loc["B", "fill_date"] == op.index[3]          # 翌取引日
    assert f.loc["B", "fill_price"] == pytest.approx(53.0)
    assert f.loc["A", "fill_date"] == op.index[4]          # A は翌日欠損→繰延
    assert f.loc["A", "fill_price"] == pytest.approx(104.0)
    assert pd.isna(f.loc["Z", "fill_price"])               # 不明銘柄＝未約定


def test_next_open_fills_max_days_limit():
    op = _open_panel()
    op.loc[op.index[3]:, "A"] = np.nan                     # 以降ずっと欠損
    f = next_open_fills(["A"], op.index[2], op, max_days=3).set_index("key")
    assert pd.isna(f.loc["A", "fill_price"])


def test_yen_positions_pnl_signed_and_nan_safe():
    notional = pd.Series({"A": 100_000.0, "B": -50_000.0, "C": 30_000.0})
    rel = pd.Series({"A": 0.02, "B": 0.10})                # C は欠損→0扱い
    pnl = yen_positions_pnl(notional, rel)
    assert pnl == pytest.approx(100_000 * 0.02 - 50_000 * 0.10)


def test_drawdown_status_thresholds():
    idx = pd.date_range("2025-01-31", periods=4, freq="ME")
    ok = pd.Series([0.01, 0.02, -0.01, 0.01], index=idx)
    _, cur, st = drawdown_status(ok)
    assert st == "OK"
    alert = pd.Series([0.05, -0.09, 0.0, 0.0], index=idx)
    _, cur, st = drawdown_status(alert)
    assert "ALERT" in st and cur == pytest.approx(-0.09)
    stop = pd.Series([0.0, -0.10, -0.04, -0.03], index=idx)
    _, cur, st = drawdown_status(stop)
    assert "STOP" in st                                    # 累積 −16% 超


def test_drawdown_counts_from_initial_capital():
    # 初月から負け続けても「ピーク未更新だからDD=0」にならない（元本基準）
    idx = pd.date_range("2025-01-31", periods=2, freq="ME")
    _, cur, st = drawdown_status(pd.Series([-0.05, -0.05], index=idx))
    assert cur == pytest.approx(0.95 * 0.95 - 1.0)         # −9.75%
    assert "ALERT" in st


def test_apply_actual_fills_overrides_and_slippage():
    fills = pd.DataFrame({"key": ["A", "B"], "fill_date": pd.to_datetime(
        ["2026-06-01", "2026-06-01"]), "fill_price": [100.0, 50.0]})
    actual = pd.DataFrame({"key": ["A"], "fill_price": [100.5]})
    out, slip = apply_actual_fills(fills, actual)
    assert out.set_index("key").loc["A", "fill_price"] == pytest.approx(100.5)
    assert out.set_index("key").loc["B", "fill_price"] == pytest.approx(50.0)
    assert slip["A"] == pytest.approx(0.005)               # 実測スリッページ +50bp
    same, slip2 = apply_actual_fills(fills, None)
    assert slip2.empty and same.equals(fills)