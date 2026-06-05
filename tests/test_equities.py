"""日本株クロスセクション基盤（純関数）の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.equities.universe import filter_common_stocks, select_universe
from invest_system.equities.panel import (
    assemble_panel,
    forward_returns,
    trailing_momentum,
)
from invest_system.equities.fundamentals import point_in_time
from invest_system.equities.factors import (
    cross_sectional_zscore,
    market_cap,
    sector_neutralize,
)
from invest_system.equities.backtest import long_short_returns


# --- universe ---------------------------------------------------------------
def test_filter_common_stocks_excludes_etf():
    listed = pd.DataFrame({
        "Code": ["13010", "13060", "72030", "99840"],
        "Mkt": ["0111", "0109", "0111", "0112"],  # 1306 は ETF(0109)
    })
    out = filter_common_stocks(listed)
    assert set(out["Code"]) == {"13010", "72030", "99840"}


def test_select_universe_by_liquidity():
    listed = pd.DataFrame({"Code": ["A", "B", "C"], "Mkt": ["0111"] * 3})
    # B が最も売買代金大、C は観測不足
    turnover = pd.DataFrame({
        "A": [1.0, 1.0, 1.0],
        "B": [9.0, 9.0, 9.0],
        "C": [5.0, np.nan, np.nan],
    }, index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    uni = select_universe(listed, turnover, top_n=2, min_obs=2)
    assert uni == ["B", "A"]  # 流動性順、C は min_obs 未満で除外


# --- panel ------------------------------------------------------------------
def test_assemble_panel_and_dedup():
    snaps = {
        pd.Timestamp("2024-01-31"): pd.DataFrame(
            {"Code": ["A", "B", "A"], "AdjC": [100.0, 200.0, 101.0]}),
        pd.Timestamp("2024-02-29"): pd.DataFrame(
            {"Code": ["A", "B"], "AdjC": [110.0, 190.0]}),
    }
    panel = assemble_panel(snaps, "AdjC")
    assert list(panel.columns) == ["A", "B"]
    assert panel.loc[pd.Timestamp("2024-01-31"), "A"] == 101.0  # 重複は last
    assert panel.loc[pd.Timestamp("2024-02-29"), "B"] == 190.0


def test_forward_returns_alignment_no_lookahead():
    price = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 99.0]},
        index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    fr = forward_returns(price)
    assert fr.iloc[0]["A"] == pytest.approx(0.10)   # t0→t1
    assert fr.iloc[0]["B"] == pytest.approx(-0.10)
    assert fr.iloc[1]["A"] == pytest.approx(0.10)
    assert np.isnan(fr.iloc[2]["A"])                # 最終行は未実現


def test_trailing_momentum_skip():
    price = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1]},
        index=pd.date_range("2024-01-31", periods=4, freq="ME"))
    mom = trailing_momentum(price, lookback=2, skip=1)
    assert mom.iloc[2]["A"] == pytest.approx(0.10)  # 110/100-1
    assert mom.iloc[3]["A"] == pytest.approx(0.10)  # 121/110-1
    assert np.isnan(mom.iloc[1]["A"])


# --- fundamentals: ポイントインタイム（先読み排除）-------------------------
def test_point_in_time_no_lookahead_and_asof():
    fund = pd.DataFrame({
        "Code": ["100", "100"],
        "DiscDate": pd.to_datetime(["2024-01-15", "2024-04-20"]),
        "EPS": [10.0, 20.0],
    })
    rebal = pd.to_datetime(["2024-01-10", "2024-01-31", "2024-04-19", "2024-04-30"])
    pit = point_in_time(fund, rebal, ["EPS"], lag_days=1)
    eps = pit["EPS"]["100"]
    assert np.isnan(eps.loc["2024-01-10"])             # 開示前 → NaN
    assert eps.loc["2024-01-31"] == 10.0               # 1月開示が反映
    assert eps.loc["2024-04-19"] == 10.0               # 4/20開示は 4/19 時点で未反映
    assert eps.loc["2024-04-30"] == 20.0               # 4月開示が反映


def test_point_in_time_same_day_excluded_by_lag():
    fund = pd.DataFrame({
        "Code": ["100"], "DiscDate": pd.to_datetime(["2024-01-15"]), "EPS": [10.0]})
    rebal = pd.to_datetime(["2024-01-15"])            # 開示当日
    pit = point_in_time(fund, rebal, ["EPS"], lag_days=1)
    assert np.isnan(pit["EPS"]["100"].loc["2024-01-15"])  # lag=1 で当日は使わない


# --- factors ----------------------------------------------------------------
def test_market_cap():
    price = pd.DataFrame({"A": [10.0]}, index=[pd.Timestamp("2024-01-31")])
    shares = pd.DataFrame({"A": [1000.0]}, index=[pd.Timestamp("2024-01-31")])
    tr = pd.DataFrame({"A": [100.0]}, index=[pd.Timestamp("2024-01-31")])
    mc = market_cap(price, shares, tr)
    assert mc.loc[pd.Timestamp("2024-01-31"), "A"] == 9000.0  # 10*(1000-100)


def test_cross_sectional_zscore():
    df = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]},
                      index=[pd.Timestamp("2024-01-31")])
    z = cross_sectional_zscore(df, winsor=10)
    row = z.iloc[0]
    assert row["B"] == pytest.approx(0.0)
    assert row["A"] == pytest.approx(-1.224744871, rel=1e-6)
    assert row["C"] == pytest.approx(1.224744871, rel=1e-6)


def test_sector_neutralize_within_group_demean():
    df = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [10.0], "D": [20.0]},
                      index=[pd.Timestamp("2024-01-31")])
    sector = pd.Series({"A": "X", "B": "X", "C": "Y", "D": "Y"})
    out = sector_neutralize(df, sector)
    r = out.iloc[0]
    assert r["A"] == pytest.approx(-1.0) and r["B"] == pytest.approx(1.0)
    assert r["C"] == pytest.approx(-5.0) and r["D"] == pytest.approx(5.0)


# --- backtest ---------------------------------------------------------------
def test_long_short_returns_sign_and_costs():
    idx = [pd.Timestamp("2024-01-31")]
    factor = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0]}, index=idx)
    fwd = pd.DataFrame({"A": [-0.05], "B": [0.0], "C": [0.0], "D": [0.05]}, index=idx)
    # コスト無し：D ロング(+0.05) − A ショート(−0.05) = 0.10
    r0 = long_short_returns(factor, fwd, quantile=0.25, min_names=4)
    assert r0.iloc[0] == pytest.approx(0.10)
    # コスト有り：初回は全建てで回転=2、片道10bps → 0.002 控除
    r1 = long_short_returns(factor, fwd, quantile=0.25, costs_bps=10, min_names=4)
    assert r1.iloc[0] == pytest.approx(0.10 - 0.002)
