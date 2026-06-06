"""イベント系シグナル（予想改訂・決算サプライズ）の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.equities.events import (
    days_to_next_announcement, earnings_surprise, expected_announcement_month,
    forecast_revision,
)


def test_forecast_revision_per_code():
    fund = pd.DataFrame({
        "Code": ["100", "100", "100", "200"],
        "DiscDate": pd.to_datetime(["2024-02-01", "2024-05-01", "2024-08-01",
                                    "2024-05-01"]),
        "FEPS": [100.0, 110.0, 99.0, 50.0],
    })
    out = forecast_revision(fund).set_index(["Code", "DiscDate"])["fcst_revision"]
    # 100: 2回目 +10%、3回目 -10%。200は1開示のみ→改訂なし
    assert out.loc[("100", pd.Timestamp("2024-05-01"))] == pytest.approx(0.10)
    assert out.loc[("100", pd.Timestamp("2024-08-01"))] == pytest.approx(-0.10)
    assert ("200", pd.Timestamp("2024-05-01")) not in out.index


def test_earnings_surprise():
    fund = pd.DataFrame({
        "Code": ["100"], "DiscDate": pd.to_datetime(["2024-05-01"]),
        "EPS": [120.0], "FEPS": [100.0],
    })
    out = earnings_surprise(fund)
    assert out["surprise"].iloc[0] == pytest.approx(0.20)   # (120-100)/100


def test_expected_announcement_month():
    fund = pd.DataFrame({
        "Code": ["100", "100", "200"],
        "DiscDate": pd.to_datetime(["2023-05-15", "2024-05-15", "2023-08-10"]),
    })
    rebal = pd.to_datetime(["2024-04-30", "2024-07-31"])
    m = expected_announcement_month(fund, rebal)
    # 100は5月発表 → 4月末(翌月=5月)に発表見込みTrue、7月末(翌月=8月)はFalse
    assert bool(m.loc[pd.Timestamp("2024-04-30"), "100"]) is True
    assert bool(m.loc[pd.Timestamp("2024-07-31"), "100"]) is False
    # 200は8月発表 → 7月末(翌月=8月)にTrue
    assert bool(m.loc[pd.Timestamp("2024-07-31"), "200"]) is True


def test_days_to_next_announcement():
    fund = pd.DataFrame({
        "Code": ["100", "100"],
        "DiscDate": pd.to_datetime(["2024-02-01", "2024-05-01"]),  # 間隔90日
    })
    dates = pd.to_datetime(["2024-05-02", "2024-07-15"])
    p = days_to_next_announcement(fund, dates)["100"]
    # 次回予測=2024-05-01+90日=2024-07-30。残日数は 89 / 15。
    assert p.loc[pd.Timestamp("2024-05-02")] == pytest.approx(89)
    assert p.loc[pd.Timestamp("2024-07-15")] == pytest.approx(15)


def test_empty_inputs():
    assert forecast_revision(pd.DataFrame()).empty
    assert earnings_surprise(pd.DataFrame()).empty
    assert expected_announcement_month(pd.DataFrame(), [pd.Timestamp("2024-01-31")]).empty
    assert days_to_next_announcement(pd.DataFrame(), [pd.Timestamp("2024-01-31")]).empty
