"""イベント系シグナル（予想改訂・決算サプライズ）の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.equities.events import earnings_surprise, forecast_revision


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


def test_empty_inputs():
    assert forecast_revision(pd.DataFrame()).empty
    assert earnings_surprise(pd.DataFrame()).empty
