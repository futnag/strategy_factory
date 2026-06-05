"""信用・空売り分析層（純関数）の検証。ネットワーク不要。"""
import numpy as np
import pandas as pd
import pytest

from invest_system.equities.margin import (
    _concat_dir,
    margin_imbalance,
    sector_short_ratio,
    short_interest,
)


def test_concat_dir_skips_empty_markers(tmp_path):
    d = tmp_path / "margin_weekly"
    d.mkdir()
    pd.DataFrame({"Date": [pd.Timestamp("2024-01-05")], "Code": ["100"],
                  "LongVol": [10.0], "ShrtVol": [2.0]}).to_parquet(d / "a.parquet")
    # 空マーカー（祝日キャッシュ）は無視される
    pd.DataFrame({"_empty": pd.Series([], dtype="bool")}).to_parquet(d / "b.parquet")
    out = _concat_dir("margin_weekly", base=str(tmp_path))
    assert list(out["Code"]) == ["100"]
    assert "_empty" not in out.columns


def test_margin_imbalance():
    weekly = pd.DataFrame({"Date": [pd.Timestamp("2024-01-05")], "Code": ["100"],
                           "LongVol": [30.0], "ShrtVol": [10.0]})
    out = margin_imbalance(weekly)
    assert out["margin_imbalance"].iloc[0] == pytest.approx((30 - 10) / 40)
    assert out["short_to_long"].iloc[0] == pytest.approx(10 / 30)


def test_short_interest_sums_reporters():
    # 同一 (CalcDate, Code) の複数報告者を合算
    pos = pd.DataFrame({
        "CalcDate": [pd.Timestamp("2024-01-04")] * 3,
        "Code": ["100", "100", "200"],
        "ShrtPosToSO": [0.01, 0.02, 0.05],
    })
    out = short_interest(pos).sort_values("Code").reset_index(drop=True)
    assert out.loc[out["Code"] == "100", "short_interest"].iloc[0] == pytest.approx(0.03)
    assert out.loc[out["Code"] == "200", "short_interest"].iloc[0] == pytest.approx(0.05)
    assert set(out.columns) == {"Date", "Code", "short_interest"}


def test_sector_short_ratio():
    ratio = pd.DataFrame({"Date": [pd.Timestamp("2024-01-04")], "S33": ["0050"],
                          "SellExShortVa": [600.0], "ShrtWithResVa": [300.0],
                          "ShrtNoResVa": [100.0]})
    out = sector_short_ratio(ratio)
    # (300+100)/(600+300+100) = 0.4
    assert out["sector_short_ratio"].iloc[0] == pytest.approx(0.4)


def test_empty_inputs():
    assert margin_imbalance(pd.DataFrame()).empty
    assert short_interest(pd.DataFrame()).empty
    assert sector_short_ratio(pd.DataFrame()).empty
