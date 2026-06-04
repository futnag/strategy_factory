"""J-Quants V2 パース関数（ネットワーク不要）の検証。"""
import pandas as pd

from invest_system.data.sources.jquants import (
    adjusted_close_col,
    parse_daily_quotes,
    parse_listed_info,
    parse_statements,
)


def test_parse_daily_quotes_v2_abbreviated_fields():
    # V2 は略称列（O/H/L/C/V, AdjC）
    recs = [
        {"Date": "2024-01-04", "Code": "86970", "O": "100", "H": "110",
         "L": "90", "C": "105", "V": "1000", "AdjC": "105"},
        {"Date": "2024-01-05", "Code": "86970", "O": "105", "H": "120",
         "L": "100", "C": "115", "V": "2000", "AdjC": "115"},
    ]
    df = parse_daily_quotes(recs)
    assert df["C"].tolist() == [105.0, 115.0]
    assert df["AdjC"].tolist() == [105.0, 115.0]
    assert df["Code"].iloc[0] == "86970"
    assert df["Date"].iloc[0] == pd.Timestamp("2024-01-04")


def test_parse_daily_quotes_empty():
    assert parse_daily_quotes([]).empty


def test_adjusted_close_col_resolves_v2_and_v1():
    assert adjusted_close_col(pd.DataFrame(columns=["Date", "AdjC", "C"])) == "AdjC"
    assert adjusted_close_col(pd.DataFrame(columns=["Date", "AdjustmentClose"])) == "AdjustmentClose"
    assert adjusted_close_col(pd.DataFrame(columns=["Date", "Close"])) == "Close"


def test_parse_listed_info_passthrough():
    recs = [{"Code": "86970", "CompanyName": "日本取引所グループ", "Sector33Code": "7200"}]
    df = parse_listed_info(recs)
    assert df["Code"].iloc[0] == "86970"
    assert df["Sector33Code"].iloc[0] == "7200"


def test_parse_statements_dates():
    recs = [{"DisclosedDate": "2024-02-14", "LocalCode": "86970", "NetSales": "1000"}]
    df = parse_statements(recs)
    assert df["DisclosedDate"].iloc[0] == pd.Timestamp("2024-02-14")
    assert df["LocalCode"].iloc[0] == "86970"
