"""J-Quants パース関数（ネットワーク不要）の検証。"""
import pandas as pd

from invest_system.data.sources.jquants import (
    parse_daily_quotes,
    parse_listed_info,
    parse_statements,
)


def test_parse_daily_quotes_values_and_types():
    recs = [
        {"Date": "2024-01-04", "Code": "86970", "Open": "100", "High": "110",
         "Low": "90", "Close": "105", "Volume": "1000", "AdjustmentClose": "105"},
        {"Date": "2024-01-05", "Code": "86970", "Open": "105", "High": "120",
         "Low": "100", "Close": "115", "Volume": "2000", "AdjustmentClose": "115"},
    ]
    df = parse_daily_quotes(recs)
    assert df["Close"].tolist() == [105.0, 115.0]
    assert df["AdjustmentClose"].tolist() == [105.0, 115.0]
    assert df["Code"].iloc[0] == "86970"
    assert df["Date"].iloc[0] == pd.Timestamp("2024-01-04")


def test_parse_daily_quotes_empty():
    df = parse_daily_quotes([])
    assert df.empty


def test_parse_listed_info():
    recs = [{"Code": "86970", "CompanyName": "日本取引所グループ",
             "Sector33Code": "7200", "MarketCode": "0111"}]
    df = parse_listed_info(recs)
    assert df["Code"].iloc[0] == "86970"
    assert df["Sector33Code"].iloc[0] == "7200"


def test_parse_statements_dates():
    recs = [{"DisclosedDate": "2024-02-14", "LocalCode": "86970",
             "NetSales": "1000", "Profit": "200"}]
    df = parse_statements(recs)
    assert df["DisclosedDate"].iloc[0] == pd.Timestamp("2024-02-14")
    assert df["LocalCode"].iloc[0] == "86970"
