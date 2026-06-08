"""市場系（信用・空売り）パーサの検証。ライブで確認した実フィールドを使用。"""
import json

import pandas as pd
import pytest

from invest_system.data.sources.jquants import parse_markets


def test_weekly_margin_fields_numeric():
    recs = [{"Date": "2016-06-10", "Code": "72030", "ShrtVol": "1055500",
             "LongVol": "12991700", "ShrtNegVol": "194200", "LongNegVol": "1598000",
             "ShrtStdVol": "861300", "LongStdVol": "11393700", "IssType": "2"}]
    df = parse_markets(recs)
    assert df["Date"].iloc[0] == pd.Timestamp("2016-06-10")
    assert df["ShrtVol"].iloc[0] == 1055500.0
    assert df["LongVol"].iloc[0] == 12991700.0
    assert pd.api.types.is_numeric_dtype(df["LongStdVol"])
    assert df["Code"].iloc[0] == "72030"        # コードは文字列
    assert df["IssType"].iloc[0] == "2"          # 区分は文字列


def test_short_ratio_fields():
    recs = [{"Date": "2026-05-29", "S33": "0050", "SellExShortVa": "2852953000",
             "ShrtWithResVa": "1325645350", "ShrtNoResVa": "116463500"}]
    df = parse_markets(recs)
    assert df["S33"].iloc[0] == "0050"           # 業種コードは文字列
    assert df["SellExShortVa"].iloc[0] == 2852953000.0
    assert pd.api.types.is_numeric_dtype(df["ShrtNoResVa"])


def test_short_positions_fields():
    recs = [{"DiscDate": "2018-03-26", "CalcDate": "2018-03-22", "Code": "72030",
             "SSName": "Barclays Bank PLC", "ShrtPosToSO": "0.0123",
             "ShrtPosShares": "1500000", "ShrtPosUnits": "15000",
             "PrevRptDate": "2018-03-20", "PrevRptRatio": "0.0100", "Notes": "-"}]
    df = parse_markets(recs)
    assert df["DiscDate"].iloc[0] == pd.Timestamp("2018-03-26")
    assert df["CalcDate"].iloc[0] == pd.Timestamp("2018-03-22")
    assert df["ShrtPosToSO"].iloc[0] == pytest.approx(0.0123)
    assert df["SSName"].iloc[0] == "Barclays Bank PLC"   # 文字列保持


def test_margin_alert_nested_pubreason_jsonified():
    # margin-alert の PubReason は入れ子dict → Parquet保存可能にJSON文字列化される
    recs = [{"PubDate": "2026-05-29", "Code": "13250", "AppDate": "2026-05-28",
             "PubReason": {"Restricted": "0", "DailyPublication": "0",
                           "PrecautionByJSF": "1"},
             "ShrtOut": "10000", "ShrtOutRatio": "0.05", "SLRatio": "1.2"}]
    df = parse_markets(recs)
    assert df["PubDate"].iloc[0] == pd.Timestamp("2026-05-29")
    assert df["ShrtOut"].iloc[0] == 10000.0
    assert df["SLRatio"].iloc[0] == pytest.approx(1.2)
    # PubReason は str 化され、JSONとして復元可能
    val = df["PubReason"].iloc[0]
    assert isinstance(val, str)
    assert json.loads(val)["PrecautionByJSF"] == "1"


def test_index_bars_fields():
    recs = [{"Date": "2026-05-29", "Code": "0000", "O": "3917.83", "H": "3984.58",
             "L": "3914.77", "C": "3957.17"}]
    df = parse_markets(recs)
    assert df["Code"].iloc[0] == "0000"                 # 指数コードは文字列
    assert df["C"].iloc[0] == pytest.approx(3957.17)
    assert df["Date"].iloc[0] == pd.Timestamp("2026-05-29")


def test_investor_types_fields():
    recs = [{"PubDate": "2026-05-01", "StDate": "2026-04-20", "EnDate": "2026-04-24",
             "Section": "TSEPrime", "FrgnSell": "100", "FrgnBuy": "150",
             "FrgnBal": "50", "IndBuy": "30", "PropBal": "-5"}]
    df = parse_markets(recs)
    assert df["Section"].iloc[0] == "TSEPrime"          # 区分は文字列
    assert df["FrgnBal"].iloc[0] == 50.0                # 海外差引（数値）
    assert df["StDate"].iloc[0] == pd.Timestamp("2026-04-20")
    assert df["EnDate"].iloc[0] == pd.Timestamp("2026-04-24")
    assert pd.api.types.is_numeric_dtype(df["FrgnBuy"])


def test_index_options_fields():
    recs = [{"Date": "2026-05-29", "Code": "131060018", "O": "0", "C": "5.5",
             "Strike": "38000", "CM": "2026-06", "PCDiv": "1", "Settle": "5.0",
             "Theo": "5.2", "UnderPx": "38500", "IV": "0.21", "OI": "1804",
             "LTD": "2026-06-11", "SQD": "2026-06-12"}]
    df = parse_markets(recs)
    assert df["Strike"].iloc[0] == 38000.0
    assert df["IV"].iloc[0] == pytest.approx(0.21)        # インプライドボラ（数値）
    assert df["UnderPx"].iloc[0] == 38500.0               # 原資産（NK225）
    assert df["Code"].iloc[0] == "131060018"              # 契約コードは文字列
    assert df["PCDiv"].iloc[0] == "1"                     # Put/Call区分は文字列
    assert df["LTD"].iloc[0] == pd.Timestamp("2026-06-11")  # 最終取引日


def test_parse_markets_empty():
    assert parse_markets([]).empty
