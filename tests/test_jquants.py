"""J-Quants V2 パース関数（ネットワーク不要）の検証。"""
import pandas as pd

from invest_system.data.sources.jquants import (
    adjusted_close_col,
    parse_daily_quotes,
    parse_listed_info,
    parse_statements,
)


def test_parse_daily_quotes_v2_abbreviated_fields():
    # V2 略称列（ライブ /equities/bars/daily で確認済み）:
    #   O/H/L/C, UL/LL, Vo(出来高), Va(売買代金), AdjFactor, AdjC, AdjVo
    recs = [
        {"Date": "2024-01-04", "Code": "86970", "O": "100", "H": "110",
         "L": "90", "C": "105", "UL": "0", "LL": "0", "Vo": "1000",
         "Va": "105000", "AdjFactor": "1.0", "AdjC": "105", "AdjVo": "1000"},
        {"Date": "2024-01-05", "Code": "86970", "O": "105", "H": "120",
         "L": "100", "C": "115", "UL": "0", "LL": "0", "Vo": "2000",
         "Va": "230000", "AdjFactor": "1.0", "AdjC": "115", "AdjVo": "2000"},
    ]
    df = parse_daily_quotes(recs)
    assert df["C"].tolist() == [105.0, 115.0]
    assert df["AdjC"].tolist() == [105.0, 115.0]
    # 出来高・売買代金・調整出来高も数値化される（旧 _NUMERIC では文字列のまま残った）
    assert df["Vo"].tolist() == [1000.0, 2000.0]
    assert df["Va"].tolist() == [105000.0, 230000.0]
    assert df["AdjVo"].tolist() == [1000.0, 2000.0]
    assert pd.api.types.is_numeric_dtype(df["Vo"])
    # Code は文字列のまま（先頭ゼロ・識別子を保持）
    assert df["Code"].iloc[0] == "86970"
    assert df["Date"].iloc[0] == pd.Timestamp("2024-01-04")


def test_parse_daily_quotes_empty():
    assert parse_daily_quotes([]).empty


def test_adjusted_close_col_resolves_v2_and_v1():
    assert adjusted_close_col(pd.DataFrame(columns=["Date", "AdjC", "C"])) == "AdjC"
    assert adjusted_close_col(pd.DataFrame(columns=["Date", "AdjustmentClose"])) == "AdjustmentClose"
    assert adjusted_close_col(pd.DataFrame(columns=["Date", "Close"])) == "Close"


def test_parse_listed_info_passthrough():
    # V2 略称（ライブ /equities/master で確認済み）: CoName, S33/S33Nm, Mkt ...
    recs = [{"Code": "86970", "CoName": "日本取引所グループ",
             "S33": "7200", "S33Nm": "その他金融業", "Mkt": "0111"}]
    df = parse_listed_info(recs)
    assert df["Code"].iloc[0] == "86970"
    assert df["S33"].iloc[0] == "7200"  # セクターコードは文字列のまま保持
    assert df["CoName"].iloc[0] == "日本取引所グループ"


def test_parse_statements_summary_v2_fields():
    # V2 /fins/summary 略称（ライブ確認済み）:
    #   DiscDate, Code, Sales, OP, NP, EPS, BPS, Eq, EqAR, TA, ShOutFY ...
    recs = [{"DiscDate": "2024-02-14", "Code": "86970", "Sales": "1000",
             "OP": "200", "NP": "150", "EPS": "75.5", "BPS": "1200",
             "Eq": "5000", "EqAR": "0.45", "TA": "11000", "ShOutFY": "2000000"}]
    df = parse_statements(recs)
    assert df["DiscDate"].iloc[0] == pd.Timestamp("2024-02-14")  # 日付化
    assert df["Code"].iloc[0] == "86970"                          # コードは文字列
    # 財務数値が float 化される（バリュー/クオリティ計算に必須）
    assert df["EPS"].iloc[0] == 75.5
    assert df["Eq"].iloc[0] == 5000.0
    assert df["NP"].iloc[0] == 150.0
    assert pd.api.types.is_numeric_dtype(df["Sales"])
    assert pd.api.types.is_numeric_dtype(df["EqAR"])
