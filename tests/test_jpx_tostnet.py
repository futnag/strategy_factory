"""JPX ToSTNeT 超大口スクレイパ（parse / update の DI）を検証。

合成 xlsx を openpyxl で生成して純関数をオフラインで検証する（JPX 実ファイルは使わない）。
列構成・実例は 2026/06/24 取引分の検証結果（tostnet_monitoring_plan.md §0.3）に倣う。
"""
import io

import pandas as pd
import pytest
from openpyxl import Workbook

from invest_system.data.sources.jpx_tostnet import (
    CANON_COLUMNS,
    add_sector,
    daily_sector_summary,
    daily_summary,
    load_tostnet,
    parse_index_links,
    parse_trading_excel,
    update_tostnet,
)

# 検証済みの実ヘッダ（日英スラッシュ結合・9列）
HEADER = [
    "公表日/Publication_Date", "取引日/Trading_Date", "約定時刻/Trade_Time",
    "銘柄コード/Code", "銘柄名_日本語/Issue_Name_Japanese",
    "銘柄名_英語/Issue_Name_English", "価格_円/Price_yen",
    "売買高_株/Trading_Volume_shares", "売買代金_円/Trading_Value_yen",
]
# 実例（2026/06/24取引）。数値はカンマ区切り文字列、'285A' は英数字コード。
ROWS = [
    ["2026/06/25", "20260624", "15:31", "6861", "キーエンス",
     "KEYENCE CORPORATION", "76,539.1160", "78,000", "5,970,051,048"],
    ["2026/06/25", "20260624", "16:28", "285A", "キオクシアＨＤ",
     "Kioxia Holdings Corporation", "92,500.0000", "70,000", "6,475,000,000"],
]


def make_xlsx(data_rows, header=HEADER) -> bytes:
    """合成 ToSTNeT Excel を bytes で返す。"""
    wb = Workbook()
    ws = wb.active
    ws.append(header)
    for r in data_rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


INDEX_HTML = """
<table>
  <tr class="bg-even"><th>公表日</th><th>取引内容</th></tr>
  <tr><td>2026/06/25</td><td><a href="/markets/equities/tostnet/t13vrt000001ib45-att/20260624_ToSTNeT_Trading_Information.xlsx"><img src="x"></a></td></tr>
  <tr><td>2026/06/22</td><td><a href="/markets/equities/tostnet/t13vrt000001i1r5-att/20260619_ToSTNeT_Trading_Information.xlsx"><img src="x"></a></td></tr>
</table>
"""


# --- parse_index_links --------------------------------------------------
def test_parse_index_links_extracts_and_sorts_desc():
    links = parse_index_links(INDEX_HTML)
    assert links == [
        ("20260624",
         "/markets/equities/tostnet/t13vrt000001ib45-att/20260624_ToSTNeT_Trading_Information.xlsx"),
        ("20260619",
         "/markets/equities/tostnet/t13vrt000001i1r5-att/20260619_ToSTNeT_Trading_Information.xlsx"),
    ]


def test_parse_index_links_dedup_and_empty():
    assert parse_index_links("<p>該当取引なし</p>") == []
    dup = INDEX_HTML + INDEX_HTML
    assert len(parse_index_links(dup)) == 2          # 重複日は1つに


# --- parse_trading_excel ------------------------------------------------
def test_parse_trading_excel_schema_and_values():
    df = parse_trading_excel(make_xlsx(ROWS))
    assert list(df.columns) == CANON_COLUMNS
    assert len(df) == 2
    # Code は文字列（'285A' を含む）
    assert df["Code"].tolist() == ["6861", "285A"]
    # カンマ除去 → float
    assert df["Price"].tolist() == pytest.approx([76539.116, 92500.0])
    assert df["Volume"].tolist() == pytest.approx([78000.0, 70000.0])
    assert df["TurnoverValue"].tolist() == pytest.approx([5.970051048e9, 6.475e9])
    # 日付・時刻・名称・source
    assert (df["Date"] == pd.Timestamp("2026-06-24")).all()
    assert (df["PublicationDate"] == pd.Timestamp("2026-06-25")).all()
    assert df["TradeTime"].tolist() == ["15:31", "16:28"]
    assert df["CompanyNameEnglish"].iloc[0] == "KEYENCE CORPORATION"
    assert (df["source"] == "jpx_scrape").all()


def test_parse_trading_excel_empty_is_canonical():
    df = parse_trading_excel(make_xlsx([]))        # ヘッダのみ＝該当取引なし
    assert list(df.columns) == CANON_COLUMNS
    assert df.empty


def test_parse_trading_excel_handles_title_row_above_header():
    wb = Workbook()
    ws = wb.active
    ws.append(["ToSTNeT取引 超大口約定情報"])       # タイトル行
    ws.append(HEADER)
    for r in ROWS:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    df = parse_trading_excel(buf.getvalue())
    assert len(df) == 2 and df["Code"].tolist() == ["6861", "285A"]


# --- update_tostnet（DI・冪等） -----------------------------------------
def _fakes():
    xlsx = {"20260624": make_xlsx(ROWS), "20260619": make_xlsx([])}  # 0619は0件

    def fake_index(url=None):
        return INDEX_HTML

    def fake_excel(path):
        import re
        date = re.search(r"(\d{8})_ToSTNeT", path).group(1)
        return xlsx[date]

    return fake_index, fake_excel


def test_update_tostnet_idempotent(tmp_path):
    fake_index, fake_excel = _fakes()
    rep = update_tostnet(str(tmp_path), fetch_index=fake_index,
                         fetch_excel=fake_excel, pause=0, verbose=False)
    assert rep["available"] == 2
    assert rep["fetched"] == 2 and rep["empty"] == 1 and rep["rows"] == 2
    assert (tmp_path / "20260624.parquet").exists()
    assert (tmp_path / "20260619.parquet").exists()   # 空マーカーも残る

    # 2回目は全てスキップ（K水増しせず・再DLしない＝冪等）
    rep2 = update_tostnet(str(tmp_path), fetch_index=fake_index,
                          fetch_excel=fake_excel, pause=0, verbose=False)
    assert rep2["fetched"] == 0 and rep2["skipped"] == 2


def test_load_and_summary(tmp_path):
    fake_index, fake_excel = _fakes()
    update_tostnet(str(tmp_path), fetch_index=fake_index,
                   fetch_excel=fake_excel, pause=0, verbose=False)
    df = load_tostnet(str(tmp_path))
    assert len(df) == 2 and set(df["Code"]) == {"6861", "285A"}

    s = daily_summary(df)
    assert s.loc[pd.Timestamp("2026-06-24"), "trade_count"] == 2
    assert s.loc[pd.Timestamp("2026-06-24"), "unique_stocks"] == 2
    # 合計代金 ≈ (5.97+6.475) 十億円 = 124.45 億円
    assert s.loc[pd.Timestamp("2026-06-24"), "total_value_oku"] == pytest.approx(124.45, abs=0.05)
    assert s.loc[pd.Timestamp("2026-06-24"), "top_code"] == "285A"  # 6.475B が最大


def test_load_tostnet_missing_dir_returns_canonical(tmp_path):
    df = load_tostnet(str(tmp_path / "nope"))
    assert list(df.columns) == CANON_COLUMNS and df.empty


# --- セクター付与・集計（smap を DI） ------------------------------------
def test_add_sector_maps_and_buckets_unmapped():
    df = parse_trading_excel(make_xlsx(ROWS))
    smap = pd.Series({"6861": "電気機器"})            # 285A は未収載（ETF等扱い）
    out = add_sector(df, smap=smap)
    assert out.loc[out["Code"] == "6861", "sector"].iloc[0] == "電気機器"
    assert out.loc[out["Code"] == "285A", "sector"].iloc[0] == "その他/ETF等"


def test_add_sector_empty():
    df = parse_trading_excel(make_xlsx([]))
    out = add_sector(df, smap=pd.Series(dtype=object))
    assert "sector" in out.columns and out.empty


def test_daily_sector_summary_values():
    df = parse_trading_excel(make_xlsx(ROWS))
    smap = pd.Series({"6861": "電気機器", "285A": "電気機器"})
    s = daily_sector_summary(df, smap=smap)
    assert list(s.columns) == [
        "Date", "sector", "total_value_oku", "trade_count", "stock_count"]
    row = s[s["sector"] == "電気機器"].iloc[0]
    assert row["trade_count"] == 2 and row["stock_count"] == 2
    assert row["total_value_oku"] == pytest.approx(124.45, abs=0.05)
