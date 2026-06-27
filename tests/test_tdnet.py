"""TDnet パーサ（純関数）のオフライン検証。"""
import pandas as pd
import pytest

from invest_system.data.sources.tdnet import (
    CANON_COLUMNS,
    classify_title,
    load_tdnet,
    normalize_code,
    parse_list_page,
    parse_page_links,
    update_tdnet,
)

SAMPLE_HTML = """
<table id="main-list-table">
<tr>
<td class="oddnew-L kjTime" noWrap>16:30</td>
<td class="oddnew-M kjCode" noWrap>330A0</td>
<td class="oddnew-M kjName" noWrap>Ｇ－ＴａｌｅｎｔＸ</td>
<td class="oddnew-M kjTitle" align="left"><a href="140120260617572928.pdf" target="_blank">自己株式取得に係る事項の決定に関するお知らせ</a></td>
<td class="oddnew-M kjXbrl" noWrap align="center"> </td>
<td class="oddnew-M kjPlace" noWrap align="left">東</td>
<td class="oddnew-R kjHistroy" align="left"></td>
</tr>
<tr>
<td class="evennew-L kjTime" noWrap>11:00</td>
<td class="evennew-M kjCode" noWrap>94010</td>
<td class="evennew-M kjName" noWrap>ＴＢＳＨＤ</td>
<td class="evennew-M kjTitle" align="left"><a href="140120260617572345.pdf">自己株式立会外買付取引(ToSTNeT-3)による自己株式の取得結果に関するお知らせ</a></td>
<td class="evennew-M kjXbrl" noWrap align="center"><div class="xbrl-mask"><a href="081220260617572345.zip">XBRL</a></div></td>
<td class="evennew-M kjPlace" noWrap align="left">東</td>
<td class="oddnew-R kjHistroy" align="left"></td>
</tr>
</table>
<div class="pager-O" onClick="pagerLink('I_list_001_20260617.html')">1</div>
<div class="pager-M" onClick="pagerLink('I_list_002_20260617.html')">2</div>
"""


def test_normalize_code():
    assert normalize_code("330A0") == "330A0"
    assert normalize_code("5867") == "58670"
    assert normalize_code(" 9401 ") == "94010"


def test_classify_title():
    assert "buyback_announce" in classify_title("自己株式取得に係る事項の決定に関するお知らせ")
    assert "buyback_result" in classify_title(
        "自己株式立会外買付取引(ToSTNeT-3)による自己株式の取得結果に関するお知らせ")
    assert classify_title("役員人事に関するお知らせ") == ""


def test_parse_list_page():
    df = parse_list_page(SAMPLE_HTML, "20260617")
    assert list(df.columns) == CANON_COLUMNS
    assert len(df) == 2
    assert df.iloc[0]["code"] == "330A0"
    assert "buyback_announce" in df.iloc[0]["event_tags"]
    assert df.iloc[0]["doc_id"] == "140120260617572928"
    assert df.iloc[1]["xbrl_url"].endswith(".zip")


def test_parse_page_links():
    pages = parse_page_links(SAMPLE_HTML)
    assert pages == ["I_list_001_20260617.html", "I_list_002_20260617.html"]


def test_update_tdnet_idempotent(tmp_path):
    def fake_fetch_day(d):
        return parse_list_page(SAMPLE_HTML, d)

    cache = tmp_path / "tdnet"
    rep1 = update_tdnet(str(cache), dates=["20260617"], fetch_day_fn=fake_fetch_day, verbose=False)
    assert rep1["fetched"] == 1 and rep1["rows"] == 2
    rep2 = update_tdnet(str(cache), dates=["20260617"], fetch_day_fn=fake_fetch_day, verbose=False)
    assert rep2["skipped"] == 1 and rep2["fetched"] == 0
    loaded = load_tdnet(str(cache))
    assert len(loaded) == 2