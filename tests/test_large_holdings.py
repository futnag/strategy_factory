"""大量保有（府令060）本体抽出と C2 ユニバース判定の検証。ネットワーク不要。"""
import csv
import io

from invest_system.equities import large_holdings as lh


def _csv(rows: list[tuple]) -> str:
    """EDINET XBRL→CSV と同じ 9 列 TSV を生成（要素ID, 項目名, …, 値）。"""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t")
    w.writerow(["要素ID", "項目名", "コンテキストID", "相対年度", "連結・個別",
                "期間・時点", "ユニットID", "単位", "値"])
    for eid, val in rows:
        w.writerow([eid, "名", "c", "-", "-", "時点", "-", "", val])
    return buf.getvalue()


def test_report_class_from_title():
    # 訂正は「大量保有報告書」語を含むため最優先で判定する
    assert lh.report_class("大量保有報告書") == "new"
    assert lh.report_class("変更報告書No.4") == "change"
    assert lh.report_class("変更報告書 No.12") == "change"
    assert lh.report_class("大量保有報告書（令和3年12月28日付け訂正報告書の添付）") == "correction"
    assert lh.report_class("") == "unknown"


def test_is_special_report():
    # 一般(01)=False / 特例(02,03)=True。新規/変更の末尾とは独立に先頭で判定。
    assert lh.is_special_report("010000") is False
    assert lh.is_special_report("010002") is False
    assert lh.is_special_report("030000") is True
    assert lh.is_special_report("020002") is True
    assert lh.is_special_report("090001") is False


def test_fields_lvh_new_activist():
    purpose = ("純投資、及び、状況に応じて経営陣その他の関係者に対する助言・提案、"
               "重要提案行為等を行う。")
    rec = lh.parse_xbrl_csv(_csv([
        (lh.E_TITLE, "大量保有報告書"),
        (lh.E_PURPOSE, purpose),
        (lh.E_RATIO, "0.0902"),
        (lh.E_RATIO_PREV, "0.0797"),
        (lh.E_N_JOINT, "1"),
        (lh.E_SHARES_HELD, "27590801"),
        (lh.E_SHARES_OUT, "305775520"),
        (lh.E_FILER_EDINET, "E33309"),
        (lh.E_FILER_NAME_JP, "3Dインベストメント・パートナーズ"),
        (lh.E_OBLIG_DATE, "2026-06-05"),
    ]))
    f = lh.fields_lvh(rec)
    assert f["report_class"] == "new"
    assert f["is_important_proposal"] is True
    assert abs(f["holding_ratio"] - 0.0902) < 1e-9
    assert abs(f["holding_ratio_prev"] - 0.0797) < 1e-9
    assert f["n_joint_holders"] == 1
    assert f["filer_edinet"] == "E33309"
    assert f["oblig_date"] == "2026-06-05"


def test_fields_lvh_pure_investment_not_important():
    rec = lh.parse_xbrl_csv(_csv([
        (lh.E_TITLE, "大量保有報告書"),
        (lh.E_PURPOSE, "純投資"),
        (lh.E_RATIO, "0.0531"),
    ]))
    f = lh.fields_lvh(rec)
    assert f["is_important_proposal"] is False
    assert f["report_class"] == "new"
    assert f["holding_ratio_prev"] is None      # 「－」/欠損は None


def test_fields_lvh_dash_ratio_is_none():
    rec = lh.parse_xbrl_csv(_csv([
        (lh.E_TITLE, "変更報告書No.2"),
        (lh.E_RATIO, "－"),
    ]))
    f = lh.fields_lvh(rec)
    assert f["report_class"] == "change"
    assert f["holding_ratio"] is None


def test_tag_filer_registry_hit_and_miss():
    hit = lh.tag_filer("エフィッシモ・キャピタル", "E11852")
    assert hit["in_registry"] is True
    assert hit["canonical_group"] == "effissimo"
    miss = lh.tag_filer("どこかの無関係な合同会社", "E99999")
    assert miss["in_registry"] is False
    assert miss["canonical_group"] is None
