"""TOB 案件テーブル（チェーン集約＋本体抽出）の検証。ネットワーク不要。"""
import csv
import io

import pandas as pd

from invest_system.data.sources import edinet as ed
from invest_system.equities import tob_events as te


def _csv(rows: list[tuple]) -> str:
    """EDINET XBRL→CSV と同じ 9 列 TSV を生成（要素ID, 項目名, …, 値）。"""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t")
    w.writerow(["要素ID", "項目名", "コンテキストID", "相対年度", "連結・個別",
                "期間・時点", "ユニットID", "単位", "値"])
    for eid, val in rows:
        w.writerow([eid, "名", "c", "-", "-", "時点", "-", "", val])
    return buf.getvalue()


def test_extract_price_handles_fullwidth_and_commas():
    assert te.extract_price("株券普通株式１株につき、金338円新株予約権…") == 338
    assert te.extract_price("普通株式１株につき　金1,320円") == 1320
    assert te.extract_price("金７，６００円") == 7600          # 全角
    assert te.extract_price("該当事項なし") is None


def test_extract_dates_and_period():
    t = "買付け等の期間2026年４月27日(月)から2026年５月28日(木)まで"
    assert te.extract_dates(t) == ["2026-04-27", "2026-05-28"]
    assert te._period(t) == ("2026-04-27", "2026-05-28")
    assert te._period(None) == (None, None)
    # 届出当初ブロックは末尾に公告日が混入＝先頭2つ(開始/終了)を採る（公告日を拾わない）
    t3 = ("買付け等の期間2026年４月27日から2026年５月28日まで(20営業日)"
          "公告日2026年４月27日")
    assert te._period(t3) == ("2026-04-27", "2026-05-28")


def test_result_keyword_precedence():
    assert te._result_from_text("…下限以上となったので成立しました") == "成立"
    assert te._result_from_text("…下限に満たず不成立となりました") == "不成立"
    assert te._result_from_text("中立的な記述") is None


def test_result_by_number_comparison_without_keyword():
    # 実データの成立文は「成立」を含まず数値で述べる＝応募総数 vs 下限で判定
    succ = ("応募株券等の総数(929,650株)が買付予定数の下限(624,100株)以上となり、"
            "かつ上限を超えました")
    fail = "応募株券等の総数(100株)が買付予定数の下限(624,100株)に満たなかった"
    assert te._result_from_text(succ) == "成立"
    assert te._result_from_text(fail) == "不成立"
    # 別表現「応募株券等の数の合計」にも対応（実データ S100Y3C1 の文型）
    alt = ("応募株券等の数の合計（48,373,328株）が買付予定数の下限（26,115,700株）"
           "以上となりました")
    assert te._result_from_text(alt) == "成立"


def test_parse_xbrl_csv_roundtrip_and_multiline():
    text = _csv([
        (te.E_PRICE, "株券普通株式１株につき、金338円"),
        (te.E_RESULT, "応募が下限以上\nとなったので成立しました"),   # 改行入り
    ])
    rec = te.parse_xbrl_csv(text)
    assert rec[te.E_PRICE] == "株券普通株式１株につき、金338円"
    assert "成立" in rec[te.E_RESULT] and "\n" in rec[te.E_RESULT]  # 複数行を1セルに


def test_fields_240_250_270():
    rec240 = te.parse_xbrl_csv(_csv([
        (te.E_PRICE, "株券普通株式１株につき、金338円"),
        (te.E_ORIG_PERIOD, "買付け等の期間2026年４月27日から2026年５月28日まで"),
        (te.E_RATIO, "0.5010"),
    ]))
    f = te.fields_240(rec240)
    assert f["initial_price"] == 338
    assert (f["period_start"], f["period_end"]) == ("2026-04-27", "2026-05-28")
    assert abs(f["purchase_ratio_pct"] - 0.5010) < 1e-9

    rec250 = te.parse_xbrl_csv(_csv([
        (te.E_AMEND_TARGET, "S100Y4AZ"),
        (te.E_PRICE, "株券普通株式１株につき金7,600円"),
        (te.E_PERIOD, "買付け等の期間2026年５月15日から2026年６月25日まで"),
    ]))
    g = te.fields_250(rec250)
    assert g["amends_doc_id"] == "S100Y4AZ" and g["price"] == 7600
    assert g["period_end"] == "2026-06-25"

    rec270 = te.parse_xbrl_csv(_csv([
        (te.E_RESULT, "下限以上となり成立しました"),
        (te.E_TOR_PERIOD, "2026年４月27日から2026年５月28日まで"),
    ]))
    h = te.fields_270(rec270)
    assert h["result"] == "成立" and h["actual_period_end"] == "2026-05-28"


def _deal_listing() -> pd.DataFrame:
    recs = [
        # 案件1：240 起点＋250×2＋270＋290(対象者提出=secCode 持つ)
        {"docID": "D1", "ordinanceCode": "040", "docTypeCode": "240",
         "filerName": "買収者A", "edinetCode": "EAcq", "subjectEdinetCode": "ETgt",
         "submitDateTime": "2026-05-01 09:00"},
        {"docID": "D1a", "ordinanceCode": "040", "docTypeCode": "250",
         "parentDocID": "D1", "submitDateTime": "2026-05-10 10:00"},
        {"docID": "D1b", "ordinanceCode": "040", "docTypeCode": "250",
         "parentDocID": "D1", "submitDateTime": "2026-05-20 10:00"},
        {"docID": "D1r", "ordinanceCode": "040", "docTypeCode": "270",
         "parentDocID": "D1", "submitDateTime": "2026-06-05 10:00"},
        {"docID": "D1o", "ordinanceCode": "040", "docTypeCode": "290",
         "parentDocID": "D1", "secCode": "12340", "submitDateTime": "2026-05-02 10:00"},
        # 競合：同一対象 ETgt の別 240（窓内）
        {"docID": "D2", "ordinanceCode": "040", "docTypeCode": "240",
         "filerName": "買収者B", "subjectEdinetCode": "ETgt",
         "submitDateTime": "2026-05-15 09:00"},
        # 自社株TOB（府令050）＝案件に含めない
        {"docID": "S1", "ordinanceCode": "050", "docTypeCode": "240",
         "submitDateTime": "2026-05-03 09:00"},
    ]
    return ed.parse_documents(recs)


def test_build_deal_chains_structure_and_competing():
    deals = te.build_deal_chains(_deal_listing())
    assert set(deals["deal_id"]) == {"D1", "D2"}        # 050 は除外
    d1 = deals[deals["deal_id"] == "D1"].iloc[0]
    assert d1["n_amend"] == 2                            # 250×2
    assert d1["has_report"] and d1["n_opinion"] == 1
    assert d1["target_edinet"] == "ETgt"
    assert d1["target_sec"] == "12340"                  # 290(対象者)の secCode から復元
    assert not d1["has_withdrawal"]
    # 同一対象・窓内の 2 案件は競合フラグ
    assert bool(d1["competing"]) and bool(deals[deals["deal_id"] == "D2"].iloc[0]["competing"])


def test_build_deal_chains_empty():
    empty = ed.parse_documents([])
    out = te.build_deal_chains(empty)
    assert list(out["deal_id"]) == []
