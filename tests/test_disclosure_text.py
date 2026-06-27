"""開示テキスト抽出・前年比類似度（候補①テキストα Stage1）の純関数を検証。"""
import io
import zipfile

import pandas as pd

from invest_system.equities.disclosure_text import (
    extract_narrative,
    yoy_text_similarity,
    yoy_text_similarity_idf,
)

_HEADER = ["element_id", "item_name", "context_id", "rel_year", "consolidated",
           "period_type", "unit_id", "unit", "value"]


def _make_type5_zip(path, rows):
    """合成 type=5 ZIP（UTF-16・タブ区切り・先頭ヘッダ行）を書く。"""
    lines = ["\t".join(_HEADER)] + ["\t".join(r) for r in rows]
    data = ("\n".join(lines)).encode("utf-16")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("jpcrp_main.csv", data)


def test_extract_narrative_selects_mda_only(tmp_path):
    mda = "当期のわが国経済は緩やかな回復基調となり、当社の業績は増収増益となりました。" * 5
    boiler = "金融商品の状況に関する事項。デリバティブ取引は行わない方針であります。" * 5
    zp = tmp_path / "S1_5.zip"
    _make_type5_zip(zp, [
        ["jpcrp_cor:BizOverviewTextBlock", "業績等の概要 [テキストブロック]",
         "CurrentYearDuration", "0", "Consolidated", "Duration", "", "", mda],
        ["jpcrp_cor:FinInstrTextBlock", "金融商品関係、連結財務諸表 [テキストブロック]",
         "CurrentYearDuration", "0", "Consolidated", "Duration", "", "", boiler],
        ["jppfs_cor:NetSales", "売上高", "CurrentYearDuration", "0",
         "Consolidated", "Duration", "JPY", "円", "123456789"],
    ])
    out = extract_narrative(str(zp))
    assert "業績は増収増益" in out          # MD&A は採用
    assert "金融商品" not in out            # 定型注記は除外
    assert "123456789" not in out          # 数値ファクトは除外


def test_yoy_text_similarity_change_lowers_score():
    texts = pd.DataFrame([
        {"Code": "13010", "submitDateTime": pd.Timestamp("2020-06-20"),
         "text": "あいうえおかきくけこ" * 50},
        {"Code": "13010", "submitDateTime": pd.Timestamp("2021-06-20"),
         "text": "あいうえおかきくけこ" * 50},                 # 前年と同一 → ~1
        {"Code": "13010", "submitDateTime": pd.Timestamp("2022-06-20"),
         "text": "さしすせそたちつてと" * 50},                 # 大きく変化 → 低
        {"Code": "99990", "submitDateTime": pd.Timestamp("2021-06-20"),
         "text": "単発提出"},                                  # ペア無し
    ])
    sim = yoy_text_similarity(texts)
    assert len(sim) == 2                                       # 13010 の2ペアのみ
    by = sim.set_index("submitDateTime")["text_sim"]
    assert by[pd.Timestamp("2021-06-20")] > 0.99              # 同一＝高類似
    assert by[pd.Timestamp("2022-06-20")] < by[pd.Timestamp("2021-06-20")]  # 変化＝低下


def test_yoy_text_similarity_empty():
    assert yoy_text_similarity(pd.DataFrame(columns=["Code", "submitDateTime", "text"])).empty


def test_yoy_idf_change_lowers_score():
    texts = pd.DataFrame([
        {"Code": "A", "submitDateTime": pd.Timestamp("2020-06-20"),
         "text": "事業等のリスクは為替変動と原材料価格の高騰である。" * 8},
        {"Code": "A", "submitDateTime": pd.Timestamp("2021-06-20"),
         "text": "事業等のリスクは為替変動と原材料価格の高騰である。" * 8},   # 不変
        {"Code": "B", "submitDateTime": pd.Timestamp("2020-06-20"),
         "text": "対処すべき課題は人材確保と海外展開の推進である。" * 8},
        {"Code": "B", "submitDateTime": pd.Timestamp("2021-06-20"),
         "text": "対処すべき課題は新規事業の創出と構造改革の断行である。" * 8},  # 変化
    ])
    sim = yoy_text_similarity_idf(texts, min_corpus=1, min_df=1)
    assert len(sim) == 2                          # 2021 を後者とする2ペア
    a = sim[sim["Code"] == "A"]["text_sim"].iloc[0]
    b = sim[sim["Code"] == "B"]["text_sim"].iloc[0]
    assert a > b                                  # 不変 > 変化
